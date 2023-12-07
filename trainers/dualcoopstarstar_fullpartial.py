import os.path as osp
import os
import copy
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision.models._utils import IntermediateLayerGetter
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .detr_part import DETRPart
from .hungarian_matcher import HungarianMatcher
from .loss_utils import get_match_loss_fns, get_opt_loss_fns
from .fixed_prompt_utils import FIXED_PROMPTS_DICT
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    #will embed the classnames alongside the prompt(s) and ensemble the embeddings
    def _init_fixed_prompt(self, cfg, classnames, clip_model):
        templates = FIXED_PROMPTS_DICT[cfg.TRAINER.DualCoOpStarStar.FIXED_PROMPT_KEY]
        texts = [[template.format(classname) for template in templates] for classname in classnames]
        texts = [clip.tokenize(texts_sub) for texts_sub in texts]
        with torch.no_grad():
            texts = torch.cat(texts).cuda()
            fixed_text_embeddings = copy.deepcopy(clip_model).cuda().encode_text(texts)
            fixed_text_embeddings = torch.reshape(fixed_text_embeddings, (len(classnames), len(templates), -1))
            fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)
            fixed_text_embeddings = torch.mean(fixed_text_embeddings, dim=1)
            fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)
            self.fixed_text_embeddings = fixed_text_embeddings

    #def _init_temperature(self):
    #    temperature = torch.tensor(3.91, dtype=dtype)  # 50
    #    self.temperature = nn.Parameter(temperature)

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        self.n_cls = n_cls
        self.prompt_mode = cfg.TRAINER.DualCoOpStarStar.PROMPT_MODE
        assert(self.prompt_mode in ['pos_only_fixed_prompt', 'pos_only_learnable_prompt', 'pos_and_neg_learnable_prompt'])
        if self.prompt_mode == 'pos_only_fixed_prompt':
            #self._init_temperature()
            self._init_fixed_prompt(cfg, classnames, clip_model)
            return

        n_ctx = cfg.TRAINER.DualCoOpStarStar.N_CTX
        ctx_init = cfg.TRAINER.DualCoOpStarStar.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            assert(False) #in the original code, this would lead to an error where the positive and negative prompts weren't defined
            ## use given words to initialize context vectors
            #ctx_init = ctx_init.replace("_", " ")
            #n_ctx = len(ctx_init.split(" "))
            #prompt = clip.tokenize(ctx_init)
            #with torch.no_grad():
            #    embedding = clip_model.token_embedding(prompt).type(dtype)
            #ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            #prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.DualCoOpStarStar.CSC:
                print("Initializing class-specific contexts")
                #ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_pos = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                if self.prompt_mode == 'pos_and_neg_learnable_prompt':
                    ctx_vectors_neg = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                #ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_pos = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                if self.prompt_mode == 'pos_and_neg_learnable_prompt':
                    ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype)

            #nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial negative context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        #self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        if self.prompt_mode == 'pos_and_neg_learnable_prompt':
            self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        # temperature = torch.tensor(4.6, dtype=dtype)  # 
        # temperature = torch.tensor(4.24, dtype=dtype)  # 70
        #self._init_temperature()
        #spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        #self.spatial_T = nn.Parameter(spatial_T)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # class agnostic token suffix
        prompts_nocls = [prompt_prefix + "."] * len(classnames)
        tokenized_prompts_nocls = torch.cat([clip.tokenize(p) for p in prompts_nocls])
        with torch.no_grad():
            embedding_nocls = clip_model.token_embedding(tokenized_prompts_nocls).type(dtype)
        self.register_buffer("token_suffix_nocls", embedding_nocls[:, 1 + n_ctx :, :])  # EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.DualCoOpStarStar.CLASS_TOKEN_POSITION

    #embeddings will already be computed, ensembled, and normalized!
    def _forward_fixed_prompt(self):
        return self.fixed_text_embeddings

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concatenated with cls word embeddings.
        """

        if self.prompt_mode == 'pos_only_fixed_prompt':
            return self._forward_fixed_prompt()

        #ctx = self.ctx
        ctx_pos = self.ctx_pos
        if self.prompt_mode == 'pos_and_neg_learnable_prompt':
            ctx_neg = self.ctx_neg

        if ctx_pos.dim() == 2:
            #ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            if self.prompt_mode == 'pos_and_neg_learnable_prompt':
                ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            #prompts = torch.cat(
            #    [
            #        prefix,  # (n_cls, 1, dim)
            #        ctx,     # (n_cls, n_ctx, dim)
            #        suffix,  # (n_cls, *, dim)
            #    ],
            #    dim=1,
            #)

            prompts_pos = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx_pos,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            if self.prompt_mode == 'pos_and_neg_learnable_prompt':
                if neg_prompt_wcls:
                    prompts_neg = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx_neg,     # (n_cls, n_ctx, dim)
                            suffix,  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )
                else:
                    prompts_neg = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim)
                            ctx_neg,     # (n_cls, n_ctx, dim)
                            suffix_nocls,  # (n_cls, *, dim)
                        ],
                        dim=1,
                    )


        elif self.class_token_position == "middle":
            assert(False) #this would only make the evidential prompt
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            assert(False) #this would only make the evidential prompt
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        if self.prompt_mode == 'pos_and_neg_learnable_prompt':
            return prompts_pos, prompts_neg
        else:
            assert(self.prompt_mode == 'pos_learnable_prompt')
            return prompts_pos


class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        if self.prompt_learner.prompt_mode != 'pos_only_fixed_prompt':
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        self.model = clip_model
        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)

        if self.cfg.DETR_CHEAT_MODE == 'none' and self.cfg.HUNGARIAN_CHEAT_MODE == 'none':
            num_queries = int(round(self.cfg.TRAINER.DualCoOpStarStar.NUM_QUERIES_PER_CLASS * len(classnames)))
        else:
            num_queries = len(classnames)

        self.detr_part = DETRPart(clip_model,
                                    num_queries,
                                    self.cfg.TRAINER.DualCoOpStarStar.NUM_ENCODER_LAYERS,
                                    self.cfg.TRAINER.DualCoOpStarStar.NUM_DECODER_LAYERS,
                                    self.cfg.TRAINER.DualCoOpStarStar.USE_POSITIONAL_EMBEDDING_IN_LOWER_PART,
                                    self.cfg.TRAINER.DualCoOpStarStar.USE_POSITIONAL_EMBEDDING_IN_UPPER_PART,
                                    detr_cheat_mode=self.cfg.DETR_CHEAT_MODE)

        #self.positional_embedding = self.model.visual.attnpool.positional_embedding[1::]
        #self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        #self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        #self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        #self.c_linear_bias = self.model.visual.attnpool.c_proj.bias
        #
        #self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        temperature_pos = torch.tensor(3, dtype=self.dtype)  # 50
        self.temperature_pos = nn.Parameter(temperature_pos)
        if self.prompt_learner.prompt_mode == 'pos_and_neg_learnable_prompt':
            temperature_neg = torch.tensor(3, dtype=self.dtype)  # 50
            self.temperature_neg = nn.Parameter(temperature_neg)
        else:
            assert(self.prompt_learner.prompt_mode in ['pos_only_learnable_prompt', 'pos_only_fixed_prompt'])
            bias_for_sigmoid = torch.tensor(-6, dtype=self.dtype)
            self.bias_for_sigmoid = nn.Parameter(bias_for_sigmoid)

        #temperature_wta = torch.tensor(5.3, dtype=self.dtype)  # 50
        #self.temperature_wta = nn.Parameter(temperature_wta)

    def get_learnable_model(self):
        misc_params_dict = {}
        if self.cfg.TRAINER.DualCoOpStarStar.TEMPERATURE_IS_LEARNABLE:
            misc_params_dict['temperature_pos'] = self.temperature_pos

        if self.prompt_learner.prompt_mode == 'pos_and_neg_learnable_prompt':
            if self.cfg.TRAINER.DualCoOpStarStar.TEMPERATURE_IS_LEARNABLE:
                misc_params_dict['temperature_neg'] = self.temperature_neg
        else:
            assert(self.prompt_learner.prompt_mode in ['pos_only_fixed_prompt', 'pos_only_learnable_prompt'])
            misc_params_dict['bias_for_sigmoid'] = self.bias_for_sigmoid

        misc_params = nn.ParameterDict(misc_params_dict)
        learnable_model_dict = {'misc_params' : misc_params, 'decoder_part' : self.detr_part.decoder_part, 'prompt_learner' : self.prompt_learner}
        learnable_model = nn.ModuleDict(learnable_model_dict)
        return learnable_model

    def encode_image(self, x):
        def stem(x):
            for conv, bn in [(self.visual_encoder.conv1, self.visual_encoder.bn1), \
                (self.visual_encoder.conv2, self.visual_encoder.bn2), (self.visual_encoder.conv3, self.visual_encoder.bn3)]:
                x = self.visual_encoder.relu(bn(conv(x)))
            x = self.visual_encoder.avgpool(x)
            return x

        x = x.type(self.visual_encoder.conv1.weight.dtype)
        x = stem(x)
        x = self.visual_encoder.layer1(x)
        x = self.visual_encoder.layer2(x)
        x = self.visual_encoder.layer3(x)
        x = self.visual_encoder.layer4(x)
        return x
    
    def forward(self, image):
        #run resnet backbone (modulo final attention layer) to get patch tokens
        #not sure why the original code didn't make this torch.no_grad()...
        with torch.no_grad():
            patch_tokens = self.encode_image(image)
        
        (batch_size, embed_dim, H, W) = patch_tokens.shape

        #run DETR part
        query_tokens = self.detr_part(patch_tokens) #(batch_size, num_queries, output_dim)
        query_tokens = query_tokens / query_tokens.norm(dim=-1, keepdim=True)
        assert(len(query_tokens.shape) == 3)
        assert(query_tokens.shape[0] == batch_size)
        num_queries, output_dim = query_tokens.shape[-2:]
        
        #text part
        if self.prompt_learner.prompt_mode == 'pos_only_fixed_prompt':
            text_features_pos = self.prompt_learner() #already normalized in this case
        elif self.prompt_learner.prompt_mode == 'pos_only_learnable_prompt':
            prompts_pos = self.prompt_learner()
            text_features_pos = self.text_encoder(prompts_pos, self.tokenized_prompts)
            text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
        elif self.prompt_learner.prompt_mode == 'pos_and_neg_learnable_prompt':
            prompts_pos, prompts_neg = self.prompt_learner()
            text_features_pos = self.text_encoder(prompts_pos, self.tokenized_prompts)
            text_features_neg = self.text_encoder(prompts_neg, self.tokenized_prompts)
            text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
            text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)
        else:
            assert(False)

        #compute logits that could be plugged directly into a sigmoid
        #this will be DenseCLIP's final output
        #should have shape (batch_size, num_queries, num_classes)
        assert(text_features_pos.shape == (self.prompt_learner.n_cls, output_dim))
        if self.prompt_learner.prompt_mode in ['pos_only_fixed_prompt', 'pos_only_learnable_prompt']:
            logits_pos = self.temperature_pos.exp() * query_tokens @ text_features_pos.t()
            logits = logits_pos + self.bias_for_sigmoid
        else:
            assert(self.prompt_learner.prompt_mode == 'pos_and_neg_learnable_prompt')
            logits_pos = self.temperature_pos.exp() * query_tokens @ text_features_pos.t()
            logits_neg = self.temperature_neg.exp() * query_tokens @ text_features_neg.t()
            logits = logits_pos - logits_neg

        return logits


@TRAINER_REGISTRY.register()
class DualCoOpStarStar_FullPartial(TrainerX):
    @torch.no_grad()
    def test(self, split=None):

        """A generic testing pipeline."""
        assert(self.evaluator.is_for_dualcoopstarstar)

        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
            print("Do evaluation on {} set".format(split))
        else:
            split = "test"
            data_loader = self.test_loader
            print("Do evaluation on test set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            if self.cfg.COMPUTE_RANDOM_CHANCE:
                logits_hungarian = logits_max = torch.tensor(np.random.uniform(0,1,size=label.shape))
            else:
                logits_hungarian, logits_max = self.model_inference(input)

            self.evaluator.process({'hungarian' : logits_hungarian, 'max' : logits_max}, label)

        results = self.evaluator.evaluate()

        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_filename = os.path.join(results_dir, 'results-%03d.pkl'%(self.epoch + 1))
        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)

        for k, v in results.items():
            if k != 'mAP':
                continue

            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results['mAP']

    #this should only be used for testing!
    #it will actually give you logits for each class!
    #these logits can be plugged directly into sigmoid to get probs!
    #it'll return a DIFFERENT shape than self.model(input)!
    #specifically, it'll return logits_hungarian, logits_max, which are two different ways of reducing out the query dimension
    #logits_hungarian ensures that each class picks a different query (and obviously no two queries pick the same class)
    #logits_max just picks the max query for each class independently
    def model_inference(self, input):
        assert(self.cfg.TRAIN.LOSSFUNC == 'hungarian')
        with torch.no_grad(): #(why wouldn't they normally do this?)
            logits = self.model(input)
            logits_hungarian, matches = self.hungarian_matcher.do_inference(logits)
            logits_max, max_indices = torch.max(logits, dim=1, keepdim=False)
            assert(logits_hungarian.shape == (logits.shape[0], logits.shape[2]))
            assert(logits_max.shape == (logits.shape[0], logits.shape[2]))
            return logits_hungarian, logits_max

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DualCoOpStarStar.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        if self.cfg.COMPUTE_RANDOM_CHANCE:
            return

        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        #do hungarian first, because I think model-loading might take a long time
        if self.cfg.TRAIN.LOSSFUNC == 'hungarian':
            pos_match_loss_fn, neg_match_loss_fn = get_match_loss_fns(self.cfg.TRAIN.MATCH_LOSS_FN_TYPE)
            pos_opt_loss_fn, neg_opt_loss_fn = get_opt_loss_fns(self.cfg.TRAIN.OPT_LOSS_FN_TYPE)
            self.hungarian_matcher = HungarianMatcher(pos_match_loss_fn, neg_match_loss_fn, pos_opt_loss_fn, neg_opt_loss_fn, hungarian_cheat_mode=self.cfg.HUNGARIAN_CHEAT_MODE, loss_averaging_mode=self.cfg.TRAIN.LOSS_AVERAGING_MODE)
        else:
            assert(False)

        print('|||||||||||||||||||||||||||||||||||||| Building Caption_dual')

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DualCoOpStarStar.PREC == "fp32" or cfg.TRAINER.DualCoOpStarStar.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model = DenseCLIP(cfg, classnames, clip_model)
        self.learnable_model = self.model.get_learnable_model() #this is a Module that only contains the learnable parts

        print("Turning off ALL gradients...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        print("And turning them back on for the learnable parts!")
        for name, param in self.learnable_model.named_parameters():
            param.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.learnable_model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.learnable_model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("learnable_model", self.learnable_model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DualCoOpStarStar.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.DualCoOpStarStar.PREC
        if prec == "amp":
            assert(False) #this does NOT compute a multilabel loss!
            with autocast():
                output_g, output, _, _ = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            logits = self.model(image)
            if self.cfg.TRAIN.LOSSFUNC == 'hungarian':
                loss, matches = self.hungarian_matcher(logits, label)
            else:
                assert(False)
            
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            for k in sorted(state_dict.keys()):
                if 'token_prefix' in k or 'token_suffix' in k:
                    print('found fixed token vector key "%s", deleting...'%(k))
                    del state_dict[k]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
