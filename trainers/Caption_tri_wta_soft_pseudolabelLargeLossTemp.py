import os.path as osp
import os
import glob
import math
import numpy as np
import pickle

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

from .utils import soft_cross_entropy, softmax_sigmoid_BCEloss, \
    norm_logits_BCEloss, sigmoid_focal_loss, sigmoid_ASL_loss, dualcoop_loss,dualcoop_softmax_loss
from .loss_utils import AsymmetricLoss_softmax_partial_fn
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
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.Caption.N_CTX
        ctx_init = cfg.TRAINER.Caption.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.Caption.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_pos = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_pos = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f'Initial negtive context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized
        
        # temperature = torch.tensor(4.6, dtype=dtype)  # 
        # temperature = torch.tensor(4.24, dtype=dtype)  # 70
        temperature = torch.tensor(3.91, dtype=dtype)  # 50
        self.temperature = nn.Parameter(temperature)
        spatial_T = torch.tensor(3.0, dtype=dtype)  # 20
        self.spatial_T = nn.Parameter(spatial_T)

        if cfg.TRAINER.Caption.USE_BIAS:
            bias = torch.tensor(0.0, dtype=dtype) #trust me bro, 0 is the most reasonable default here
            self.bias = nn.Parameter(bias)

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

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.Caption.CLASS_TOKEN_POSITION

    def forward(self, neg_prompt_wcls=True):
        """
        Returns current learned ctx embeddings, concated with cls word embeddings.
        """
        ctx = self.ctx
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_pos = ctx_pos.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx_neg = ctx_neg.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        suffix_nocls = self.token_suffix_nocls

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

            prompts_pos = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx_pos,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

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

        return prompts, prompts_pos, prompts_neg, self.temperature, self.spatial_T


class DenseCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model, return_interm_layers=False):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)

        self.model = clip_model
        self.return_interm_layers = return_interm_layers
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.visual_encoder = IntermediateLayerGetter(self.model.visual, return_layers)
        self.positional_embedding = self.model.visual.attnpool.positional_embedding[1::]
        self.v_linear_weight = self.model.visual.attnpool.v_proj.weight
        self.v_linear_bias = self.model.visual.attnpool.v_proj.bias
        self.c_linear_weight = self.model.visual.attnpool.c_proj.weight
        self.c_linear_bias = self.model.visual.attnpool.c_proj.bias
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        temperature_pos = torch.tensor(3, dtype=self.dtype)  # 50
        self.temperature_pos = nn.Parameter(temperature_pos)
        temperature_neg = torch.tensor(3, dtype=self.dtype)  # 50
        self.temperature_neg = nn.Parameter(temperature_neg)
        temperature_wta = torch.tensor(5.3, dtype=self.dtype)  # 50
        self.temperature_wta = nn.Parameter(temperature_wta)

    
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
    
    def forward(self, image, norm=True):
        image_feat = self.encode_image(image)
        b, c, h, w = image_feat.shape
        x = image_feat.reshape(b, c, h * w).permute(2, 0, 1)

        x = F.linear(x, self.v_linear_weight, self.v_linear_bias)
        x = F.linear(x, self.c_linear_weight, self.c_linear_bias)
        image_features = x
        
        image_feature_, _ = self.model.visual.attnpool(image_feat, if_pos=False)
        # ===============================================================

        prompts, prompts_pos, prompts_neg, _, spatial_T = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features_pos = self.text_encoder(prompts_pos, tokenized_prompts)
        text_features_neg = self.text_encoder(prompts_neg, tokenized_prompts)
    
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_feature_ = image_feature_ / image_feature_.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
        text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

        logit_scale_pos = self.temperature_pos.exp() 
        logit_scale_neg = self.temperature_neg.exp() 
        logit_scale_wta = self.temperature_wta.exp() 
        logits = image_features @ text_features.t()    #  HW * B * C,  cls * C,  HW * B * cls
        logits_pos = logit_scale_pos * image_features @ text_features_pos.t()    #  HW * B * C,  cls * C,  HW * B * cls
        logits_neg = logit_scale_neg * image_features @ text_features_neg.t()    #  HW * B * C,  cls * C,  HW * B * cls

        map_pos = image_features @ text_features_pos.t()

        if self.cfg.MODE == 'pos200':
            w_pos = 1+(map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
            logits_pos *= w_pos
        elif self.cfg.MODE == 'pos_test':
            w_pos = (map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
            logits_pos *= w_pos

        elif self.cfg.MODE == 'pos_norm':
            w_pos = (map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)

            w_pos = w_pos/w_pos.max(dim=2,keepdim=True)[0]
            logits_pos *= w_pos


        elif self.cfg.MODE == 'pos_norm_det':
            w_pos = (map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)

            w_pos = w_pos/w_pos.max(dim=2,keepdim=True)[0]
            logits_pos *= w_pos

            spatial_T =  torch.tensor(3.0, dtype=self.dtype)


        elif self.cfg.MODE == 'pos_neg':

            w_pos = 1+(map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
            logits_pos *= w_pos

            w_neg = 1+(map_pos*200*map_pos.min(dim=2,keepdim=True)[0]).softmax(dim=2)
            logits_neg *= w_neg

        elif self.cfg.MODE == 'pos_neg_test':

            w_pos = (map_pos*200*map_pos.max(dim=2,keepdim=True)[0]).softmax(dim=2)
            logits_pos *= w_pos

            w_neg = (map_pos*200*map_pos.min(dim=2,keepdim=True)[0]).softmax(dim=2)
            logits_neg *= w_neg

        else:
            raise ValueError



        
        logits_g = image_feature_ @ text_features.t()    # B * C,  cls * C,  B * cls
        logits_pos_g = logit_scale_pos * image_feature_ @ text_features_pos.t()    # B * C,  cls * C,  B * cls
        logits_neg_g = logit_scale_neg * image_feature_ @ text_features_neg.t() 

        prob_ = torch.nn.functional.softmax(logits * spatial_T.exp(), dim=0)
        logits_pos = torch.sum(logits_pos * prob_, dim=0)
        logits_neg = torch.sum(logits_neg * prob_, dim=0)
        
  

        logits_ = torch.cat([torch.unsqueeze(logits_neg,1), torch.unsqueeze(logits_pos,1)], dim=1)
        logits_g = torch.cat([torch.unsqueeze(logits_neg_g,1), torch.unsqueeze(logits_pos_g,1)], dim=1)

        #collapse into a single logit
        logits_ = logits_[:,1,:] - logits_[:,0,:]
        logits_g = logits_g[:,1,:] - logits_g[:,0,:]
        
        if self.cfg.TRAINER.Caption.USE_BIAS:
            logits_ = logits_ + self.prompt_learner.bias #keep it in prompt_learner so it'll be learnable

        return logits_g, logits_, image_features, text_features

        
def softlogits2siglogits(softlogits):
    #how to turn softmax logits into sigmoid logits?
    #probs[i] = exp(-softlogits[i]) / sum_j exp(-softlogits[j])
    #probs[i] = 1 / (1 + exp(-siglogits[i]))
    #...
    #siglogits[i] = log(exp(softlogits[i]) / (sum_j exp(softlogits[j]) - exp(softlogits[i])))
    #             = softlogits[i] - log(sum_j exp(softlogits[j]) - exp(softlogits[i]))
    #best way to make it numerically stable is probably just to subtract the max of softlogits
    softlogits = softlogits - torch.max(softlogits, 1, keepdim=True)[0]
    sumexps = torch.sum(torch.exp(softlogits), 1, keepdim=True)
    siglogits = softlogits - torch.log(sumexps - torch.exp(softlogits))
    return siglogits


@TRAINER_REGISTRY.register()
class Caption_tri_wta_soft_pseudolabelLargeLossTemp(TrainerX):
    ''' This inherits from TrainerX, however it will NOT look at the label during train-time! '''

    @torch.no_grad()
    def eval_training_pseudolabels(self):
        #FIXME: this looks at ALL examples, i.e. it ignores pseudolabel_weights
        #you'll have to make it look at weights later
        print('eval_training_pseudolabels...')
        assert(self.evaluator.is_for_dualcoopstarstar)
        self.dm.train_loader_x_complete.dataset.skip_image = True
        pseudolabel_filenames = sorted(glob.glob(osp.join(self.output_dir, 'pseudolabels.pth.tar-*')))
        pseudolabel_results = {}
        for pseudolabel_filename in tqdm(pseudolabel_filenames):
            self.evaluator.reset()
            checkpoint = torch.load(pseudolabel_filename, map_location='cpu')
            pseudolabel_probs = checkpoint['pseudolabel_probs']
            t = 0
            for batch in self.dm.train_loader_x_complete:
                label = batch['label']
                assert('img' not in batch)
                assert(torch.all((label == 1) | (label == -1)).item())
                idx = batch['idx']
                probs_batch = pseudolabel_probs[idx,:]
                
                #in this case, pseudolabels are 0 or 1, not logits, but we can still compute an mAP that's hopefully meaningful
                self.evaluator.process({'default' : probs_batch}, label)
                if t % 10 == 0:
                    print('t=%d'%(t))

                t += 1

            res = self.evaluator.evaluate(report_individual_classes=True)
            assert(checkpoint['epoch'] not in pseudolabel_results)
            pseudolabel_results[checkpoint['epoch']] = res
            print('Epoch %s training pseudolabel mAP = %s'%(str(checkpoint['epoch']), str(res['mAP'])))

        pseudolabel_results_filename = osp.join(self.output_dir, 'training_pseudolabel_results.pkl')
        with open(pseudolabel_results_filename, 'wb') as f:
            pickle.dump(pseudolabel_results, f)

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

        if self.cfg.COMPUTE_ZSCLIP:
            text_embeddings = self.get_initializing_text_embeddings()

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            if self.cfg.COMPUTE_RANDOM_CHANCE:
                logits = torch.tensor(np.random.uniform(0,1,size=label.shape))
            elif self.cfg.COMPUTE_ZSCLIP:
                image_embeddings = self.clip_model.encode_image(input)
                assert(len(image_embeddings.shape) == 2)
                assert(image_embeddings.shape[0] == input.shape[0])
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
                softlogits = self.clip_model.logit_scale.exp() * image_embeddings @ text_embeddings.t()
                logits = softlogits2siglogits(softlogits)
            else:
                _, logits, _, _ = self.model_inference(input)

            self.evaluator.process({'default' : logits}, label)

        results = self.evaluator.evaluate()

        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        if self.is_after_train:
            results_filename = os.path.join(results_dir, 'results-after_train.pkl')
        else:
            results_filename = os.path.join(results_dir, 'results-%03d.pkl'%(self.epoch + 1))

        with open(results_filename, 'wb') as f:
            pickle.dump(results, f)

        for k, v in results.items():
            if k != 'mAP':
                continue

            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        return results['mAP']

    def check_cfg(self, cfg):
        assert cfg.TRAINER.Caption.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        if self.cfg.COMPUTE_RANDOM_CHANCE or self.cfg.EVAL_TRAINING_PSEUDOLABELS:
            return
        
        cfg = self.cfg
        
        classnames = self.dm.dataset.classnames
        print('|||||||||||||||||||||||||||||||||||||| Building Caption_dual')

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.Caption.PREC == "fp32" or cfg.TRAINER.Caption.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        if self.cfg.COMPUTE_ZSCLIP:
            self.clip_model = clip_model
            self.clip_model.to(self.device)
            return

        print("Building custom CLIP")
        # self.model = CustomCLIP(cfg, classnames, clip_model)
        self.model = DenseCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.Caption.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def before_train(self):
        super().before_train()
        self.initialize_pseudolabels()

    def after_epoch(self):
        super().after_epoch()
        self.save_pseudolabels(self.epoch, self.output_dir)

    def save_pseudolabels(self, epoch, output_dir, is_init=False):
        print('save_pseudolabels...')
        checkpoint = {'epoch' : (epoch + 1 if not is_init else 'init'), 'pseudolabel_probs':self.pseudolabel_probs,'pseudolabel_weights':self.pseudolabel_weights}
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_filename = osp.join(output_dir, 'pseudolabels.pth.tar-' + ('%03d'%(epoch + 1) if not is_init else 'init'))
        torch.save(checkpoint, checkpoint_filename)

    #get text embeddings for initializing the pseudolabels
    #this would be the place to do ensembling
    #we expect the output to be normalized
    #shape (num_classes, embed_dim)
    def get_initializing_text_embeddings(self):
        templates = FIXED_PROMPTS_DICT[self.cfg.TRAIN.PSEUDOLABEL_INIT_PROMPT_KEY]
        classnames = self.dm.dataset.classnames
        texts = [[template.format(classname) for template in templates] for classname in classnames]
        texts = [clip.tokenize(texts_sub) for texts_sub in texts]
        with torch.no_grad():
            texts = torch.cat(texts).to(self.device)
            if self.cfg.COMPUTE_ZSCLIP:
                fixed_text_embeddings = self.clip_model.encode_text(texts)
            else:
                fixed_text_embeddings = self.model.model.encode_text(texts)

            fixed_text_embeddings = torch.reshape(fixed_text_embeddings, (len(classnames), len(templates), -1))
            fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)
            fixed_text_embeddings = torch.mean(fixed_text_embeddings, dim=1)
            fixed_text_embeddings = fixed_text_embeddings / fixed_text_embeddings.norm(dim=-1, keepdim=True)
            return fixed_text_embeddings

    def initialize_pseudolabels_top1_positive(self):
        #do initialization (set self.init_pseudolabel_probs)
        num_samples = len(self.dm.dataset.train_x)
        self.init_pseudolabel_probs = torch.zeros((num_samples,len(self.dm.dataset.classnames)),dtype=self.model.dtype,device=self.device)
        text_embeddings = self.get_initializing_text_embeddings()
        for batch in tqdm(self.dm.train_loader_x_complete):
            images = batch['img'].to(self.device)
            idx = batch['idx'].to(self.device)
            image_embeddings = self.model.model.encode_image(images)
            assert(len(image_embeddings.shape) == 2)
            assert(image_embeddings.shape[0] == images.shape[0])
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            scores = image_embeddings @ text_embeddings.t()
            winners = torch.argmax(scores, dim=1)
            self.init_pseudolabel_probs[idx, winners] = 1.0

        #do observation (set self.observed_mask)
        if self.cfg.TRAIN.PSEUDOLABEL_OBSERVATION_METHOD == 'observe_positives':
            self.observed_mask = self.init_pseudolabel_probs.detach().clone()
        elif self.cfg.TRAIN.PSEUDOLABEL_OBSERVATION_METHOD == 'observe_nothing':
            self.observed_mask = torch.zeros_like(self.init_pseudolabel_probs)
        else:
            assert(False)

        #weights and bookkeeping
        self.pseudolabel_weights = torch.ones_like(self.init_pseudolabel_probs)
        self.pseudolabel_probs = self.init_pseudolabel_probs.detach().clone()

    #should call this before any training
    def initialize_pseudolabels(self):
        print('initialize_pseudolabels...')
        print(self.device)
        with torch.no_grad(): #NOTE: if your initialization method requires gradients, you'll have to modify this
            if self.cfg.TRAIN.PSEUDOLABEL_INIT_METHOD == 'top1_positive':
                self.initialize_pseudolabels_top1_positive()
            else:
                assert(False)

        self.save_pseudolabels(None, self.output_dir, is_init=True)

    def should_freeze_pseudolabels(self):
        #special case where they should always be frozen
        if self.cfg.TRAIN.MAX_EPOCH_FOR_DELTA_REL == 0 or self.cfg.TRAIN.DELTA_REL == 0.0:
            assert(self.cfg.TRAIN.MAX_EPOCH_FOR_DELTA_REL == 0 and self.cfg.TRAIN.DELTA_REL == 0.0)
            return True

        return self.epoch < self.cfg.TRAIN.PSEUDOLABEL_FREEZE_DURATION

    #NOTE: This will have side effects!
    #output should be logits
    #This will update self.pseudolabel_probs (which is just for bookkeeping at this point)
    #and self.pseudolabel_weights (which is always 1 at this point but could be for bookkeeping later)
    def compute_loss_LargeLossTemp(self, idx, output):
        #get loss_matrix and corrected_loss_matrix
        #labels should be equal to init_pseudolabel_probs (trust me, just think of init_pseudolabel_probs as the "assumed" labels)
        #then just 1 - labels to get corrected_loss_matrix
        #it's okay to flip observed losses, they'll get zeroed out before being used to select the topK losses
        #(when you put ones in self.observed_mask, you're doing that zeroing-out)
        labels = self.init_pseudolabel_probs[idx,:]
        loss_matrix = F.binary_cross_entropy_with_logits(output, labels, reduction='none')
        corrected_loss_matrix = F.binary_cross_entropy_with_logits(output, 1 - labels, reduction='none')

        #decide whether to freeze pseudolabels
        if self.should_freeze_pseudolabels():
            self.pseudolabel_probs[idx,:] = self.init_pseudolabel_probs[idx,:]
            return torch.mean(torch.sum(self.pseudolabel_weights[idx,:] * loss_matrix, dim=-1))

        #figure out threshold
        #you should use something like (delta_rel=0.2, max_epoch_for_delta_rel=9)
        #OR, (delta_rel=0.04, max_epoch_for_delta_rel=49)
        top_prop = min(self.epoch, self.cfg.TRAIN.MAX_EPOCH_FOR_DELTA_REL) * self.cfg.TRAIN.DELTA_REL / 100.0
        k = math.ceil(top_prop * output.shape[0] * output.shape[1])
        unobserved_loss = loss_matrix * (1 - self.observed_mask[idx,:])
        topk = torch.topk(unobserved_loss.flatten(), k)
        topk_lossvalue = topk.values[-1]

        #apply threshold to flip some of the unobserved labels
        final_loss_matrix = torch.where(unobserved_loss < topk_lossvalue, loss_matrix, corrected_loss_matrix)
        loss = torch.mean(torch.sum(self.pseudolabel_weights[idx,:] * loss_matrix, dim=-1))
        pseudolabel_probs = torch.where(unobserved_loss < topk_lossvalue, labels, 1 - labels)
        self.pseudolabel_probs[idx,:] = pseudolabel_probs
        return loss

    def forward_backward(self, batch):
        image, idx = self.parse_batch_train(batch)

        prec = self.cfg.TRAINER.Caption.PREC
        if prec == "amp":
            assert(False) #this doesn't support multilabel loss
            with autocast():
                output_g, output, _, _ = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_g, output, _, _ = self.model(image)
            assert(self.cfg.TRAIN.PSEUDOLABEL_UPDATE_MODE == 'LargeLossTemp')
            loss = self.compute_loss_LargeLossTemp(idx, output)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            # "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    #return input, idx
    #note that we do NOT look at ground-truth labels!
    def parse_batch_train(self, batch):
        input = batch["img"]
        input = input.to(self.device)
        idx = batch["idx"]
        idx = idx.to(self.device)
        return input, idx

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
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
