import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .detr_transformer import DETRTransformer


def interpolate_pos_encoding(positional_embedding, dim, h, w):
    positional_embedding_ = positional_embedding.unsqueeze(0)
    class_pos_embed = positional_embedding_[:, :1]
    patch_pos_embed = positional_embedding_[:, 1:]
    N = patch_pos_embed.shape[1]
    if h*w == N and h == w:
        return class_pos_embed[0], patch_pos_embed[0]

    h0 = h
    w0 = w
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
        mode='bicubic',
    ) #1CHW now
    assert(patch_pos_embed.shape[-2:] == (h, w))
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim) #1(HW)C now
    return class_pos_embed[0], patch_pos_embed[0]


class CustomAttentionPool2d(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.positional_embedding = clip_model.visual.attnpool.positional_embedding
        self.k_proj = clip_model.visual.attnpool.k_proj
        self.q_proj = clip_model.visual.attnpool.q_proj
        self.v_proj = clip_model.visual.attnpool.v_proj
        self.c_proj = clip_model.visual.attnpool.c_proj
        self.num_heads = clip_model.visual.attnpool.num_heads

    #query_tokens should have dimension (batch_size, num_queries, embed_dim)
    #patch_tokens should have dimension (batch_size, embed_dim, H, W)
    #output will have dimension (batch_size, num_queries, output_dim)
    #if_pos controls if we interpolate and use the positional embedding, or don't use it at all
    #This is because the original RN101 has a 7x7 positional embedding (for 224x224 images), but we're running it on 448x448 images (so probably 14x14 embedding)
    def forward(self, query_tokens, patch_tokens, if_pos=True):
        H, W = patch_tokens.shape[-2:]
        patch_tokens = patch_tokens.flatten(start_dim=2).permute(2, 0, 1) #(HW, batch_size, embed_dim)
        query_tokens = query_tokens.permute(1, 0, 2) #(num_queries, batch_size, embed_dim)
        if if_pos:
            class_pos_embed, patch_pos_embed = interpolate_pos_encoding(self.positional_embedding, patch_tokens.shape[-1], H, W)
            assert(patch_pos_embed.shape == (patch_tokens.shape[0], patch_tokens.shape[2]))
            assert(class_pos_embed.shape == (1, query_tokens.shape[2]))
            patch_tokens = patch_tokens + patch_pos_embed[:, None, :].to(patch_tokens.dtype)
            query_tokens = query_tokens + class_pos_embed[:, None, :].to(query_tokens.dtype)

        query_tokens, _ = F.multi_head_attention_forward(
            query=query_tokens, key=patch_tokens, value=patch_tokens,
            embed_dim_to_check=patch_tokens.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        query_tokens = query_tokens.permute(1, 0, 2) #(batch_size, num_queries, output_dim)
        return query_tokens


''' this can optionally include an "encoder" that sits above the image backbone '''
''' at the very least it'll have a "decoder" that sits between the backbone (and optional encoder) and the CustomAttentionPool2d '''
''' this class should contain ALL of the learnable params, including any projections before or after the "decoder" (and optional encoder) '''
class DecoderPart(nn.Module):
    #num_queries, num_encoder_layers, num_decoder_layers, use_positional_embedding are hyperparams that we plan to search over
    #if use_positional_embedding is True, then we'll obtain it by stealing the one from base_positional_embedding (without making it learnable) and putting it through the same input projection as the patch_tokens
    #note that we also use base_positional_embedding to figure out the size we'll need for the input and output projections
    def __init__(self, num_queries, num_encoder_layers, num_decoder_layers, use_positional_embedding, base_positional_embedding, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, activation='relu', normalize_before=False, return_intermediate_dec=True):
        super().__init__()
        self.inout_dim = base_positional_embedding.shape[-1]
        self.use_positional_embedding = use_positional_embedding
        if self.use_positional_embedding:
            self.base_positional_embedding = base_positional_embedding.data.cuda() #the .data part keeps it from being a parameter of DecoderPart

        self.input_proj = nn.Linear(self.inout_dim, d_model)
        self.output_proj = nn.Linear(d_model, self.inout_dim)
        self.query_tokens = nn.Embedding(num_queries, d_model)
        self.transformer = DETRTransformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
                 num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
                 activation=activation, normalize_before=normalize_before, return_intermediate_dec=return_intermediate_dec)

    #patch_tokens should have dimension (batch_size, inout_dim, H, W)
    #output will have dimension (batch_size, num_queries, inout_dim)
    def forward(self, patch_tokens):
        H, W = patch_tokens.shape[-2:]
        patch_tokens = patch_tokens.flatten(start_dim=2).permute(0, 2, 1) #(batch_size, H*W, inout_dim)
        patch_tokens = self.input_proj(patch_tokens) #(batch_size, H*W, d_model)
        if self.use_positional_embedding:
            _, pos_embed = interpolate_pos_encoding(self.base_positional_embedding, self.inout_dim, H, W) #(H*W, inout_dim)
            pos_embed = self.input_proj(pos_embed) #(H*W, d_model)
        else:
            pos_embed = None

        output = self.transformer(patch_tokens, self.query_tokens.weight, pos_embed) #(*, batch_size, num_queries, d_model)
        output = output[-1] #(batch_size, num_queries, d_model) (sorry, we're not doing auxiliary supervision, at least not for now...)
        output = self.output_proj(output) #(batch_size, num_queries, inout_dim)
        return output


''' this is the main thing '''
''' it assumes that you've already applied the image backbone (not including the AttentionPool2d layer at the top) '''
''' it will use DecoderPart to do all the learnable stuff and output query_tokens '''
''' then, it will give those query_tokens to CustomAttentionPool2d (alongside the same patch_tokens that went into DecoderPart) to get the final query_tokens '''
class DETRPart(nn.Module):
    def __init__(self,clip_model,num_queries,num_encoder_layers,num_decoder_layers,use_positional_embedding_in_lower_part,use_positional_embedding_in_upper_part,detr_cheat_mode='none'):
        super().__init__()
        self.decoder_part = DecoderPart(num_queries,num_encoder_layers,num_decoder_layers,use_positional_embedding_in_lower_part,clip_model.visual.attnpool.positional_embedding)
        self.detr_cheat_mode = detr_cheat_mode
        self.if_pos = use_positional_embedding_in_upper_part
        if self.detr_cheat_mode == 'none':
            self.custom_attnpool = CustomAttentionPool2d(clip_model)
            self.custom_attnpool.train(clip_model.visual.attnpool.training) #basically make this behave as if we had just used attnpool itself
        else:
            assert(self.detr_cheat_mode == 'copies_of_clip_query')
            self.custom_attnpool = CustomAttentionPool2d(clip_model)
            self.custom_attnpool.train(clip_model.visual.attnpool.training) #basically make this behave as if we had just used attnpool itself
            self.num_queries = num_queries

    #patch_tokens should have dimension (batch_size, embed_dim, H, W)
    #output will have dimension (batch_size, num_queries, output_dim)
    #(FIXME: pretty sure embed_dim is 2048, but double-check that output_dim is something like 512)
    def forward(self, patch_tokens):
        if self.detr_cheat_mode == 'none':
            query_tokens = self.decoder_part(patch_tokens)
            query_tokens = self.custom_attnpool(query_tokens, patch_tokens, if_pos=self.if_pos)
            return query_tokens
        else:
            assert(self.detr_cheat_mode == 'copies_of_clip_query')
            query_tokens = torch.mean(patch_tokens.flatten(start_dim=2).permute(0,2,1), dim=1, keepdim=True)
            assert(query_tokens.shape == (patch_tokens.shape[0], 1, patch_tokens.shape[1]))
            query_tokens = torch.tile(query_tokens, (1, self.num_queries, 1))
            assert(query_tokens.shape == (patch_tokens.shape[0], self.num_queries, patch_tokens.shape[1]))
            query_tokens = self.custom_attnpool(query_tokens, patch_tokens, if_pos=self.if_pos)
            return query_tokens
