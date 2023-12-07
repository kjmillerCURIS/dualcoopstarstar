import os
import sys
import torch
import torch.nn as nn
from clip import clip


class Meow(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        #self.base_positional_embedding = clip_model.visual.attnpool.positional_embedding
        self.base_positional_embedding_data = clip_model.visual.attnpool.positional_embedding.data


def load_clip_to_cpu():
    url = clip._MODELS['RN101']
    model_path = clip._download(url)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    
    model = clip.build_model(state_dict or model.state_dict())
    meow = Meow(model)
    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    load_clip_to_cpu()
