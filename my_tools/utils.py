import torch
import numpy as np


def FT(x): return torch.FloatTensor(x)
def LT(x): return torch.LongTensor(x)


def stack(x, dim=0, lib=np):
    if lib == np:
        return lib.stack(x, axis=dim)
    elif lib == torch:
        return lib.stack(x, dim=dim)
    else:
        raise NotImplementedError

def clamp(x, min=None, max=None, lib=np):
    if lib == np:
        return lib.clip(x, a_min=min, a_max=max)
    elif lib == torch:
        return lib.clamp(x, min=min, max=max)
    else:
        raise NotImplementedError
