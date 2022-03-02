import torch

def argmax(vals, mask):
    argmax = -1
    for i in range(len(vals)):
        if mask[i]:
            if argmax == -1 or vals[i] > vals[argmax]:
                argmax = i
    assert argmax != -1
    return argmax


class RARingBuffer:
    def __init__(self, dim, dtype, device):
        self.storage = torch.zeros(dim, dtype=dtype, device=device)
        self.maxlen = dim[0]
        self.ptr = 0

    def add(self, v):
        self.storage[self.ptr] = torch.tensor(v)
        self.ptr = (self.ptr + 1) % self.maxlen

    def gather(self, idxs):
        return self.storage[idxs]

    def __len__(self):
        return self.maxlen
