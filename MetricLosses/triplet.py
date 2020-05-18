import torch
import numpy as np

def triplet(anchor, positive, negative, margin=1.):
    assert torch.norm(anchor,2)==torch.norm(positive,2)==torch.norm(negative,2)==1, 'anchor, positive and negative should be norm 1'
    
    loss = max(0, anchor.dot(negative) - anchor.dot(positive) + margin)
    
    return loss

def test():
    a = torch.tensor(np.array([1/2, 1/2, 1/2, 1/2]))
    b = torch.tensor(np.array([1/2, 1/2, -1/2, -1/2]))
    c = torch.tensor(np.array([1/2, -1/2, -1/2, -1/2]))
    print(triplet(a, b, c))

# test()