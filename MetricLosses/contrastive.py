import torch
import numpy as np

def contrastive(anchor, element, sign):
    assert sign in [1,-1], 'sign should be 1 or -1'
    assert torch.norm(anchor,2)==torch.norm(element,2)==1, 'anchor and element should be norm 1'
    if sign==1:
        loss = 1-anchor.dot(element)
    else:
        loss = max(0, anchor.dot(element))
    
    return loss

def test():
    a = torch.tensor(np.array([1/2, 1/2, 1/2, 1/2]))
    b = torch.tensor(np.array([1/2, 1/2, -1/2, -1/2]))
    print(contrastive(a, b, 1))
    print(contrastive(a, b, -1))

# test()