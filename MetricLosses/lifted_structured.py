import torch
import numpy as np

def lifted_structured():
    assert torch.norm(anchor,2)==torch.norm(positive,2)==torch.norm(negative,2)==1, 'anchor, positive and negative should be norm 1'
    
    loss = max(0, anchor.dot(negative) - anchor.dot(positive) + margin)
    
    return loss


#!/usr/bin/env python

"""
    pytorch_lifted_loss.py
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

def lifted_loss(score, target, margin=1):
    """
      Lifted loss, per "Deep Metric Learning via Lifted Structured Feature Embedding" by Song et al
      Implemented in `pytorch`
    """

    loss = 0
    counter = 0
    
    bsz = score.size(0)
    mag = (score ** 2).sum(1).expand(bsz, bsz)
#     print(mag)
    sim = score.mm(score.transpose(0, 1))
#     print(sim)
    
    dist = (mag + mag.transpose(0, 1) - 2 * sim)
    dist = torch.nn.functional.relu(dist).sqrt()
    
    for i in range(bsz):
        t_i = target[i].item()
        
        for j in range(i + 1, bsz):
            t_j = target[j].item()
            
            if t_i == t_j:
                # Negative component
                # !! Could do other things (like softmax that weights closer negatives)
                l_ni = (margin - dist[i][target != t_i]).exp().sum()
                l_nj = (margin - dist[j][target != t_j]).exp().sum()
                l_n  = (l_ni + l_nj).log()
                
                # Positive component
                l_p  = dist[i,j]
                
                loss += torch.nn.functional.relu(l_n + l_p) ** 2
                counter += 1
    
    return loss / (2 * counter)

# --

if __name__ == "__main__":
    import numpy as np
    np.random.seed(123)
    
    batch = 20
    score = np.random.uniform(0, 1, (batch, 3))
    target = np.random.choice(range(3), batch)
    
    print(lifted_loss(Variable(torch.FloatTensor(score)), Variable(torch.LongTensor(target))))