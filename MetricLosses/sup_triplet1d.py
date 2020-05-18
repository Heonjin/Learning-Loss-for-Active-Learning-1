import torch

def sup_triplet1d(input, target, margin=1.0, reduction='mean', mining_threshold = 8):
    n = input.size(0)
    assert n % 3 ==0, 'the batch size is not multiple of 3.'
    
    loss = 0.
    count = 0
    epsilon = 1e-6
    for i in range(n//3):
        threeinput = input[3*i:3*i+3]
        threetarget = target[3*i:3*i+3]
        _target = torch.log((threetarget[2]-threetarget[1]).abs()+epsilon) - torch.log((threetarget[1]-threetarget[0]).abs()+epsilon)
        _target = _target.detach()
        temp = (torch.log( (threeinput[2] - threeinput[1]).abs()+epsilon) - torch.log((threeinput[1]-threeinput[0]).abs()+epsilon) - _target).pow(2)
        
        if temp.item() < mining_threshold:
            count += 1
            loss += temp
#             print(temp)
    if count ==0:
        return 0
    return 3*loss / count

if __name__ == "__main__":
    import numpy as np
    np.random.seed(123)
    
    batch = 30
    score = torch.tensor(np.random.uniform(0, 1, (batch, 1)))
    target = torch.tensor(np.random.uniform(0, 1, (batch, 1)))
    
    print(sup_triplet1d(score,target))