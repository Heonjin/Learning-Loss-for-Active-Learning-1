import torch

def mc_triplet_log1d(input, classes, margin=0., mining_threshold = 8):
    n = input.size(0)
    print(min(input).item(), max(input).item())
    loss = 0.
    count = 0
    epsilon = 1e-6
    diff = input.expand(n,n)
    diff = diff - torch.transpose(diff, 0,1)
    diff = diff.abs()#pow(2)
    diff = torch.log(diff)
    i = 0
    while count<1600 and i <=n:
        for j in range(i+1,n):
            for k in range(j+1,n):
                dif = diff[i][j] - diff[i][k]
                if classes[i] == classes[j] and classes[i] != classes[k] and dif.item()> -margin:
                    temp = dif + margin
                    loss += temp
                    count += 1
                elif classes[i] == classes[k] and classes[i] != classes[j] and dif.item() < margin :
                    temp = -dif + margin
                    loss += temp
                    count += 1
                else:
                    continue
#         print(i, count)
        i += 1
    if count ==0:
        return 0
    return loss / count

if __name__ == "__main__":
    import numpy as np
    np.random.seed(123)
    
    batch = 9
    score = torch.tensor(np.random.uniform(0, 1, (batch, 1)))
    classes = torch.tensor(np.random.choice(range(10), batch))
    print(classes)
    print(mc_triplet_log1d(score,classes))