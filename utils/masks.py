from numpy import dtype
import torch

def make_src_mask(lgt):
    batch_len = len(lgt)
    mask = torch.zeros([batch_len, int(lgt[0].item())], dtype=torch.bool)
    for idx, l in enumerate(lgt):
        for i in range(int(l.item())):
            mask[idx][i] = 1
    return mask.unsqueeze(1)

def make_txt_mask(lgt):
    """
        Create text mask from a sequence of length (not necessarily sorted)

        Input:
        lgt: sequence of length ([3,4,1,2...]): 1d Tensor

        Output:
        A 3 dimension Tensor of mask [B, 1, M] with B is mini-batch length,

    """
    m = torch.max(lgt).item()
    
    txt_mask = []

    for l in lgt:
        txt_mask.append([1,]*int(l.item()) + [0,]*int(m-l))
    
    return torch.BoolTensor(txt_mask).unsqueeze(1)


# if __name__ == "__main__":
#     print(make_src_mask(torch.Tensor([8,7,6,5])))