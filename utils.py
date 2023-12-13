import os
import torch


def build_dir(dir_path, root_path = None):
    if root_path is not None:
        new_dir_path = os.path.join(root_path, dir_path)
    else:
        new_dir_path = dir_path
    if not os.path.isdir(new_dir_path):
        os.makedirs(new_dir_path)
    return new_dir_path


def make_argmax_with_mask(x, mask):
    y = (x >= mask.unsqueeze(1).unsqueeze(1)).to(torch.float)
    top_vals, _ = torch.topk(x, 1)
    y2 = (x >= top_vals).to(torch.float)
    y = torch.clamp(y2 +y, max=1.)
    return y



