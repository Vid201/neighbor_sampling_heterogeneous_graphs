from tqdm import tqdm
import numpy as np
import torch

def get_col_slice(x, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    outs = []
    chunk = 100000
    for i in tqdm(range(start_row_idx, end_row_idx, chunk)):
        j = min(i + chunk, end_row_idx)
        outs.append(x[i:j, start_col_idx:end_col_idx].copy())
    return np.concatenate(outs, axis=0)

def save_col_slice(x_src, x_dst, start_row_idx, end_row_idx, start_col_idx, end_col_idx):
    assert x_src.shape[0] == end_row_idx - start_row_idx
    assert x_src.shape[1] == end_col_idx - start_col_idx
    chunk, offset = 100000, start_row_idx
    for i in tqdm(range(0, end_row_idx - start_row_idx, chunk)):
        j = min(i + chunk, end_row_idx - start_row_idx)
        x_dst[offset + i:offset + j, start_col_idx:end_col_idx] = x_src[i:j]

def construct_sinusoid_encoding_table(num_positions, hidden_dim, padding_idx=None):
    def angle(position, hidden_idx):
        return position / np.power(10000, 2 * (hidden_idx // 2) / hidden_dim)
    
    def pos_angle_vec(position):
        return [angle(position, j) for j in range(hidden_dim)]
    
    sin_table = torch.tensor([pos_angle_vec(i) for i in range(num_positions)])
    sin_table[:, 0::2] = torch.sin(sin_table[:, 0::2])
    sin_table[:, 1::2] = torch.cos(sin_table[:, 1::2])
    return sin_table

def unique(x, dim=None):
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse_flip, perm_flip = inverse.flip([0]), perm.flip([0])
    return unique, inverse, inverse_flip.new_empty(unique.size(0)).scatter_(0, inverse_flip, perm_flip)