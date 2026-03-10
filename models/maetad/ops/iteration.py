import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from .temporal_deform_attn import DeformAttn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import warnings


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


def deform_attn_core_pytorch(value, temporal_lens, sampling_locations, attention_weights):
    '''deformable attention implemeted with grid_sample.'''
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([T_ for T_ in temporal_lens], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, T_ in enumerate(temporal_lens):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, 1, T_)
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


class CausalTemporalDeformableAttention(nn.Module):
    """
        Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
    """
    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        grid_init = torch.linspace(-self.n_points, self.n_points, self.n_heads * self.n_points + 1)
        grid_init = torch.cat([
            grid_init[:self.n_heads * self.n_points // 2],
            grid_init[self.n_heads * self.n_points // 2 + 1:]
        ], dim=0)[:, None]
        grid_init = grid_init.view(self.n_heads, 1, self.n_points, 1).repeat(1, self.n_levels, 1, 1)

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_temporal_lens,
                input_level_start_index, input_padding_mask=None):
        """
        :param query (= src + pos)         (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], left (0), right (1), including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (t) to form reference segments
        :param input_flatten (=src)        (N, \sum_{l=0}^{L-1} T_l, C)
        :param input_temporal_lens         (n_levels), [T_0, T_1, ..., T_(L-1)]
        :param input_level_start_index     (n_levels, ), [0, T_0, T_1, T_2, ..., T_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} T_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert input_temporal_lens.sum() == Len_in
        Len_values = input_temporal_lens[:self.n_levels].sum().item()

        value = self.value_proj(input_flatten[:, :Len_values])
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[:, :Len_values, None], float(0))
        value = value.view(N, Len_values, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 1)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)

        sampling_offsets = -F.relu(-sampling_offsets)  

        offset_normalizer = input_temporal_lens[..., None]  # (n_levels, 1)
        sampling_locations = reference_points[:, :, None, :self.n_levels, None, :] \
                              + sampling_offsets / offset_normalizer[None, None, None, :self.n_levels, None, :]


        sampling_locations = torch.cat((sampling_locations, torch.ones_like(sampling_locations) * 0.5), dim=-1)


        output = deform_attn_core_pytorch(value, input_temporal_lens[:self.n_levels],
                                          sampling_locations, attention_weights)

        # 7. 输出投影
        output = self.output_proj(output)

        return output 


class IterativeReconstructionModule(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4, window_size=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        self.deform_attn = CausalTemporalDeformableAttention(
            d_model=d_model, 
            n_levels=1, 
            n_heads=n_heads, 
            n_points=n_points
        )
        
        self.gru = nn.GRUCell(d_model, d_model)
        
        self.repair_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.temperature = 0.1

    def forward(self, x, pos_embed=None):
            """
            :param x: [B, T, C]
            :param pos_embed:  [1, T, C]
            """
            if pos_embed is not None:
                x = x + pos_embed
                
            B, T, C = x.shape
            device = x.device
            
            h_t = torch.zeros(B, C, device=device)
            output_features = []
            
            temporal_lens = torch.tensor([self.window_size], device=device).long()
            level_start_index = torch.tensor([0], device=device).long()
            ref_pts = torch.ones((B, 1, 1, 1), device=device)

            for t in range(T):
                start_idx = max(0, t - self.window_size + 1)
                history_slice = x[:, start_idx : t + 1, :]
                pad_len = self.window_size - history_slice.shape[1]
                
                padding_mask = torch.zeros((B, self.window_size), device=device, dtype=torch.bool)
                if pad_len > 0:
                    history_window = F.pad(history_slice, (0, 0, pad_len, 0))
                    padding_mask[:, :pad_len] = True 
                else:
                    history_window = history_slice

                context_t, _ = self.deform_attn(
                    query=h_t.unsqueeze(1),
                    reference_points=ref_pts,
                    input_flatten=history_window,
                    input_temporal_lens=temporal_lens,
                    input_level_start_index=level_start_index,
                    input_padding_mask=padding_mask
                )
                context_t = context_t.squeeze(1)

                gate = self.repair_net(torch.cat([x[:, t, :], context_t], dim=-1))
                refined_x_t = x[:, t, :] + gate * (context_t - x[:, t, :])
                
                h_t = self.gru(refined_x_t, h_t)
                output_features.append(h_t)

            return torch.stack(output_features, dim=1)