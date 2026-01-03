# ------------------------------------------------------------------------
# Modified from TadTR (Modified from TadTR (https://github.com/Hist-RobotCVGroup/maetad)
# Copyright (c) 2025. Tao Xu.
# ------------------------------------------------------------------------------------------------
# Modified from TadTR (https://github.com/xlliu7/TadTR)
# Copyright (c) 2021. Xiaolong Liu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_



def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class DeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        """
        Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                'd_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in DeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.seq2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        # 2p / 总P
        grid_init = torch.linspace(-self.n_points, self.n_points, self.n_heads * self.n_points + 1)
        grid_init = torch.cat([
            # 目的: 1.去除参考点(中心位置)，2.为下方的广播机制元素数量对齐奠定基础； 做法: 对num分离为两部分，并在第列维上进行拼接，因为本身只有单维度
            grid_init[:self.n_heads * self.n_points // 2],
            grid_init[self.n_heads * self.n_points // 2 + 1:]
        ], dim=0)[:, None]

        # 该部分均是对数量进行改变，可以不考虑语义
        # .view() 是对原张量广播，元素位置不变（按照默认编号排列）！不过该处没有用到
        # repeat即重复，可理解为指定维度扩张，，如shape（1, self.n_levels, 1, 1）就是只对第二个维度组进行扩张，由先前的该维度的单元素到4维元素
        # (self.n_heads * self.n_points,)->(self.n_heads, 1, self.n_points, 1)->(self.n_heads, self.n_levels, self.n_points, 1)
        # 注！.view()会直接调用原张量，也就是改变视图，改变矩阵元素会直接影响实例本身
        grid_init = grid_init.view(self.n_heads, 1, self.n_points, 1).repeat(1, self.n_levels, 1, 1) 

        with torch.no_grad():
            # 采样偏置项sampling_offsets.bias的shape  (self.n_heads, self.n_levels, self.n_points, 1)->(head1, 头1各层级展开, head2，头2的各层级展开)
            # ​​nn.Parameter​​：将展平后的张量转换为可学习参数，纳入模型优化过程
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_temporal_lens, input_level_start_index, input_padding_mask=None):
        """
        以下表示shape
        输入实体数据:param query (= src + pos)         (N, Length_{query}, C) src是源文件,此处的Length_{query}是一种原始特征图展平后空间位置总数（输入时间步/片段总数）
        :param reference_points            (N, Length_{query}, n_levels, 1), range in [0, 1], left (0), right (1), including padding area
                                        or (N, Length_{query}, n_levels, 2), add additional (t) to form reference segments
        输入实体数据:param input_flatten (=src)        (N, \sum_{l=0}^{L-1} T_l, C) # [N, 不同尺度层数的所有时间步数/clip数, 维度]
        :param input_temporal_lens         (n_levels), [T_0, T_1, ..., T_(L-1)]
        :param input_level_start_index     (n_levels, ), [0, T_0, T_1, T_2, ..., T_{L-1}] 整体意思是(n_levels, )shape中,x_level值对应的T_X(层级的时间步,也就是序列长度)
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} T_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        # 输入准备与验证 + 验证
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert input_temporal_lens.sum() == Len_in

        # Len_values = sum([input_temporal_lens[i].item() for i in self.n_levels])计算时间步个数（使用切片删去了特征维度），标量
        Len_values = input_temporal_lens[:self.n_levels].sum().item()

        # value_proj -(linear)-> value，截取有效时间步（非填充）！！！公式: value = W_v·input_flatten + b_v
        value = self.value_proj(input_flatten[:, :Len_values])
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[:, :Len_values, None], float(0))
        value = value.view(N, Len_values, self.n_heads,
                           self.d_model // self.n_heads)
        # the predicted offset in temporal axis. They are *absolute* values, not normalized

        # 一、采样偏移量生成（通过线性层计算）
        # do：对query进行计算后，对张量shape重塑，并满足下次使用需求
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 1)
        
        # 二、注意力权重生成
        # do：对query计算后，再对shape重塑，此时应满足接下来对权重点的softmax，故将-1维改为self.n_levels * self.n_points满足需求
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        # 对注意力权重使用softmax归一化后再展开
        attention_weights = F.softmax(
            attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # 常规可变性注意力
        # shape[-1] == 1, 即观察最后一个维度的实体数据数量
        if reference_points.shape[-1] == 1:
            # the reference points are normalized, but the offset are unnormalized   参考点本身归一化
            # so we need to normalize the offsets
            # (n_levels), [T_0, T_1, ..., T_(L-1)] ->(n_levels, 1）
            offset_normalizer = input_temporal_lens[..., None]
            # (N, Length_{query}, n_heads, n_levels, n_points, 1)
            # 对应层进行相除，因为算数运算只会影响实体数据，且维度对应，故可以算出
            sampling_locations = reference_points[:, :, None, :self.n_levels, None, :] \
                + sampling_offsets / \
                offset_normalizer[None, None, None, :self.n_levels, None, :] # 用步长总数为基数，计算采样点偏移量的归一化，也就是采样偏移是在时间步上进行的
            
        # 解码器可变形注意力   此时参考点是2维，分别表示归一化位置以及总长度占比（也是一种归一化）
        # deform attention in the l-th (l >= 2) decoder layer when segment refinement is enabled
        elif reference_points.shape[-1] == 2: # 最后一维两列的情况 eg.（1，2，2）
            # offsets are related with the size of the reference segment
            # (N, Length_{query}, n_heads, n_levels, n_points, 2)
            sampling_locations = reference_points[:, :, None, :self.n_levels, None, :1] \
                + sampling_offsets / (self.n_points * self.n_heads // 2) * \
                reference_points[:, :, None, :self.n_levels, None, 1:]# * 0.5

        # torch.ones_like(A)*0.5   表示生成与Ashape相同的元素全1张量，乘0.5（也就是元素全0.5）
        # torch.cat((A, B), dim=-1)代表拼接A和B，dim=-1代表在最后一维进行拼接使(x,y)->(x,y, 0.5)
        sampling_locations = torch.cat((sampling_locations, torch.ones_like(sampling_locations)*0.5), dim=-1)
        # [:self.n_levels]代表去掩码等干扰
        input_temporal_lens = input_temporal_lens[:self.n_levels]
        output = deform_attn_core_pytorch(value, input_temporal_lens, sampling_locations, attention_weights)

        output = self.output_proj(output)
        return output, (sampling_locations, attention_weights)


def deform_attn_core_pytorch(value, temporal_lens, sampling_locations, attention_weights):
    '''deformable attention implemeted with grid_sample.'''
    # value = (N, Len_values, self.n_heads, self.d_model // self.n_heads)
    # 其中，Len_values 为总尺度时间步
    N_, S_, M_, D_ = value.shape
    # (N, Length_{query}, n_heads, n_levels, n_points, 1or2, 0.5)
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    # 其中，两个T_分别为：输出的数组[T_0, T_1, ..., T_3]；参与循环量
    # temporal_lens表示已经经过滤去多于层的包含四层时间步数的一维张量，在spilt中本质上表示按照相应尺度大小在Len_values上分离相应片段

    # eg：将483个元素的Len_values分为每层包含[256, 128, 64, 32]的四部分
    # temporal_lens = [T_0, T_1, ..., T_3]; value_list = [[T_0], [T_1], ..., [T_3]]
    value_list = value.split([T_ for T_ in temporal_lens], dim=1)
    # (N, Length_{query}, n_heads, n_levels, n_points, 1or2, 0.5)->(N, Length_{query}, n_heads, n_levels, 2*n_points-1, 1or2, 0)算数运算不作用与结构维度
    sampling_grids = 2 * sampling_locations - 1
    # value = value.flatten(2).transpose(1, 2).reshape(N_*M_, D_, 1, S_)
    # 创建空列表用于存储各层级的采样结果
    sampling_value_list = []
    for lid_, T_ in enumerate(temporal_lens):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, 1, T_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    # .transpose转置
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()
