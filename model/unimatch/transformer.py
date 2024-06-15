import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos
        

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1


def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge


def generate_shift_window_attn_mask(input_resolution, window_size_h, window_size_w,
                                    shift_size_h, shift_size_w, device=torch.device('cuda')):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (slice(0, -window_size_h),
                slice(-window_size_h, -shift_size_h),
                slice(-shift_size_h, None))
    w_slices = (slice(0, -window_size_w),
                slice(-window_size_w, -shift_size_w),
                slice(-shift_size_w, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True)

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def upsample_flow_with_mask(flow, up_mask, upsample_factor,
                            is_depth=False):
    # convex upsampling following raft

    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = 1 if is_depth else upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h,
                              upsample_factor * w)  # [B, 2, K*H, K*W]

    return up_flow


def split_feature_1d(feature,
                     num_splits=2,
                     ):
    # feature: [B, W, C]
    b, w, c = feature.size()
    assert w % num_splits == 0

    b_new = b * num_splits
    w_new = w // num_splits

    feature = feature.view(b, num_splits, w // num_splits, c
                           ).view(b_new, w_new, c)  # [B*K, W/K, C]

    return feature


def merge_splits_1d(splits,
                    h,
                    num_splits=2,
                    ):
    b, w, c = splits.size()
    new_b = b // num_splits // h

    splits = splits.view(new_b, h, num_splits, w, c)
    merge = splits.view(
        new_b, h, num_splits * w, c)  # [B, H, W, C]

    return merge


def window_partition_1d(x, window_size_w):
    """
    Args:
        x: (B, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, C)
    """
    B, W, C = x.shape
    x = x.view(B, W // window_size_w, window_size_w, C).view(-1, window_size_w, C)
    return x


def generate_shift_window_attn_mask_1d(input_w, window_size_w,
                                       shift_size_w, device=torch.device('cuda')):
    # calculate attention mask for SW-MSA
    img_mask = torch.zeros((1, input_w, 1)).to(device)  # 1 W 1
    w_slices = (slice(0, -window_size_w),
                slice(-window_size_w, -shift_size_w),
                slice(-shift_size_w, None))
    cnt = 0
    for w in w_slices:
        img_mask[:, w, :] = cnt
        cnt += 1

    mask_windows = window_partition_1d(img_mask, window_size_w)  # nW, window_size, 1
    mask_windows = mask_windows.view(-1, window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, window_size, window_size
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** .5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


def single_head_full_attention_1d(q, k, v,
                                  h=None,
                                  w=None,
                                  ):
    # q, k, v: [B, L, C]

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    scale_factor = c ** 0.5

    scores = torch.matmul(q, k.permute(0, 1, 3, 2)) / scale_factor  # [B, H, W, W]

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v).view(b, -1, c)  # [B, H*W, C]

    return out


def single_head_split_window_attention(q, k, v,
                                       num_splits=1,
                                       with_shift=False,
                                       h=None,
                                       w=None,
                                       attn_mask=None,
                                       ):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits, channel_last=True)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                          ) / scale_factor  # [B*K*K, H/K*W/K, H/K*W/K]

    if with_shift:
        scores += attn_mask.repeat(b, 1, 1)

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

    out = merge_splits(out.view(b_new, h // num_splits, w // num_splits, c),
                       num_splits=num_splits, channel_last=True)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


def single_head_split_window_attention_1d(q, k, v,
                                          relative_position_bias=None,
                                          num_splits=1,
                                          with_shift=False,
                                          h=None,
                                          w=None,
                                          attn_mask=None,
                                          ):
    # q, k, v: [B, L, C]

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * h

    window_size_w = w // num_splits

    q = q.view(b * h, w, c)  # [B*H, W, C]
    k = k.view(b * h, w, c)
    v = v.view(b * h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=-shift_size_w, dims=1)
        k = torch.roll(k, shifts=-shift_size_w, dims=1)
        v = torch.roll(v, shifts=-shift_size_w, dims=1)

    q = split_feature_1d(q, num_splits=num_splits)  # [B*H*K, W/K, C]
    k = split_feature_1d(k, num_splits=num_splits)
    v = split_feature_1d(v, num_splits=num_splits)

    scores = torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1)
                          ) / scale_factor  # [B*H*K, W/K, W/K]

    if with_shift:
        # attn_mask: [K, W/K, W/K]
        scores += attn_mask.repeat(b * h, 1, 1)  # [B*H*K, W/K, W/K]

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*H*K, W/K, C]

    out = merge_splits_1d(out, h, num_splits=num_splits)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=shift_size_w, dims=2)

    out = out.view(b, -1, c)

    return out


class SelfAttnPropagation(nn.Module):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(self, in_channels,
                 **kwargs,
                 ):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, flow,
                local_window_attn=False,
                local_window_radius=1,
                **kwargs,
                ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow,
                                                  local_window_radius=local_window_radius)

        b, c, h, w = feature0.size()

        query = feature0.view(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.view(b, flow.size(1), h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = torch.matmul(query, key.permute(0, 2, 1)) / (c ** 0.5)  # [B, H*W, H*W]
        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, value)  # [B, H*W, 2]
        out = out.view(b, h, w, value.size(-1)).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(self, feature0, flow,
                                  local_window_radius=1,
                                  ):
        assert flow.size(1) == 2 or flow.size(1) == 1  # flow or disparity or depth
        assert local_window_radius > 0

        b, c, h, w = feature0.size()

        value_channel = flow.size(1)

        feature0_reshape = self.q_proj(feature0.view(b, c, -1).permute(0, 2, 1)
                                       ).reshape(b * h * w, 1, c)  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1

        feature0_proj = self.k_proj(feature0.view(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)

        feature0_window = F.unfold(feature0_proj, kernel_size=kernel_size,
                                   padding=local_window_radius)  # [B, C*(2R+1)^2), H*W]

        feature0_window = feature0_window.view(b, c, kernel_size ** 2, h, w).permute(
            0, 3, 4, 1, 2).reshape(b * h * w, c, kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]

        flow_window = F.unfold(flow, kernel_size=kernel_size,
                               padding=local_window_radius)  # [B, 2*(2R+1)^2), H*W]

        flow_window = flow_window.view(b, value_channel, kernel_size ** 2, h, w).permute(
            0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2, value_channel)  # [B*H*W, (2R+1)^2, 2]

        scores = torch.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]

        prob = torch.softmax(scores, dim=-1)

        out = torch.matmul(prob, flow_window).view(b, h, w, value_channel
                                                   ).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

        return out


class TransformerLayer(nn.Module):
    def __init__(self,
                 d_model=128,
                 nhead=1,
                 no_ffn=False,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.no_ffn = no_ffn

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None,
                is_self_attn=False
                ):
        # source, target: [B, L, C]
        query, key, value = source, target, target

        # for stereo: 2d attn in self-attn, 1d attn in cross-attn
        #is_self_attn = (query - key).abs().max() < 1e-6

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C]
        value = self.v_proj(value)  # [B, L, C]

        if attn_type == 'swin' and attn_num_splits > 1:  # self, cross-attn: both swin 2d
            if self.nhead > 1:
                # we observe that multihead attention slows down the speed and increases the memory consumption
                # without bringing obvious performance gains and thus the implementation is removed
                raise NotImplementedError
            else:
                message = single_head_split_window_attention(query, key, value,
                                                             num_splits=attn_num_splits,
                                                             with_shift=with_shift,
                                                             h=height,
                                                             w=width,
                                                             attn_mask=shifted_window_attn_mask,
                                                             )

        elif attn_type == 'self_swin2d_cross_1d':  # self-attn: swin 2d, cross-attn: full 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]

                else:
                    # cross attn 1d
                    message = single_head_full_attention_1d(query, key, value,
                                                            h=height,
                                                            w=width,
                                                            )

        elif attn_type == 'self_swin2d_cross_swin1d':  # self-attn: swin 2d, cross-attn: swin 1d
            if self.nhead > 1:
                raise NotImplementedError
            else:
                if is_self_attn:
                    if attn_num_splits > 1:
                        # self attn shift window
                        message = single_head_split_window_attention(query, key, value,
                                                                     num_splits=attn_num_splits,
                                                                     with_shift=with_shift,
                                                                     h=height,
                                                                     w=width,
                                                                     attn_mask=shifted_window_attn_mask,
                                                                     )
                    else:
                        # full 2d attn
                        message = single_head_full_attention(query, key, value)  # [N, L, C]
                else:
                    if attn_num_splits > 1:
                        assert shifted_window_attn_mask_1d is not None, print(self_attn, attn_type)
                        # cross attn 1d shift
                        message = single_head_split_window_attention_1d(query, key, value,
                                                                        num_splits=attn_num_splits,
                                                                        with_shift=with_shift,
                                                                        h=height,
                                                                        w=width,
                                                                        attn_mask=shifted_window_attn_mask_1d,
                                                                        )
                    else:
                        message = single_head_full_attention_1d(query, key, value,
                                                                h=height,
                                                                w=width,
                                                                )

        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(self,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(TransformerBlock, self).__init__()

        self.self_attn = TransformerLayer(d_model=d_model,
                                          nhead=nhead,
                                          no_ffn=True,
                                          ffn_dim_expansion=ffn_dim_expansion,
                                          )

        self.cross_attn_ffn = TransformerLayer(d_model=d_model,
                                               nhead=nhead,
                                               ffn_dim_expansion=ffn_dim_expansion,
                                               )

    def forward(self, source, target,
                height=None,
                width=None,
                shifted_window_attn_mask=None,
                shifted_window_attn_mask_1d=None,
                attn_type='swin',
                with_shift=False,
                attn_num_splits=None
                ):
        # source, target: [B, L, C]

        # self attention
        source = self.self_attn(source, source,
                                height=height,
                                width=width,
                                shifted_window_attn_mask=shifted_window_attn_mask,
                                attn_type=attn_type,
                                with_shift=with_shift,
                                attn_num_splits=attn_num_splits,
                                is_self_attn=True
                                )

        # cross attention and ffn
        source = self.cross_attn_ffn(source, target,
                                     height=height,
                                     width=width,
                                     shifted_window_attn_mask=shifted_window_attn_mask,
                                     shifted_window_attn_mask_1d=shifted_window_attn_mask_1d,
                                     attn_type=attn_type,
                                     with_shift=with_shift,
                                     attn_num_splits=attn_num_splits,
                                     is_self_attn=False
                                     )

        return source


class FeatureTransformer(nn.Module):
    def __init__(self,
                 num_layers=6,
                 d_model=128,
                 nhead=1,
                 ffn_dim_expansion=4,
                 ):
        super(FeatureTransformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model,
                             nhead=nhead,
                             ffn_dim_expansion=ffn_dim_expansion,
                             )
            for i in range(num_layers)])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feature0, feature1,
                attn_type='swin',
                attn_num_splits=None,
                **kwargs,
                ):

        b, c, h, w = feature0.shape
        assert self.d_model == c

        feature0 = feature0.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]
        feature1 = feature1.flatten(-2).permute(0, 2, 1)  # [B, H*W, C]

        # 2d attention
        if 'swin' in attn_type and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # 1d attention
        if 'swin1d' in attn_type and attn_num_splits > 1:
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask_1d = generate_shift_window_attn_mask_1d(
                input_w=w,
                window_size_w=window_size_w,
                shift_size_w=window_size_w // 2,
                device=feature0.device,
            )  # [K, W/K, W/K]

        else:
            shifted_window_attn_mask_1d = None

        # concat feature0 and feature1 in batch dimension to compute in parallel
        concat0 = torch.cat((feature0, feature1), dim=0)  # [2B, H*W, C]
        concat1 = torch.cat((feature1, feature0), dim=0)  # [2B, H*W, C]

        for i, layer in enumerate(self.layers):
            assert shifted_window_attn_mask_1d is not None, print(attn_type + f"_{i}_check")
            concat0 = layer(concat0, concat1,
                            height=h,
                            width=w,
                            attn_type=attn_type,
                            with_shift='swin' in attn_type and attn_num_splits > 1 and i % 2 == 1,
                            attn_num_splits=attn_num_splits,
                            shifted_window_attn_mask=shifted_window_attn_mask,
                            shifted_window_attn_mask_1d=shifted_window_attn_mask_1d
                            )

            # update feature1
            concat1 = torch.cat(concat0.chunk(chunks=2, dim=0)[::-1], dim=0)

        feature0, feature1 = concat0.chunk(chunks=2, dim=0)  # [B, H*W, C]

        # reshape back
        feature0 = feature0.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        feature1 = feature1.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

        return feature0, feature1


def global_correlation_softmax_stereo(feature0, feature1,
                                      ):
    # global correlation on horizontal direction
    b, c, h, w = feature0.shape

    x_grid = torch.linspace(0, w - 1, w, device=feature0.device)  # [W]

    feature0 = feature0.permute(0, 2, 3, 1)  # [B, H, W, C]
    feature1 = feature1.permute(0, 2, 1, 3)  # [B, H, C, W]

    correlation = torch.matmul(feature0, feature1) / (c ** 0.5)  # [B, H, W, W]

    # mask subsequent positions to make disparity positive
    mask = torch.triu(torch.ones((w, w)), diagonal=1).type_as(feature0)  # [W, W]
    valid_mask = (mask == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1)  # [B, H, W, W]

    correlation[~valid_mask] = -torch.finfo(torch.float16).tiny

    prob = F.softmax(correlation, dim=-1)  # [B, H, W, W]

    correspondence = (x_grid.view(1, 1, 1, w) * prob).sum(-1)  # [B, H, W]

    # NOTE: unlike flow, disparity is typically positive
    disparity = x_grid.view(1, 1, w).repeat(b, h, 1) - correspondence  # [B, H, W]

    return disparity.unsqueeze(1), prob  # feature resolution


def local_correlation_softmax_stereo(feature0, feature1, local_radius,
                                     ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1).contiguous()  # [B, H*W, 2]

    local_h = 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(0, 0,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1), 2]
    sample_coords = coords.unsqueeze(-2) + window_grid  # [B, H*W, (2R+1), 2]

    sample_coords_softmax = sample_coords

    # exclude coords that are out of image space
    valid_x = (sample_coords[:, :, :, 0] >= 0) & (sample_coords[:, :, :, 0] < w)  # [B, H*W, (2R+1)^2]
    valid_y = (sample_coords[:, :, :, 1] >= 0) & (sample_coords[:, :, :, 1] < h)  # [B, H*W, (2R+1)^2]

    valid = valid_x & valid_y  # [B, H*W, (2R+1)^2], used to mask out invalid values when softmax

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode='zeros', align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)]
    feature0_view = feature0.permute(0, 2, 3, 1).contiguous().view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)]

    # mask invalid locations
    corr[~valid] = -torch.finfo(torch.float16).max

    prob = F.softmax(corr, -1)  # [B, H*W, (2R+1)]

    correspondence = torch.matmul(prob.unsqueeze(-2),
                                  sample_coords_softmax).squeeze(-2).view(
        b, h, w, 2).permute(0, 3, 1, 2).contiguous()  # [B, 2, H, W]

    flow = correspondence - coords_init  # flow at feature resolution
    match_prob = prob

    flow_x = -flow[:, :1]  # [B, 1, H, W]

    return flow_x, match_prob


def upsample_flow_with_mask(flow, up_mask, upsample_factor):
    # convex upsampling following raft

    mask = up_mask
    b, flow_channel, h, w = flow.shape
    mask = mask.view(b, 1, 9, upsample_factor, upsample_factor, h, w)  # [B, 1, 9, K, K, H, W]
    mask = torch.softmax(mask, dim=2)

    multiplier = upsample_factor
    up_flow = F.unfold(multiplier * flow, [3, 3], padding=1)
    up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)  # [B, 2, 9, 1, 1, H, W]

    up_flow = torch.sum(mask * up_flow, dim=2)  # [B, 2, K, K, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)  # [B, 2, K, H, K, W]
    up_flow = up_flow.reshape(b, flow_channel, upsample_factor * h,
                              upsample_factor * w)  # [B, 2, K*H, K*W]

    return up_flow


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def local_correlation_with_flow(feature0, feature1,
                                flow,
                                local_radius,
                                padding_mode='zeros',
                                dilation=1,
                                ):
    b, c, h, w = feature0.size()
    coords_init = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    coords = coords_init.view(b, 2, -1).permute(0, 2, 1)  # [B, H*W, 2]

    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1

    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=feature0.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(-1, 2).repeat(b, 1, 1, 1)  # [B, 1, (2R+1)^2, 2]
    sample_coords = coords.unsqueeze(-2) + window_grid * dilation  # [B, H*W, (2R+1)^2, 2]

    # flow can be zero when using features after transformer
    if not isinstance(flow, float):
        sample_coords = sample_coords + flow.view(
            b, 2, -1).permute(0, 2, 1).unsqueeze(-2)  # [B, H*W, (2R+1)^2, 2]
    else:
        assert flow == 0.

    # normalize coordinates to [-1, 1]
    sample_coords_norm = normalize_coords(sample_coords, h, w)  # [-1, 1]
    window_feature = F.grid_sample(feature1, sample_coords_norm,
                                   padding_mode=padding_mode, align_corners=True
                                   ).permute(0, 2, 1, 3)  # [B, H*W, C, (2R+1)^2]
    feature0_view = feature0.permute(0, 2, 3, 1).view(b, h * w, 1, c)  # [B, H*W, 1, C]

    corr = torch.matmul(feature0_view, window_feature).view(b, h * w, -1) / (c ** 0.5)  # [B, H*W, (2R+1)^2]

    corr = corr.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()  # [B, (2R+1)^2, H, W]

    return corr