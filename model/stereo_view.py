import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv

from skimage import morphology
import numpy as np


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r
        self.use_cuda = use_cuda

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
        #yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

        xx_range = torch.arange(dim_y, dtype=torch.int32)
        #yy_range = torch.arange(dim_x, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        #yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        #yy_channel = torch.matmul(yy_range, yy_ones)

        # transpose y
        #yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (dim_y - 1)
        #yy_channel = yy_channel.float() / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        #yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        #yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

        if torch.cuda.is_available and self.use_cuda:
            input_tensor = input_tensor.cuda()
            xx_channel = xx_channel.cuda()
            #yy_channel = yy_channel.cuda()
        out = torch.cat([input_tensor, xx_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)


        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''
    def __init__(self, c):
        super().__init__()   
        self.scale = c ** -0.5

        self.l_proj1 = CoordConv2d(c, c, kernel_size=1, stride=1, padding=0, with_r=True)
        self.r_proj1 = CoordConv2d(c, c, kernel_size=1, stride=1, padding=0, with_r=True)
    
        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax(-1)

        self.beta1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma1 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta3 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma3 = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x, res_enc, tr_feat):
        x_left, x_right = x.chunk(2, dim=0)
        res_enc_l, res_enc_r = res_enc.chunk(2, dim=0)
        tr_faet_l, tr_faet_r = tr_feat.chunk(2, dim=0)
        b, c, h, w = x_left.shape

        ### M_{right_to_left}
        Q = (self.l_proj1(x_left) + self.beta1 * tr_faet_l).permute(0, 2, 3, 1)  # B * H * W * C
        S = (self.r_proj1(x_right) + self.beta1 * tr_faet_r).permute(0, 2, 1, 3)  # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                            S.contiguous().view(-1, c, w)) * self.scale  # (B*H) * W * W
        score_T = score.permute(0,2,1)

        # mask #
        mask_l = torch.triu(torch.ones((w, w)), diagonal=1).type_as(x_left)  # [W, W]
        valid_mask_l = (mask_l == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1).reshape(b*h, w, w)  # [B, H, W, W]
        mask_r = torch.transpose(mask_l, 0, 1)
        valid_mask_r = (mask_r == 0).unsqueeze(0).unsqueeze(0).repeat(b, h, 1, 1).reshape(b*h, w, w)  # [B, H, W, W]
        score[~valid_mask_l] = torch.finfo(torch.float16).min
        score_T[~valid_mask_r] = torch.finfo(torch.float16).min
        ########

        M_right_to_left = self.softmax(score)
        M_left_to_right = self.softmax(score_T)

        buffer_R = (self.r_proj2(x_right) + self.gamma1 * res_enc_r).permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_l = torch.bmm(M_right_to_left, buffer_R).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W
        buffer_L = (self.l_proj2(x_left) + self.gamma1 * res_enc_l).permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # (B*H) * W * C
        buffer_r = torch.bmm(M_left_to_right, buffer_L).contiguous().view(b, h, w, c).permute(0, 3, 1, 2)  # B * C * H * W

        out_L = x_left + self.beta3 * buffer_l
        out_R = x_right + self.gamma3 * buffer_r

        x = torch.cat([out_L, out_R], dim=0)

        return x