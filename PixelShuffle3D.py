import torch
import torch.nn as nn

#
#
# class PixelShuffle3d(nn.Module):
#     '''
#     This class is a 3d version of pixelshuffle.
#     '''
#     def __init__(self, scale):
#         '''
#         :param scale: upsample scale
#         '''
#         super().__init__()
#         self.scale = scale
#
#     def forward(self, input):
#         batch_size, channels, in_depth, in_height, in_width = input.size()
#         nOut = channels // self.scale ** 3
#
#         out_depth = in_depth * self.scale
#         out_height = in_height * self.scale
#         out_width = in_width * self.scale
#
#         input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)
#
#         output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
#
#         return output.view(batch_size, nOut, out_depth, out_height, out_width)
#
# def calc_mean_std(feat, eps=1e-5):
#     # eps is a small value added to the variance to avoid divide-by-zero.
#     size = feat.size()
#     assert (len(size) == 5)
#     N, C = size[:2]
#     feat_var = feat.view(N, C, -1).var(dim=2) + eps
#     feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
#     feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
#     return feat_mean, feat_std
#
# def adaptive_instance_normalization(content_feat, style_feat):
#     shape = style_feat.size()[1]
#     style_mean = nn.Linear(shape, shape)(style_feat)
#     style_std = nn.Linear(shape, shape)(style_feat)
#     style_std = style_std[:, :, None, None, None]
#     style_mean = style_mean[:, :, None, None, None]
#     assert (content_feat.size()[:2] == style_std.size()[:2] == style_mean.size()[:2])
#
#     size = content_feat.size()
#     style_std = style_std.expand(size)
#     style_mean = style_mean.expand(size)
#     # style_mean, style_std = calc_mean_std(style_feat)
#
#
#     content_mean, content_std = calc_mean_std(content_feat)
#
#     normalized_feat = (content_feat - content_mean.expand(
#         size)) / content_std.expand(size)
#
#     return normalized_feat * style_std + style_mean

#
# pixel_shuffle = PixelShuffle3d(2)
#
# input = torch.randn(1, 64, 50, 50, 50)
# input2 = torch.randn(1, 512)
# input2 = nn.functional.normalize(input2)
# input2 = nn.Linear(512, 64)(input2)
#
# # size = input.size()
# # input2 = input2.view(1, 64, 1, 1, 1)
#
# # input2 = input2.expand(size)
# # z = input2.expand(size)
#
# # y = input2[-1]
# # x = calc_mean_std(input)
# x = adaptive_instance_normalization(input,  input2)
#
# print(x)
# # output = pixel_shuffle(input)
# # # print(output.shape)
# #
# # class MappingLayers(nn.Module):
# #     '''
# #     Mapping Layers Class
# #     Values:
# #         z_dim: the dimension of the noise vector, a scalar
# #         hidden_dim: the inner dimension, a scalar
# #         w_dim: the dimension of the intermediate noise vector, a scalar
# #     '''
# #
# #     def __init__(self, z_dim, hidden_dim, w_dim):
# #         super().__init__()
# #         self.mapping = nn.Sequential(
# #             # Please write a neural network which takes in tensors of
# #             # shape (n_samples, z_dim) and outputs (n_samples, w_dim)
# #             # with a hidden layer with hidden_dim neurons
# #             #### START CODE HERE ####
# #             nn.Linear(z_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Linear(hidden_dim, w_dim),
# #             #### END CODE HERE ####
# #         )
# #
# #     def forward(self, noise):
# #         '''
# #         Function for completing a forward pass of MappingLayers:
# #         Given an initial noise tensor, returns the intermediate noise tensor.
# #         Parameters:
# #             noise: a noise tensor with dimensions (n_samples, z_dim)
# #         '''
# #         return self.mapping(noise)
# #
# #     # UNIT TEST COMMENT: Required for grading
# #     def get_mapping(self):
# #         return self.mapping
# #
# # map_fn = MappingLayers(64,64,64)
# # assert tuple(map_fn(torch.randn(2, 64)).shape) == (2, 64)
# # assert len(map_fn.mapping) > 4
# # outputs = map_fn(torch.randn(1000, 64))
# # assert outputs.std() > 0.05 and outputs.std() < 0.3
# # assert outputs.min() > -2 and outputs.min() < 0
# # assert outputs.max() < 2 and outputs.max() > 0
# # layers = [str(x) for x in map_fn.get_mapping()]
# # # assert layers == ['Linear(in_features=10, out_features=20, bias=True)',
# # #                   'ReLU()',
# # #                   'Linear(in_features=20, out_features=20, bias=True)',
# # #                   'ReLU()',
# # #                   'Linear(in_features=20, out_features=30, bias=True)']
# # z_sample = torch.randn(1, 512)
# # z_sample = nn.functional.normalize(z_sample)
# # x = nn.Linear(512, 256)(z_sample)
# # x = nn.Linear(256, 64)(x)
# # x1 = x = nn.Linear(64, 32)(x)
# # x2 = x = nn.Linear(32, 16)(x)
# # x3 = x = nn.Linear(16, 8)(x)
#
# # print(map_fn)
# # print(outputs.shape)
# # print("Success!")

# from collections import OrderedDict
# import numpy as np
import torch
from torch import nn, optim
#
# from ignite.engine import *
# from ignite.handlers import *
# from ignite.metrics import *
# from ignite.utils import *
# from ignite.contrib.metrics.regression import *
# from ignite.contrib.metrics import *


# create default evaluator for doctests

# def eval_step(engine, batch):
#     return batch
#
# default_evaluator = Engine(eval_step)

# create default optimizer for doctests

# param_tensor = torch.zeros([1], requires_grad=True)
# default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

# create default model for doctests

# default_model = nn.Sequential(OrderedDict([
#     ('base', nn.Linear(4, 2)),
#     ('fc', nn.Linear(2, 1))
# ]))
#
# manual_seed(666)
# metric = SSIM(data_range=1.0)
# metric.attach(default_evaluator, 'ssim')
# preds = torch.rand([1, 64, 128, 128, 128])
# style = torch.rand([1, 64])
# target = preds * 0.82
# state = metric
# print(state.metrics['ssim'])


# from skimage.metrics import peak_signal_noise_ratio as PSNR
# from skimage.metrics import structural_similarity as SSIM
# preds = preds.numpy()
# target = target.numpy()
# PSNR = PSNR(preds, target)
# SSIM = SSIM(preds, target, multichannel=True, win_size=3)
# print('PSNR:', PSNR, "SSIM:", SSIM)

# class AdaIN(nn.Module):
#     def __init__(self, input_size, style_size):
#         super(AdaIN, self).__init__()
#         self.norm = nn.InstanceNorm2d(input_size, affine=False)
#         self.style = nn.Linear(style_size, input_size*2)
#
#     def forward(self, x, style):
#         x = self.norm(x)
#         style = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
#         gamma, beta = style.chunk(2, dim=1)
#         x = gamma * x + beta
#         return x
#
# x = AdaIN(preds, style)
# y = x


import os

import shutil

src_dir_path = r'D:\Desktop\Taiwan University datasets\0214\new_data'
# img_list = os.listdir(src_dir_path)

to_dir_path = r'D:\Desktop\data_save_MRI'
if not os.path.exists(to_dir_path):
    os.mkdir(to_dir_path)

key = "3D"

if not os.path.exists(to_dir_path):
    print("to_dir_path not exist,so create the dir")
    os.mkdir(to_dir_path, 1)
if os.path.exists(src_dir_path):
    print("src_dir_path exist")
    for file in os.listdir(src_dir_path):
        # is file
        if os.path.isdir(src_dir_path + '/' + file):
            dirpath = os.listdir(src_dir_path + '/' + file)
            for filepath in dirpath:
                # if key in filepath:
                if "3D" in filepath or "MPRA" in filepath or "mpra" in filepath:
                    print('找到包含關鍵字"' + filepath + '"的文件,路徑為----->' + src_dir_path + '/' + file)
                    print('複製至----->' + to_dir_path + file + '/' + filepath)
                    shutil.copytree(src_dir_path + '/' + file + '/' + filepath, to_dir_path + '/' + file + '/' + filepath)




