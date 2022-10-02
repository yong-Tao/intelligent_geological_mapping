import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
import time
#from model.unet import Unet
from model.UNet_CC import Unet1
from model.UNet_SOCA import Unet2
from model.UNet_GSoP import Unet3
# from model.UNet_nonlocal import Unet1
#from model.unetm import Unet
from model.unet_new import Unet4
from model.UNet_CA import Unet5
from model.CAM_UNet import Unet6
from model.MACUNet import Unet7
from model.unet_ACB import Unet8
from model.unet2plus import Unet9
from model.unet_improve import Unet10
from model.unet_improve2 import Unet11
from model.unet_improve3 import Unet12
from model.unet_improve4 import Unet13
from model.unet_MA import Unet14
from model.SE_UNet import Unet15
from model.srresnet import NetG
from model.DnCNN import DnCNN
import torch.nn.functional as F
from utils.loss.recall_loss import RecallCrossEntropy
from utils.loss.focal_loss import FocalLoss
from collections import Counter
from utils.dataset import Dataset
from .swin_transformer import SwinTransformer
from .utnet import UTNet
from model.MTUNet import MTUNet
from model.U_Transformer import U_Transformer




def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def total_variation_loss(image):
    # ---------------------------------------------------------------
    # shift one pixel and get difference (for both x and y direction)
    # ---------------------------------------------------------------
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))

    return loss


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)
    weights = np.ones(6)
    weights = 1 / np.log(1.02 + weights)
    weights[classes] = 1 / np.log(1.02 + cls_w)
    return torch.from_numpy(weights).float().cuda()


class InpTrainer(nn.Module):
    def __init__(self, opt):
        super(InpTrainer, self).__init__()
        # self.net = nn.DataParallel(Unet(num_input_channels=3, num_output_channels=8, feature_scale=1, upsample_mode='bilinear',
        #                                 norm_layer=nn.BatchNorm2d, need_sigmoid=False)).cuda()
        self.net = nn.DataParallel(Unet6(n_channels=4, out_channels= 4)).cuda()  # ,recurrence=2)).cuda()

        # self.net = nn.DataParallel(MTUNet(out_ch=7)).cuda()

        # self.net = nn.DataParallel(U_Transformer(in_channels=3, classes=7)).cuda()

        # self.net = nn.DataParallel(UTNet(in_chan = 3, base_chan = 64, num_classes = 7, reduce_size = 8, block_list='1234', num_blocks=[1, 1, 1, 1],
        #                                  projection='interp', num_heads=[4,4,4,4], attn_drop=0.1, proj_drop=0.1, bottleneck=False,
        #                                  maxpool=True, rel_pos=True, aux_loss=False)).cuda()
        # self.net = nn.DataParallel(SwinTransformer(
        #     img_size=224, patch_size=4, in_chans=3, num_classes= 7,
        #     embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        #     window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        #     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        #     norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        #     use_checkpoint=False
        # )).cuda()

        #self.net = nn.DataParallel(Unet()).cuda()
        self.opt = opt
        #self.get_weights = get_weights
        # self.loss = FocalLoss(gamma=2.0)      #, reduction='mean')
        self.loss = nn.CrossEntropyLoss()
        # self.loss = FL(alpha=None, gamma=2, class_num=6)
        # self.loss = RecallCrossEntropy(n_classes=10)  # , ignore_index=0)
        # self.final = nn.Sigmoid()
        # self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
                                                                                               #每训练step_size个epoch,更新一次参数
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10, eta_min=opt.lr * 0.01, last_epoch=-1)

    def train_onebatch(self, data, label, mask):
        self.lr_scheduler.step()
        mask = torch.repeat_interleave(mask, 4, dim=1)   #对输入张量按指定维度扩展

        outimg = self.net(data)
        # outimg = self.final(outimg)
        eg = torch.zeros_like(outimg)  # 产生一个与outimg一样大小的全0数组
        eg[:, 0, :, :] = 1

        outimg = outimg * mask + eg.cuda() * (1 - mask)

        # label = F.one_hot(label, )   #注意，交叉熵是不需要转为one-hot的

        loss = self.loss(outimg, label)#, weight)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def test_onebatch(self, data):
        output = self.net(data)
        return output

    def test_onebatch2(self, image, data2, vmax1, vmean1, vmax2, vmean2):
        output = self.net(image, data2)

        outlow = self.pad(output)
        outlow = self.low(outlow)
        aa = torch.log(self.relu(output * vmax1 + vmean1) + 0.001) / 2
        outhigh = aa[:, :, 1:, :] - aa[:, :, :-1, :]
        outhigh = (outhigh - vmean2) / vmax2
        lossL = self.lossMSE(outlow, image)
        lossH = self.lossMSE(outhigh, data2[:, :, :-1, :])
        loss = lossL + lossH

        output = output.detach().cpu().numpy()

        return output, loss.detach().cpu().numpy()

    def save_net(self, save_path=None, **kwargs):
        if save_path is None:
            timestr = time.strftime('%m%d%H%M')  # time.strftime(format[, t])获取当前本地时间
            # %Y,%m,%d,%H,%M,%S分别表示年月日时分秒
            # save_path = 'checkpoints1r/Unet64_filed0621'# + '_%s' % timestr
            # 这里checkpoints1r保存有2个cc的参数，从1-30

            save_path = 'checkpoints/Granite_three/Unet64_filed0621'  # + '_%s' % timestr
                                                 #1500列是在学院服务器，1200列是在极链服务器,前面两个均是三通道（保存在SinCam_eleven）
                                                # SinCam_eleven1是四通道1200列数据, 3通道(0,1,3)1200列数据
                                                #SinCam_eleven2是四通道1000列数据

            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_
            save_path = save_path + ".pth"
        torch.save(self.net.state_dict(), save_path)
        return save_path

    def load_net(self, save_path):
        state_dict = torch.load(save_path)
        self.net.load_state_dict(state_dict)
        return self


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# import math
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import torch
# import time
# from model.unet import Unet
# from model.UNet_CC import Unet1
# from model.UNet_SOCA import Unet2
# # from model.UNet_nonlocal import Unet1
# # from model.unetm import UNet
# # from model.Dnunet import UNet
# from model.resnet import ResNet
# from model.srresnet import NetG
# from model.DnCNN import DnCNN
# import torch.nn.functional as F
# from utils.loss.recall_loss import RecallCrossEntropy
# from utils.loss.focal_loss import FocalLoss
# from utils.loss.fc_loss import focal_loss
# from utils.dataset import Dataset
#
#
# def set_requires_grad(nets, requires_grad=False):
#     if not isinstance(nets, list):
#         nets = [nets]
#     for net in nets:
#         if net is not None:
#             for param in net.parameters():
#                 param.requires_grad = requires_grad
#
#
# def total_variation_loss(image):
#     # ---------------------------------------------------------------
#     # shift one pixel and get difference (for both x and y direction)
#     # ---------------------------------------------------------------
#     loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
#            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
#
#     return loss
#
#
# def get_weights(target):
#     t_np = target.view(-1).data.cpu().numpy()
#
#     classes, counts = np.unique(t_np, return_counts=True)
#     cls_w = np.median(counts) / counts
#     # cls_w = class_weight.compute_class_weight('balanced', classes, t_np)
#     weights = np.ones(6)
#     weights = 1 / np.log(1.02 + weights)
#     weights[classes] = 1 / np.log(1.02 + cls_w)
#     return torch.from_numpy(weights).float().cuda()
#
#
# class InpTrainer(nn.Module):
#     def __init__(self, opt):
#         super(InpTrainer, self).__init__()
#         # self.net = nn.DataParallel(Unet(num_input_channels=4, num_output_channels=12, feature_scale=1, upsample_mode='bilinear',
#         #                                norm_layer=nn.BatchNorm2d, need_sigmoid=False)).cuda()
#         self.net = nn.DataParallel(Unet1(n_channels=4, out_channels=12)).cuda()  # ,recurrence=2)).cuda()
#         # self.net = nn.DataParallel(Unet()).cuda()
#         self.opt = opt
#         # self.get_weights = get_weights
#         # self.loss = focal_loss()
#         self.loss = nn.CrossEntropyLoss()
#         # self.loss = RecallCrossEntropy(n_classes=10)  # , ignore_index=0)
#         # self.final = nn.Sigmoid()
#         # self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum=0.9)
#         self.optimizer = optim.Adam(self.net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#         self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)
#         # 每训练step_size个epoch,更新一次参数
#
#     def train_onebatch(self, data, label, mask):
#         self.lr_scheduler.step()
#         mask = torch.repeat_interleave(mask, 12, dim=1)
#
#         outimg = self.net(data)
#         # outimg = self.final(outimg)
#         eg = torch.zeros_like(outimg)  # 产生一个与outimg一样大小的全0数组
#         eg[:, 0, :, :] = 1
#
#         outimg = outimg * mask + eg.cuda() * (1 - mask)
#         """
#         # ##————————————————————————————————————————————————————————————————————
# #
# #         #您应该为所有批次指定所有类的权重，而不考虑任何特定批次中的标签。在计算损失时，PyTorch将采用与自身存在的类标签相对应的权重。
# #         label2n = label.view(-1,20*128*128)  #注意：batch_size大小
# #         # s = label2n.cpu().detach().numpy().tolist() #先转numpy再转list
# #         # s1 = set()
# #         # for item in s:
# #         #     for i in item:
# #         #         s1.add(i)
# #         # print(s1)
# #         # L = len(s1)
# #         # print(L)
# #         weights = np.zeros(6)
# #         for n in range(label2n.shape[1]):
# #             weights[label2n[0][n]] += 1
# #         #print(weights)
# #
# #         weights = weights.astype(np.float32)
# #         weights = weights / np.sum(weights)
# #         weights = 1 / np.log(1.02 + weights)        #取对数
# #         #print(weights.shape)
# #
# #         weights = torch.Tensor(weights)
# #         #print(weights.shape)
# # # —————————————————————————————————————————————————————————————————————
#
#         """
#
#         # weights = torch.Tensor(Dataset.weights)
#         # weights =get_weights(label)
#         loss = self.loss(outimg, label)  # , weights)
#
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         return loss
#
#     def test_onebatch(self, data):
#         output = self.net(data)
#         return output
#
#     def test_onebatch2(self, image, data2, vmax1, vmean1, vmax2, vmean2):
#         output = self.net(image, data2)
#
#         outlow = self.pad(output)
#         outlow = self.low(outlow)
#         aa = torch.log(self.relu(output * vmax1 + vmean1) + 0.001) / 2
#         outhigh = aa[:, :, 1:, :] - aa[:, :, :-1, :]
#         outhigh = (outhigh - vmean2) / vmax2
#         lossL = self.lossMSE(outlow, image)
#         lossH = self.lossMSE(outhigh, data2[:, :, :-1, :])
#         loss = lossL + lossH
#
#         output = output.detach().cpu().numpy()
#
#         return output, loss.detach().cpu().numpy()
#
#     def save_net(self, save_path=None, **kwargs):
#         if save_path is None:
#             timestr = time.strftime('%m%d%H%M')  # time.strftime(format[, t])获取当前本地时间
#             # %Y,%m,%d,%H,%M,%S分别表示年月日时分秒
#             # save_path = 'checkpoints1r/Unet64_filed0621'# + '_%s' % timestr
#             # 这里checkpoints1r保存有2个cc的参数，从1-30
#
#             save_path = 'checkpoints_one_cc/SinCam_eleven1/Unet64_filed0621'  # + '_%s' % timestr
#             # 1500列是在学院服务器，1200列是在极链服务器,前面两个均是三通道（保存在SinCam_eleven）
#             # SinCam_eleven1是四通道
#
#             for k_, v_ in kwargs.items():
#                 save_path += '_%s' % v_
#             save_path = save_path + ".pth"
#         torch.save(self.net.state_dict(), save_path)
#         return save_path
#
#     def load_net(self, save_path):
#         state_dict = torch.load(save_path)
#         self.net.load_state_dict(state_dict)
#         return self
#
#
# class TVLoss(nn.Module):
#     def __init__(self, TVLoss_weight=1):
#         super(TVLoss, self).__init__()
#         self.TVLoss_weight = TVLoss_weight
#
#     def forward(self, x):
#         batch_size = x.size()[0]
#         h_x = x.size()[2]
#         w_x = x.size()[3]
#         count_h = self._tensor_size(x[:, :, 1:, :])  # 算出总共求了多少次差
#         count_w = self._tensor_size(x[:, :, :, 1:])
#         h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
#         w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
#         return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
#
#     def _tensor_size(self, t):
#         return t.size()[1] * t.size()[2] * t.size()[3]
#
#
