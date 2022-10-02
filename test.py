import cv2
import torch.nn as nn
import numpy as np
from utils.config import opt
from model.Trainer import InpTrainer
import scipy.io as io
import torch
import os
import matplotlib.pyplot as plt
from sklearn import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from utils.metric.SegmentationMetric import SegmentationMetric

np.seterr(divide='ignore', invalid='ignore')
__all__ = ['SegmentationMetric']


#——————————————————定义准确率acc———————————————————————
def cal_acc(predict, label, mask):
    predict = predict * mask
    label = label * mask
    #b, h, w = np.shape(label)
    h, w = np.shape(label)
    #acc = (np.sum(predict == label) - np.sum(mask == 0)) / (b * h * w - np.sum(mask == 0))
    acc = (np.sum(predict == label) - np.sum(mask == 0)) / (h * w - np.sum(mask == 0))
    return acc


def test_model(data, model):
    model.eval()
    data = torch.tensor(data)


    with torch.no_grad():

        outputimg = model.test_onebatch(data.cuda())
        outimg = outputimg[0].detach()
        _, outidex = torch.max(outimg, 0)
        outidex = outidex.cpu().numpy()

    return outidex


if __name__ == '__main__':
    opt._parse()
#——————————————————————————————共用数据如下————————————————————————————————————
    #注意这里先将伽马能谱数据导入进来，再切割为细分的训练和测试数据
    data_GES = io.loadmat('./data1/GES.mat')["GES"]
    scale0 = np.max(np.abs(data_GES))
    data_GES = data_GES / scale0   #直接对全部的伽马能谱数据归一化

    # 对于PPMA、TRF和Cdata的处理方法与之类似
    data_PPMA = io.loadmat('./data1/PPMA.mat')["PPMA"]  # ΔT化极后磁异常
    scale1 = np.max(np.abs(data_PPMA))
    data_PPMA = data_PPMA / scale1

    data_TRF = io.loadmat('./data1/TRF.mat')["TRF"]  # ΔT磁异常化极上延2km剩余场
    scale2 = np.max(np.abs(data_TRF))
    data_TRF = data_TRF / scale2

    data_TOD = io.loadmat('./data1/TOD.mat')["TOD"]  # 航磁资小波多尺度分析的三阶细节
    scale3 = np.max(np.abs(data_TOD))
    data_TOD = data_TOD / scale3

##————————————————————————————粗分数据——————————————————————————————
    # testdata = data_GES[::5, ::5]
    # testdata = testdata[38:1702, 28:2332]
    #
    # testdata = torch.from_numpy(testdata)  #矩阵变成tensor了
    # testdata = torch.tensor(testdata, dtype=torch.float32)
    #
    # channel1 = io.loadmat('./data1/red.mat')["red"]
    # channel2 = io.loadmat('./data1/purple.mat')["purple"]
    # channel3 = io.loadmat('./data1/gray1.mat')["gray1"]
    # channel4 = io.loadmat('./data1/gray2.mat')["gray2"]
    #
    # # channel1 = channel1[::8, ::8]
    # # channel2 = channel2[::5, ::5]
    # # channel3 = channel3[::8, ::8]
    # # channel4 = channel4[::5, ::5]
    #
    # channel1 = channel1[::5, ::5]
    # channel2 = channel2[::5, ::5]
    # channel3 = channel3[::5, ::5]
    # channel4 = channel4[::5, ::5]
    #
    # channel1 = channel1[38:1702, 28:2332]
    # channel2 = channel2[38:1702, 28:2332]
    # channel3 = channel3[38:1702, 28:2332]
    # channel4 = channel4[38:1702, 28:2332]
    #
    # channel1 = torch.from_numpy(channel1)  #将数组array转换为张量tensor
    # channel1 = torch.tensor(channel1, dtype=torch.float32)
    #
    # channel2 = torch.from_numpy(channel2)
    # channel2 = torch.tensor(channel2, dtype=torch.float32)
    #
    # channel3 = torch.from_numpy(channel3)
    # channel3 = torch.tensor(channel3, dtype=torch.float32)
    #
    # channel4 = torch.from_numpy(channel4)
    # channel4 = torch.tensor(channel4, dtype=torch.float32)

# ——————————————————古近系——————————————————————————————
#     cut_GES = data_GES[:3200, :2000]  # 截取古近系2700×1600的数据
#     testdata = cut_GES[::5, ::5]
#     testdata = torch.from_numpy(testdata)  # 矩阵变成tensor了
#     testdata = torch.tensor(testdata, dtype=torch.float32)
#
#     cut_PPMA = data_PPMA[:3200, :2000]  # 后面只需要调整这里的截取范围即可
#     channel1 = cut_PPMA[::5, ::5]
#     channel1 = torch.from_numpy(channel1)  # 将数组array转换为张量tensor
#     channel1 = torch.tensor(channel1, dtype=torch.float32)
#
#     cut_TRF = data_TRF[:3200, :2000]
#     channel2 = cut_TRF[::5, ::5]
#     channel2 = torch.from_numpy(channel2)  # 将数组array转换为张量tensor
#     channel2 = torch.tensor(channel2, dtype=torch.float32)
#
#     cut_TOD = data_TOD[:3200, :2000]
#     channel3 = cut_TOD[::5, ::5]
#     channel3 = torch.from_numpy(channel3)  # 将数组array转换为张量tensor
#     channel3 = torch.tensor(channel3, dtype=torch.float32)

# ——————————————————白垩系——————————————————————————————
#     cut_GES = data_GES[:, :8700]  # 截取古近系2700×1600的数据
#     testdata = cut_GES[::5, ::5]
#     testdata = torch.from_numpy(testdata)  # 矩阵变成tensor了
#     testdata = torch.tensor(testdata, dtype=torch.float32)
#
#     cut_PPMA = data_PPMA[:, :8700]  # 后面只需要调整这里的截取范围即可
#     channel1 = cut_PPMA[::5, ::5]
#     channel1 = torch.from_numpy(channel1)  # 将数组array转换为张量tensor
#     channel1 = torch.tensor(channel1, dtype=torch.float32)
#
#     cut_TRF = data_TRF[:, :8700]
#     channel2 = cut_TRF[::5, ::5]
#     channel2 = torch.from_numpy(channel2)  # 将数组array转换为张量tensor
#     channel2 = torch.tensor(channel2, dtype=torch.float32)
#
#     cut_TOD = data_TOD[:, :8700]
#     channel3 = cut_TOD[::5, ::5]
#     channel3 = torch.from_numpy(channel3)  # 将数组array转换为张量tensor
#     channel3 = torch.tensor(channel3, dtype=torch.float32)

# ——————————————————泥盆系-石炭系——————————————————————————————
#     cut_GES = data_GES[3700:, 2260:5460]
#     testdata = cut_GES[::5, ::5]
#     testdata = torch.from_numpy(testdata)  # 矩阵变成tensor了
#     testdata = torch.tensor(testdata, dtype=torch.float32)
#
#     cut_PPMA = data_PPMA[3700:, 2260:5460]  # 后面只需要调整这里的截取范围即可
#     channel1 = cut_PPMA[::5, ::5]
#     channel1 = torch.from_numpy(channel1)  # 将数组array转换为张量tensor
#     channel1 = torch.tensor(channel1, dtype=torch.float32)
#
#     cut_TRF = data_TRF[3700:, 2260:5460]
#     channel2 = cut_TRF[::5, ::5]
#     channel2 = torch.from_numpy(channel2)  # 将数组array转换为张量tensor
#     channel2 = torch.tensor(channel2, dtype=torch.float32)
#
#     cut_TOD = data_TOD[3700:, 2260:5460]
#     channel3 = cut_TOD[::5, ::5]
#     channel3 = torch.from_numpy(channel3)  # 将数组array转换为张量tensor
#     channel3 = torch.tensor(channel3, dtype=torch.float32)

##————————-------------------震旦系-寒武系————————————————————————————————————————
    # cut_GES = data_GES[:, 5000:11100]  # 截取古近系2700×1600的数据
    # testdata = cut_GES[::5, ::5]
    # # testdata = testdata[38:1702, 34:1186]
    # # testdata = testdata[6:1734, 2:1218]
    # testdata = torch.from_numpy(testdata)  # 矩阵变成tensor了
    # testdata = torch.tensor(testdata, dtype=torch.float32)
    #
    # cut_PPMA = data_PPMA[:, 5000:11100]  # 后面只需要调整这里的截取范围即可
    # channel1 = cut_PPMA[::5, ::5]
    # # channel1 = channel1[38:1702, 34:1186]
    # # channel1 = channel1[6:1734, 2:1218]
    # channel1 = torch.from_numpy(channel1)  # 将数组array转换为张量tensor
    # channel1 = torch.tensor(channel1, dtype=torch.float32)
    #
    # cut_TRF = data_TRF[:, 5000:11100]
    # channel2 = cut_TRF[::5, ::5]
    # # channel2 = channel2[38:1702, 34:1186]
    # # channel2 = channel2[6:1734, 2:1218]
    # channel2 = torch.from_numpy(channel2)  # 将数组array转换为张量tensor
    # channel2 = torch.tensor(channel2, dtype=torch.float32)
    #
    # cut_TOD = data_TOD[:, 5000:11100]
    # channel3 = cut_TOD[::5, ::5]
    # # channel3 = channel3[38:1702, 34:1186]
    # # channel3 = channel3[6:1734, 2:1218]
    # channel3 = torch.from_numpy(channel3)  # 将数组array转换为张量tensor
    # channel3 = torch.tensor(channel3, dtype=torch.float32)

# ——————————————————花岗岩区——————————————————————————————
    cut_GES = data_GES[:, 8100:]  # 截取古近系2700×1600的数据
    testdata = cut_GES[::5, ::5]
    testdata = torch.from_numpy(testdata)  # 矩阵变成tensor了
    testdata = torch.tensor(testdata, dtype=torch.float32)

    cut_PPMA = data_PPMA[:, 8100:]  # 后面只需要调整这里的截取范围即可
    channel1 = cut_PPMA[::5, ::5]
    channel1 = torch.from_numpy(channel1)  # 将数组array转换为张量tensor
    channel1 = torch.tensor(channel1, dtype=torch.float32)

    cut_TRF = data_TRF[:, 8100:]
    channel2 = cut_TRF[::5, ::5]
    channel2 = torch.from_numpy(channel2)  # 将数组array转换为张量tensor
    channel2 = torch.tensor(channel2, dtype=torch.float32)

    cut_TOD = data_TOD[:, 8100:]
    channel3 = cut_TOD[::5, ::5]
    channel3 = torch.from_numpy(channel3)  # 将数组array转换为张量tensor
    channel3 = torch.tensor(channel3, dtype=torch.float32)

### —————————————————————————局部测试————————————————————————————————————————
    # testdata1 = testdata
    # testdata1 = torch.cat((testdata, channel1, channel2),  dim = 0)  #*****要修改****   , channel3, channel4
    testdata1 = torch.cat((testdata, channel1, channel2, channel3), dim=0)  # , channel3
    # label = io.loadmat('./data1/rough_five/label5.mat')["label5"]  # ****要修改****
    # label = io.loadmat('./data1/SinCam_six/label6.mat')["label6"]
    # label = io.loadmat('./data1/Cretaceous_nine/label9.mat')["label9"]
    label = io.loadmat('./data1/Granite_three/label3.mat')["label3"]
    # label = io.loadmat('./data1/Paleogene_three/label3.mat')["label3"]
    # label = io.loadmat('./data1/DevoCarb_five/labelV.mat')["labelV"]
    # label = io.loadmat('./data/label7.mat')["label7"]
    label = label[::5, ::5]  # 缩小5倍，每隔5个单位取一个数
    # label = label[38:1702, 34:1186]
    # label = label[6:1734, 2:1218]
    # mask = io.loadmat('./data1/rough_five/mask5.mat')["mask5"]
    # mask = io.loadmat('./data1/SinCam_six/mask6.mat')["mask6"] # *****要修改*****
    # mask = io.loadmat('./data1/Cretaceous_nine/mask9.mat')["mask9"]
    # mask = io.loadmat('./data1/Paleogene_three/mask3.mat')["mask3"]
    # mask = io.loadmat('./data1/DevoCarb_five/maskV.mat')["maskV"]
    mask = io.loadmat('./data1/Granite_three/mask3.mat')["mask3"]
    mask = mask[::5, ::5]
    # mask = mask[38:1702, 34:1186]
    # mask = mask[6:1734, 2:1218]

# ##----------------------------------1-----------------------------------------------------
    testdata1 = testdata1.numpy()
    ACC = np.zeros(80)
    for i in range(80):
        modelpath = './checkpoints/Granite_three/Unet64_filed0621_' + str(i) + '.pth' #***路径要修改***
        # SinCam_eleven保存的是三通道的训练和测试数据
        # SinCam_eleven1是四通道1200列数据
        # SinCam_eleven2是四通道1000列数据
        model = InpTrainer(opt)
        # model = nn.DataParallel(model)
        model.load_net(modelpath)
        h, w = np.shape(testdata)
        testdata1 = testdata1.reshape((1, 4, h, w)).astype(np.float32)  # .reshape(a,b)表示a行b列
        outidex = test_model(testdata1, model)  # 有点小问题？outidex矩阵的大小不一致
        # ——————————————用来控制矩阵大小一致————————————————————————
        # label = label[:-2, :-3]
        # mask = mask[:-2, :-3]
        # label = label[:-1, :]
        # mask = mask[:-1, :]
        # label = label[3:499, :]
        # mask = mask[3:499, :]
        # label = label[3:867, 5:597]
        # mask = mask[3:867, 5:597]
        acc = cal_acc(outidex, label, mask)
        ACC[i] = acc

    plt.plot(ACC, 'r')
    plt.show()
    print(np.argmax(ACC))
    print(max(ACC))
# # ##-------------------------------------------------------------------------------------

#-----------------------------整个图幅---------------------------------------------------
 #    testdata = data_GES[::5, ::5]
 #    # testdata = testdata[38:1702, 28:2332]
 #    testdata = torch.from_numpy(testdata)  # 矩阵变成tensor了
 #    testdata = torch.tensor(testdata, dtype=torch.float32)
 #
 #    channel1 = data_PPMA[::5, ::5]
 #    # channel1 = channel1[38:1702, 28:2332]
 #    channel1 = torch.from_numpy(channel1)  # 将数组array转换为张量tensor
 #    channel1 = torch.tensor(channel1, dtype=torch.float32)
 #
 #    channel2 = data_TRF[::5, ::5]
 #    # channel2 = channel2[38:1702, 28:2332]
 #    channel2 = torch.from_numpy(channel2)  # 将数组array转换为张量tensor
 #    channel2 = torch.tensor(channel2, dtype=torch.float32)
 #
 #    channel3 = data_TOD[::5, ::5]
 #    # channel3 = channel3[38:1702, 28:2332]
 #    channel3 = torch.from_numpy(channel3)  # 将数组array转换为张量tensor
 #    channel3 = torch.tensor(channel3, dtype=torch.float32)
 #
 # ### —————————————————————————全局测试————————————————————————————————————————
 #    # testdata1 = testdata
 #    # testdata1 = torch.cat((testdata, channel1, channel2),  dim = 0)  #*****要修改****  , channel3, channel4
 #    testdata1 = torch.cat((testdata, channel1, channel2, channel3), dim = 0)   #, channel3
 #    label = io.loadmat('./data1/label22.mat')["label22"]   #****要修改****
 #    # label = io.loadmat('./data1/rough_five/label5.mat')["label5"]
 #    #label = io.loadmat('./data/label7.mat')["label7"]
 #    label = label[::5, ::5]    #缩小5倍，每隔5个单位取一个数
 #    # label = label[38:1702, 28:2332]
 #    mask = io.loadmat('./data1/mask22.mat')["mask22"]   #*****要修改*****
 #    # mask = io.loadmat('./data1/rough_five/mask5.mat')["mask5"]
 #    mask = mask[::5, ::5]
 #    # mask = mask[38:1702, 28:2332]
 #
 #
 #    saveroot = './testresultTen1/Granite_three/Unet64_filed0621_7/'
 #    if not os.path.exists(saveroot):
 #        os.makedirs(saveroot)
 #
 #    modelpath = './checkpoints/Granite_three/Unet64_filed0621_7.pth'
 #    model = InpTrainer(opt)
 #    model.load_net(modelpath)
 #    h, w = np.shape(testdata)
 #    testdata1 = testdata1.numpy()
 #    testdata1 = testdata1.reshape((1, 4, h, w)).astype(np.float32)  # .reshape(a,b)表示a行b列
 #    outidex = test_model(testdata1, model)  # 有点小问题？outidex矩阵的大小不一致
 #
 #    acc = cal_acc(outidex, label, mask)
 #    print(acc)
 #
 #    io.savemat(saveroot+'result.mat',{'result':outidex})
 #
 #    inputimg1 = ((testdata1+1)*255/2).astype(np.uint8)
 #    outidex = (255 * outidex /4 ).astype(np.uint8)   #根据输出通道数（即分类类别数）调整
 #
 #
 #    cv2.imwrite(saveroot +'output.jpg', outidex)
 #    cv2.imwrite(saveroot + 'input.jpg', inputimg1)

# ————————————————————添加了计算acc的部分————————————————————————————————————————

    # imgPredict = outidex * mask
    # imgLabel = label * mask
    # metric = SegmentationMetric(6)
    # metric.addBatch(imgPredict, imgLabel)
    #
    # pa = metric.pixelAccuracy()  # 像素精度，即预测正确的像素占全部像素的比值
    # # cpa = metric.classPixelAccuracy()
    # mpa, per1 = metric.meanPixelAccuracy()  # 平均像素精度，即对每一类的像素精度，求平均
    # mIoU, per2 = metric.meanIntersectionOverUnion()
    # print('pa is : %f' % pa)
    # # print('cpa is :%f' % cpa) # 列表
    # print('mpa is : %f' % mpa, per1)
    # print('mIoU is : %f' % mIoU, per2)
