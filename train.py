import cv2
import os
import numpy as np
from torch.utils import data as data_
from tqdm import tqdm
from utils.config import opt
from utils.dataset import Dataset
from model.Trainer import InpTrainer
import scipy.io as io
import torch
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.cuda.current_device()
SAVEROOT='./result/LYT1221/'
if not os.path.exists(SAVEROOT):  #判断目录是否存在，如果不存在就创建相关目录
    os.makedirs(SAVEROOT)

def test_model(dataloader, model, ifsave=True, test_num=1000, name='test'):  #100000
                                                                        #按理说，这里的test也占有内存
    num = 0
    ACC=0
    model.eval()   #不用Batch Normalization 和 Dropout
    #for ii, (data, label, mask) in enumerate(dataloader):
    for ii, (data, label, mask,testchan1, testchan2, testchan3) in enumerate(dataloader): #enumerate 枚举，列举    #, testchan1, testchan2, testchan3, testchan4
        # 对于一个可迭代的或者可遍历的对象，enumerate将其组成一个索引序列，利用它可以同时获得索引以及对应的值  testchan2,

        data = torch.cat((data, testchan1, testchan2, testchan3), dim = 1)   #, testchan3, testchan2,

        #with torch.no_grad(): #with torch.no_grad()或者 @torch.no_grad() 中的数据不需要计算梯度，也不会进行反向传播
        outputimg = model.test_onebatch(data.cuda())
        num += 1
        if ifsave:
            for i in range(1):  #range(5)是从0-5并且不包含5
                inputimg = data[i][0].numpy()   #.numpy()是将Tensor转换为ndarray(即n维数组)
                labelimg = label[i].numpy()
                outimg = outputimg[i].detach()
                _, outidex = torch.max(outimg, 0)   #torch.max(input, dim)，其中input是softmax函数输出的一个tensor
                                                                     #dim是max函数索引的维度（取值0或者1），0代表每一列的最大值，1代表每一行的最大值
                                                                     #返回两个tensor，第一个是每行的最大值，softmax的输出最大是1，所以第一个tensor是全1的tensor
                                                                     #第二个tensor是每行最大值的索引

                outidex = outidex.cpu().numpy()     #将数据从gpu转成cpu
                inputimg = ((inputimg+1)*255/2).astype(np.uint8) #astype转换数据类型
                labelimg = (255 * labelimg /4).astype(np.uint8)
                outidex = (255*outidex/4).astype(np.uint8)

                # cv2.imwrite()用于将图像保存到任何存储设备 cv2.imwrite(filename, image)
                cv2.imwrite(SAVEROOT + name+'outimg' + str(ii) + '.jpg', outidex) #cv2.imwrite()保存图像
                cv2.imwrite(SAVEROOT + name+'inputimg' + str(ii) + '.jpg', inputimg)
                cv2.imwrite(SAVEROOT + name + 'labelimg' + str(ii) + '.jpg', labelimg)

        if ii > test_num:
            break
    #ACC/num输出准确度
    return #outputimg

#******************************************************************************
# class IMB(torch.nn.Module):
#     def __init__(self):
#         super(IMB, self).__init__()
#     def forward(self,cls_num):
#         for i in range(cls_num):
#             cls_num_list.append(self.num_per_cls_dict[i])
#         return cls_num_list
#*********************************************************************************


#def train(traindata, trainlabel, trainmask,testdata, testlabel,testmask):  #):   #,trainchan1, trainchan2, trainchan3,trainchan4,
def train(traindata, trainlabel, trainmask, testdata, testlabel, testmask, trainchan1, trainchan2, trainchan3,
          testchan1, testchan2, testchan3):#, testchan3, testchan4)
    opt._parse()   #把参数打印出来
    train_dataset = Dataset(traindata, trainlabel, trainmask, 'train', 128, 5, 10000,
                            trainchan1, trainchan2, trainchan3) #, trainchan3, trainchan4)        #裁剪的训练图片大小为128×128，scale=10
    train_dataloader = data_.DataLoader(train_dataset,
                                         batch_size = 200,  #batch_size 的大小
                                         shuffle=True)   #shuffle 洗牌，即是否打乱数据集

    test_dataset = Dataset(testdata, testlabel, testmask, 'test', 128, 5, 1000,
                           testchan1, testchan2, testchan3)#, testchan3, testchan4)  , testchan3
    test_dataloader = data_.DataLoader(test_dataset,
                                       batch_size= 20,  #test_dataloader应该也会占内存
                                       shuffle=False)

    trainer = InpTrainer(opt)
    if opt.load_net:
        trainer.load_net(opt.load_net)
    print('model construct completed')

    if opt.test_only:
        eval_result1 = test_model(test_dataloader, trainer, ifsave=True, test_num=opt.test_num, name='test')
        print('eval_loss: ', eval_result1)
        return

    #-----------------------------------------------------------------------------------

    iteration = 80  #设置学习的轮数
    LOSS = np.zeros(iteration)
    for e in range(iteration):
        for ii, (data, label, mask, trainchan1, trainchan2, trainchan3) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):#, trainchan1, trainchan2, trainchan3, trainchan4
        # for ii, (data, label, mask) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            data = torch.cat((data, trainchan1, trainchan2, trainchan3), dim=1)   #, trainchan3, trainchan3, trainchan4
            trainer.train()
            loss = trainer.train_onebatch(data.cuda(), label.cuda(), mask.cuda())
        print('loss:', loss.detach().cpu().numpy())
        LOSS[e] = loss  #保存loss的值
        test_model(test_dataloader, trainer, ifsave=True, test_num=opt.test_num, name='test')
        savemodel =trainer.save_net(best_map=e)
        print("save to %s !" % savemodel)

    x_value = list(range(iteration))    #range(n)是从0开始，n-1结束
    y_value = LOSS
    y = np.mat(y_value)
    io.savemat( './plt/Granite_three/y1.mat', {'y1':y})  #{}中是个字典
    plt.plot(x_value, y_value, c='blue')#, marker = "o", markersize = 3)      # 用plot函数绘制折线图，线条颜色设置为蓝色
    plt.title('Train_Loss', fontsize=24)       # 设置图表标题和标题字号
    #plt.tick_params(axis='both', which='major', labelsize=14)    # 设置刻度的字号
    plt.xlabel('epoch', fontsize=14)   # 设置x轴标签及其字号
    plt.ylabel('loss', fontsize=14)    # 设置y轴标签及其字号
    # plt.savefig('./plt/SinCam_six/p.tif')
    plt.show()      # 显示图表

if __name__ == '__main__':  #__name__内置变量，用于表示当前模块的名字
#————————————————————包含多个通道的数据—————————————————————————————————————
    #注意这里先将伽马能谱数据导入进来，再切割为细分的训练和测试数据
    data_GES = io.loadmat('./data1/GES.mat')["GES"]
    scale0 = np.max(np.abs(data_GES))
    data_GES = data_GES / scale0   #直接对全部的伽马能谱数据归一化

    gray_solid = io.loadmat('./data1/gray1.mat')["gray1"]
    # gray_dotted = io.loadmat('./data1/gray2.mat')["gray2"]
    red_solid = io.loadmat('./data1/red.mat')["red"]
    # purple_dotted = io.loadmat('./data1/purple.mat')["purple"]

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

##————————————————————————粗分数据处理————————————————————————————————————
##——————————————————————————————————————————————————————————————————
    # scale1 = np.max(np.abs(data_PPMA))
    # data_PPMA = data_PPMA / scale1
    #
    # scale1 = np.max(np.abs(data_PPMA))
    # data_PPMA = data_PPMA / scale1
    #
    # scale2 = np.max(np.abs(data_TRF))
    # data_TRF = data_TRF / scale2
    #
    # scale3 = np.max(np.abs(data_TOD))
    # data_TOD = data_TOD / scale3
    # #在使用时再归一化
    #
    # #gamma能谱
    # traindata = data_GES[1500:7000, : ] #训练数据         #对照matlab中label5，确认边界点是(1001,3000)（含），(1,5910)（含）
    #                                                         # 注意，python的点加1，等于MATLAB的点
    #                                                         # 同时，python的前闭后开，MATLAB是闭区间
    # testdata = data_GES[:2000, :]  #取最后的2000列为testdata
    #
    # trainchan1 = red_solid[1500:7000, : ]
    # testchan1 = red_solid[:2000,:]
    # #
    # # trainchan2 = purple_dotted[1500:7000, :]
    # # testchan2 = purple_dotted[:2000, :]
    # #
    # trainchan3 = gray_solid[1500:7000, : ]
    # testchan3 = gray_solid[:2000,:]
    # #
    # # trainchan4 = gray_dotted[1500:7000, :]
    # # testchan4 = gray_dotted[:2000, :]
    #
    # mask = io.loadmat('./data1/rough_five/mask5.mat')["mask5"]
    # trainmask = mask[1500:7000, : ]
    # testmask = mask[:2000, :]
    #
    # label = io.loadmat('./data1/rough_five/label5.mat')["label5"]
    # trainlabel = label[1500:7000, : ]
    # testlabel = label[:2000, :]   # 从label中截取trainlabel和testlabel
    #
    # train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,
    #   trainchan1, trainchan3, testchan1, testchan3)#, trainchan3, trainchan4,, testchan3, testchan4)

#********************************细分数据处理******************************************
#——————————————————————古近系————————————————————————————————————————————
    # cut_GES = data_GES[:3200, :2000]  # 截取古近系2700×1600的数据
    # traindata = cut_GES[:2000, 80:1080]  # 2700×400训练数据
    # testdata = cut_GES[2000:3200,1000:2000]  # 2700×1200测试数据
    #
    # cut_PPMA = data_PPMA[:3200, :2000]
    # trainchan1 = cut_PPMA[:2000, 80:1080]
    # testchan1 = cut_PPMA[2000:3200,1000:2000]
    #
    # cut_TRF = data_TRF[:3200, :2000]
    # trainchan2 = cut_TRF[:2000, 80:1080]
    # testchan2 = cut_TRF[2000:3200,1000:2000]
    #
    # cut_TOD = data_TOD[:3200, :2000]
    # trainchan3 = cut_TOD[:2000, 80:1080]
    # testchan3 = cut_TOD[2000:3200,1000:2000]
    #
    #
    # mask = io.loadmat('./data1/Paleogene_three/mask3.mat')["mask3"]
    # trainmask = mask[:2000, 80:1080]
    # testmask = mask[2000:3200,1000:2000]  # trainmask和testmask在这里直接截取出来
    #
    # # label和mask需要在MATLAB提前处理出来
    # label = io.loadmat('./data1/Paleogene_three/label3.mat')["label3"]
    # trainlabel = label[:2000, 80:1080]
    # testlabel = label[2000:3200,1000:2000]  # 从label中截取trainlabel和testlabel

#——————————————————————白垩系————————————————————————————————————————————
    # cut_gamma = data_GES[:, :8700]  # 截取白垩系4338×3900的数据
    # traindata = cut_gamma[:, :4000]  # 2000×2400训练数据
    # testdata = cut_gamma[5000:, 3000:5000]  # 2000×3900测试数据
    #
    # cut_PPMA = data_PPMA[:, :8700]
    # trainchan1 = cut_PPMA[:, :4000]
    # testchan1 = cut_PPMA[5000:, 3000:5000]
    #
    # cut_TRF = data_TRF[:, :8700]
    # trainchan2 = cut_TRF[:, :4000]
    # testchan2 = cut_TRF[5000:, 3000:5000]
    #
    # cut_TOD = data_TOD[:, :8700]
    # trainchan3 = cut_TOD[:, :4000]
    # testchan3 = cut_TOD[5000:, 3000:5000]
    #
    # mask = io.loadmat('./data1/Cretaceous_nine/mask9.mat')["mask9"]
    # trainmask = mask[:, :4000]
    # testmask = mask[5000:, 3000:5000]  # trainmask和testmask在这里直接截取出来
    #
    # # label和mask需要在MATLAB提前处理出来
    # label = io.loadmat('./data1/Cretaceous_nine/label9.mat')["label9"]
    # trainlabel = label[:, :4000]
    # testlabel = label[5000:, 3000:5000]  # 从label中截取trainlabel和testlabel

#——————————————————————泥盆系-石炭系———————————————————————————————————————
    # cut_gamma = data_GES[3700:, 2260:5460] #截取泥盆系-石炭系2500×1600的数据
    # traindata = cut_gamma[500:4500, 1000:2800]  #1000×900训练数据
    # testdata = cut_gamma[:2000 , :2000]   #1500×1600测试数据
    #
    # cut_PPMA = data_PPMA[3700:, 2260:5460]
    # trainchan1 = cut_PPMA[500:4500, 1000:2800]
    # testchan1 = cut_PPMA[:2000, :2000]
    #
    # cut_TRF = data_TRF[3700:, 2260:5460]
    # trainchan2 = cut_TRF[500:4500, 1000:2800]
    # testchan2 = cut_TRF[:2000, :2000]
    #
    # cut_TOD = data_TOD[3700:, 2260:5460]
    # trainchan3 = cut_TOD[500:4500, 1000:2800]
    # testchan3 = cut_TOD[:2000, :2000]
    #
    # mask = io.loadmat('./data1/DevoCarb_five/maskV.mat')["maskV"]
    # trainmask = mask[500:4500, 1000:2800]
    # testmask = mask[:2000, :2000]  # trainmask和testmask在这里直接截取出来
    #
    # # label和mask需要在MATLAB提前处理出来
    # label = io.loadmat('./data1/DevoCarb_five/labelV.mat')["labelV"]
    # trainlabel = label[500:4500, 1000:2800]
    # testlabel = label[:2000, :2000]  # 从label中截取trainlabel和testlabel
#——————————————————————震旦系-寒武系——————————————————————————————————————-——
    # cut_gamma = data_GES[:, 5000:11100]
    # traindata = cut_gamma[3200:8700, 600:3600]  #再截取(0:1000)1000的列作为训练数据
    # testdata = cut_gamma[:2000,:]  #取最后的2000列为testdata
    #
    # cut_PPMA = data_PPMA[:, 5000:11100]
    # trainchan1 = cut_PPMA[3200:8700, 600:3600]  # 1001—>1501取一半的训练数据
    # testchan1 = cut_PPMA[:2000,:]
    #
    # cut_TRF = data_TRF[:, 5000:11100]
    # trainchan2 = cut_TRF[3200:8700, 600:3600]
    # testchan2 = cut_TRF[:2000,:]
    #
    # cut_TOD = data_TOD[:, 5000:11100]
    # trainchan3 = cut_TOD[3200:8700, 600:3600]
    # testchan3 = cut_TOD[:2000,:]
    #
    # mask = io.loadmat('./data1/SinCam_six/mask6.mat')["mask6"]
    # trainmask = mask[3200:8700, 600:3600]
    # testmask = mask[:2000,:]  # trainmask和testmask在这里直接截取出来
    #
    # # label和mask需要在MATLAB提前处理出来
    # label = io.loadmat('./data1/SinCam_six/label6.mat')["label6"]
    # trainlabel = label[3200:8700, 600:3600]
    # testlabel = label[:2000,:]  # 从label中截取trainlabel和testlabel

#——————————————————————花岗岩区————————————————————————————————————————————
    cut_gamma = data_GES[:, 8100:]
    traindata = cut_gamma[3000:7000, 1000:]  #再截取(0:1000)1000的列作为训练数据
    testdata = cut_gamma[:3000, :]  #取最后的2000列为testdata

    cut_PPMA = data_PPMA[:, 8100:]
    trainchan1 = cut_PPMA[3000:7000, 1000:]  # 1001—>1501取一半的训练数据
    testchan1 = cut_PPMA[:3000, :]

    cut_TRF = data_TRF[:, 8100:]
    trainchan2 = cut_TRF[3000:7000, 1000:]
    testchan2 = cut_TRF[:3000, :]

    cut_TOD = data_TOD[:, 8100:]
    trainchan3 = cut_TOD[3000:7000, 1000:]
    testchan3 = cut_TOD[:3000, :]

    mask = io.loadmat('./data1/Granite_three/mask3.mat')["mask3"]
    trainmask = mask[3000:7000, 1000:]
    testmask = mask[:3000, :]  # trainmask和testmask在这里直接截取出来

    # label和mask需要在MATLAB提前处理出来
    label = io.loadmat('./data1/Granite_three/label3.mat')["label3"]
    trainlabel = label[3000:7000, 1000:]
    testlabel = label[:3000, :]  # 从label中截取trainlabel和testlabel

#——————————————————————————————————细分训练————————————————————————————————————————————
    train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,
          trainchan1, trainchan2, trainchan3, testchan1, testchan2, testchan3) #trainchan3,

# train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,#)
    #       trainchan1,  trainchan3, testchan1,  testchan3)   #trainchan2, testchan2,








# import cv2
# import os
# import numpy as np
# from torch.utils import data as data_
# from tqdm import tqdm
# from utils.config import opt
# from utils.dataset import Dataset
# from model.Trainer import InpTrainer
# import scipy.io as io
# import torch
# import matplotlib.pyplot as plt
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# #torch.cuda.current_device()
# SAVEROOT='./result/LYT1221/'
# if not os.path.exists(SAVEROOT):  #判断目录是否存在，如果不存在就创建相关目录
#     os.makedirs(SAVEROOT)
#
# def test_model(dataloader, model, ifsave=True, test_num=1000, name='test'):  #100000
#                                                                         #按理说，这里的test也占有内存
#     num = 0
#     ACC=0
#     model.eval()   #不用Batch Normalization 和 Dropout
#     #for ii, (data, label, mask) in enumerate(dataloader):
#     for ii, (data, label, mask, testchan1,testchan2, testchan3) in enumerate(dataloader): #enumerate 枚举，列举    , testchan3
#         # 对于一个可迭代的或者可遍历的对象，enumerate将其组成一个索引序列，利用它可以同时获得索引以及对应的值
#
#         data = torch.cat((data, testchan1, testchan3), dim = 1)   #, testchan3
#
#         #with torch.no_grad(): #with torch.no_grad()或者 @torch.no_grad() 中的数据不需要计算梯度，也不会进行反向传播
#         outputimg = model.test_onebatch(data.cuda())
#         num += 1
#         if ifsave:
#             for i in range(1):  #range(5)是从0-5并且不包含5
#                 inputimg = data[i][0].numpy()   #.numpy()是将Tensor转换为ndarray(即n维数组)
#                 labelimg = label[i].numpy()
#                 outimg = outputimg[i].detach()
#                 _, outidex = torch.max(outimg, 0)   #torch.max(input, dim)，其中input是softmax函数输出的一个tensor
#                                                                      #dim是max函数索引的维度（取值0或者1），0代表每一列的最大值，1代表每一行的最大值
#                                                                      #返回两个tensor，第一个是每行的最大值，softmax的输出最大是1，所以第一个tensor是全1的tensor
#                                                                      #第二个tensor是每行最大值的索引
#
#                 outidex = outidex.cpu().numpy()     #将数据从gpu转成cpu
#                 inputimg = ((inputimg+1)*255/2).astype(np.uint8) #astype转换数据类型
#                 labelimg = (255 * labelimg /6).astype(np.uint8)
#                 outidex = (255*outidex/6).astype(np.uint8)
#
#                 # cv2.imwrite()用于将图像保存到任何存储设备 cv2.imwrite(filename, image)
#                 cv2.imwrite(SAVEROOT + name+'outimg' + str(ii) + '.jpg', outidex) #cv2.imwrite()保存图像
#                 cv2.imwrite(SAVEROOT + name+'inputimg' + str(ii) + '.jpg', inputimg)
#                 cv2.imwrite(SAVEROOT + name + 'labelimg' + str(ii) + '.jpg', labelimg)
#
#         if ii > test_num:
#             break
#     #ACC/num输出准确度
#     return #outputimg
#
# #def train(traindata, trainlabel, trainmask,testdata, testlabel,testmask):
# def train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,trainchan1, trainchan3,
#            testchan1, testchan3):   # trainchan3,,  testchan3
#     opt._parse()   #把参数打印出来
#     train_dataset = Dataset(traindata, trainlabel, trainmask, 'train', 128, 5, 10000,  #最开始是10000
#                             trainchan1, trainchan3)        #裁剪的训练图片大小为128×128，scale=10
#     train_dataloader = data_.DataLoader(train_dataset,
#                                          batch_size=50,  #batch_size 的大小
#                                          shuffle=True)   #shuffle 洗牌，即是否打乱数据集
#
#     test_dataset = Dataset(testdata, testlabel, testmask, 'test', 128, 5, 1000,
#                            testchan1, testchan3)
#     test_dataloader = data_.DataLoader(test_dataset,
#                                        batch_size=40,  #test_dataloader应该也会占内存
#                                        shuffle=False)
#
#     trainer = InpTrainer(opt)
#     if opt.load_net:
#         trainer.load_net(opt.load_net)
#     print('model construct completed')
#
#     if opt.test_only:
#         eval_result1 = test_model(test_dataloader, trainer, ifsave=True, test_num=opt.test_num, name='test')
#         print('eval_loss: ', eval_result1)
#         return
#
#     iteration = 30   #设置学习的轮数
#     LOSS = np.zeros(iteration)
#     for e in range(iteration):
#         for ii, (data, label, mask, trainchan1, trainchan3) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
#         #for ii, (data, label, mask) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
#             data = torch.cat((data, trainchan1, trainchan3), dim=1)   #, trainchan3
#             trainer.train()
#             loss = trainer.train_onebatch(data.cuda(), label.cuda(), mask.cuda())
#         print('loss:', loss.detach().cpu().numpy())
#         LOSS[e] = loss  #保存loss的值
#         test_model(test_dataloader, trainer, ifsave=True, test_num=opt.test_num, name='test')
#         savemodel =trainer.save_net(best_map=e)
#         print("save to %s !" % savemodel)
#
#     x_value = list(range(iteration))    #range(n)是从0开始，n-1结束
#     y_value = LOSS
#     y = np.mat(y_value)
#     io.savemat( './plt/rough_five/y1.mat', {'y1':y})  #{}中是个字典
#     plt.plot(x_value, y_value, c='blue', marker = "o", markersize = 3)      # 用plot函数绘制折线图，线条颜色设置为蓝色
#     plt.title('Train_Loss', fontsize=24)       # 设置图表标题和标题字号
#     #plt.tick_params(axis='both', which='major', labelsize=14)    # 设置刻度的字号
#     plt.xlabel('epoch', fontsize=14)   # 设置x轴标签及其字号
#     plt.ylabel('loss', fontsize=14)    # 设置y轴标签及其字号
#     plt.savefig('./plt/rough_five/p.jpg')
#     plt.show()      # 显示图表
#
# if __name__ == '__main__':  #__name__内置变量，用于表示当前模块的名字
# #————————————————————包含多个通道的数据—————————————————————————————————————
#     #注意这里先将伽马能谱数据导入进来，再切割为细分的训练和测试数据
#     data_GES = io.loadmat('./data1/GES.mat')["GES"]
#     scale0 = np.max(np.abs(data_GES))
#     data_GES = data_GES / scale0   #直接对全部的伽马能谱数据归一化
#
#     gray_solid = io.loadmat('./data1/gray1.mat')["gray1"]
#     gray_dotted = io.loadmat('./data1/gray2.mat')["gray2"]
#     red_solid = io.loadmat('./data1/red.mat')["red"]
#     purple_dotted = io.loadmat('./data1/purple.mat')["purple"]
#
#     # 对于PPMA、TRF和Cdata的处理方法与之类似
#     data_PPMA = io.loadmat('./data1/PPMA.mat')["PPMA"]  # ΔT化极后磁异常
#     scale1 = np.max(np.abs(data_PPMA))
#     data_PPMA = data_PPMA / scale1
#
#     data_TRF = io.loadmat('./data1/TRF.mat')["TRF"]  # ΔT磁异常化极上延2km剩余场
#     scale2 = np.max(np.abs(data_TRF))
#     data_TRF = data_TRF / scale2
#
#     data_TOD = io.loadmat('./data1/TOD.mat')["TOD"]  # 航磁资小波多尺度分析的三阶细节
#     scale3 = np.max(np.abs(data_TOD))
#     data_TOD = data_TOD / scale3
#
# ##————————————————————————粗分数据处理————————————————————————————————————
# ##——————————————————————————————————————————————————————————————————
#     # #gamma能谱
#     traindata = data_GES[1500:7000, : ] #训练数据
#     testdata = data_GES[:2000, :]  #取最后的2000列为testdata
#     trainchan1 = red_solid[1500:7000, : ]
#     testchan1 = red_solid[:2000,:]
#     trainchan3 = gray_solid[1500:7000, : ]
#     testchan3 = gray_solid[:2000,:]
#
#     mask = io.loadmat('./data1/rough_five/mask5.mat')["mask5"]
#     trainmask = mask[1500:7000, : ]
#     testmask = mask[:2000, :]
#
#     label = io.loadmat('./data1/rough_five/label5.mat')["label5"]
#     trainlabel = label[1500:7000, : ]
#     testlabel = label[:2000, :]   # 从label中截取trainlabel和testlabel
#
#     train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,
#           trainchan1, trainchan3, testchan1, testchan3)
#
# #********************************细分数据处理******************************************
# #——————————————————————古近系————————————————————————————————————————————
#     # cut_GES = data_GES[:3200, :2000]  # 截取古近系2700×1600的数据
#     # traindata = cut_GES[:1500, :1200]  # 2700×400训练数据
#     # testdata = cut_GES[2000:3200,1000:2000]  # 2700×1200测试数据
#     #
#     # cut_PPMA = data_PPMA[:3200, :2000]
#     # trainchan1 = cut_PPMA[:1500, :1200]
#     # testchan1 = cut_PPMA[2000:3200,1000:2000]
#     #
#     # cut_TRF = data_TRF[:3200, :2000]
#     # trainchan2 = cut_TRF[:1500, :1200]
#     # testchan2 = cut_TRF[2000:3200,1000:2000]
#     #
#     # cut_TOD = data_TOD[:3200, :2000]
#     # trainchan3 = cut_TOD[:1500, :1200]
#     # testchan3 = cut_TOD[2000:3200,1000:2000]
#     #
#     #
#     # mask = io.loadmat('./data1/Paleogene_three/mask3.mat')["mask3"]
#     # trainmask = mask[:1500, :1200]
#     # testmask = mask[2000:3200,1000:2000]  # trainmask和testmask在这里直接截取出来
#     #
#     # # label和mask需要在MATLAB提前处理出来
#     # label = io.loadmat('./data1/Paleogene_three/label3.mat')["label3"]
#     # trainlabel = label[:1500, :1200]
#     # testlabel = label[2000:3200,1000:2000]  # 从label中截取trainlabel和testlabel
#
# #——————————————————————白垩系————————————————————————————————————————————
#     # cut_gamma = data_GES[:, :8700]  # 截取白垩系4338×3900的数据
#     # traindata = cut_gamma[:, :3000]  # 2000×2400训练数据
#     # testdata = cut_gamma[5000:, 3000:5000]  # 2000×3900测试数据
#     #
#     # cut_PPMA = data_PPMA[:, :8700]
#     # trainchan1 = cut_PPMA[:, :3000]
#     # testchan1 = cut_PPMA[5000:, 3000:5000]
#     #
#     # cut_TRF = data_TRF[:, :8700]
#     # trainchan2 = cut_TRF[:, :3000]
#     # testchan2 = cut_TRF[5000:, 3000:5000]
#     #
#     # cut_TOD = data_TOD[:, :8700]
#     # trainchan3 = cut_TOD[:, :3000]
#     # testchan3 = cut_TOD[5000:, 3000:5000]
#     #
#     # mask = io.loadmat('./data1/Cretaceous_nine/mask9.mat')["mask9"]
#     # trainmask = mask[:, :3000]
#     # testmask = mask[5000:, 3000:5000]  # trainmask和testmask在这里直接截取出来
#     #
#     # # label和mask需要在MATLAB提前处理出来
#     # label = io.loadmat('./data1/Cretaceous_nine/label9.mat')["label9"]
#     # trainlabel = label[:, :3000]
#     # testlabel = label[5000:, 3000:5000]  # 从label中截取trainlabel和testlabel
#
# #——————————————————————泥盆系-石炭系———————————————————————————————————————
#     # cut_gamma = data_GES[3700:, 2260:5460] #截取泥盆系-石炭系2500×1600的数据
#     # traindata = cut_gamma[3500:5000, :3200]  #1000×900训练数据
#     # testdata = cut_gamma[:2000 , :2000]   #1500×1600测试数据
#     #
#     # cut_PPMA = data_PPMA[3700:, 2260:5460]
#     # trainchan1 = cut_PPMA[3500:5000, :3200]
#     # testchan1 = cut_PPMA[:2000 , :2000]
#     #
#     # cut_TRF = data_TRF[3700:, 2260:5460]
#     # trainchan2 = cut_TRF[3500:5000, :3200]
#     # testchan2 = cut_TRF[:2000 , :2000]
#     #
#     # cut_TOD = data_TOD[3700:, 2260:5460]
#     # trainchan3 = cut_TOD[3500:5000, :3200]
#     # testchan3 = cut_TOD[:2000 , :2000]
#     #
#     # mask = io.loadmat('./data1/DevoCarb_five/maskV.mat')["maskV"]
#     # trainmask = mask[3500:5000, :3200]
#     # testmask = mask[:2000 , :2000]  # trainmask和testmask在这里直接截取出来
#     #
#     # # label和mask需要在MATLAB提前处理出来
#     # label = io.loadmat('./data1/DevoCarb_five/labelV.mat')["labelV"]
#     # trainlabel = label[3500:5000, :3200]
#     # testlabel = label[:2000 , :2000]  # 从label中截取trainlabel和testlabel
# #——————————————————————震旦系-寒武系——————————————————————————————————————-——
#     # cut_gamma = data_GES[:, 5000:11100]
#     # traindata = cut_gamma[2000:8700, :2000]  #再截取(0:1000)1000的列作为训练数据
#     # testdata = cut_gamma[:2000,:]  #取最后的2000列为testdata
#     #
#     # cut_PPMA = data_PPMA[:, 5000:11100]
#     # trainchan1 = cut_PPMA[2000:8700, :2000]  # 1001—>1501取一半的训练数据
#     # testchan1 = cut_PPMA[:2000,:]
#     #
#     # cut_TRF = data_TRF[:, 5000:11100]
#     # trainchan2 = cut_TRF[2000:8700, :2000]
#     # testchan2 = cut_TRF[:2000,:]
#     #
#     # cut_TOD = data_TOD[:, 5000:11100]
#     # trainchan3 = cut_TOD[2000:8700, :2000]
#     # testchan3 = cut_TOD[:2000,:]
#
#     # mask = io.loadmat('./data1/SinCam_six/mask6.mat')["mask6"]
#     # trainmask = mask[2000:8700, :2000]
#     # testmask = mask[:2000,:]  # trainmask和testmask在这里直接截取出来
#     #
#     # # label和mask需要在MATLAB提前处理出来
#     # label = io.loadmat('./data1/SinCam_six/label6.mat')["label6"]
#     # trainlabel = label[2000:8700, :2000]
#     # testlabel = label[:2000,:]  # 从label中截取trainlabel和testlabel
#
# #——————————————————————花岗岩区————————————————————————————————————————————
#     # cut_gamma = data_GES[:, 8100:]
#     # traindata = cut_gamma[3000:7000, 1000:]  #再截取(0:1000)1000的列作为训练数据
#     # testdata = cut_gamma[:3000, :]  #取最后的2000列为testdata
#     #
#     # cut_PPMA = data_PPMA[:, 8100:]
#     # trainchan1 = cut_PPMA[3000:7000, 1000:]  # 1001—>1501取一半的训练数据
#     # testchan1 = cut_PPMA[:3000, :]
#     #
#     # cut_TRF = data_TRF[:, 8100:]
#     # trainchan2 = cut_TRF[3000:7000, 1000:]
#     # testchan2 = cut_TRF[:3000, :]
#     #
#     # cut_TOD = data_TOD[:, 8100:]
#     # trainchan3 = cut_TOD[3000:7000, 1000:]
#     # testchan3 = cut_TOD[:3000, :]
#     #
#     # mask = io.loadmat('./data1/Granite_three/mask3.mat')["mask3"]
#     # trainmask = mask[3000:7000, 1000:]
#     # testmask = mask[:3000, :]  # trainmask和testmask在这里直接截取出来
#     #
#     # # label和mask需要在MATLAB提前处理出来
#     # label = io.loadmat('./data1/Granite_three/label3.mat')["label3"]
#     # trainlabel = label[3000:7000, 1000:]
#     # testlabel = label[:3000, :]  # 从label中截取trainlabel和testlabel
#
# #——————————————————————————————————细分训练————————————————————————————————————————————
#     # train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,
#     #       trainchan1, trainchan2, trainchan3,testchan1, testchan2, testchan3)  # , testchan3，trainchan2, testchan2,
#
# # train(traindata, trainlabel, trainmask, testdata, testlabel, testmask,#)
#     #       trainchan1,  trainchan3, testchan1,  testchan3)   #trainchan2, testchan2,
#
#
