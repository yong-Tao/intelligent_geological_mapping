from torch.utils.data import dataset
import numpy as np
import random

#pytorch读取图片，主要通过Dataset类，Dataset类作为所有datasets的基类，所有的datasets都要继承它
#通过继承 torch.utils.data.Dataset来实现，在继承的时候，要override(重写)三个方法：
#一是init： 用来初始化一些有关操作数据集的参数
#二是getitem：定义数据获取的方式（包括读取数据，对数据进行变换等），该方法支持从0到len(self)-1的索引。obj[index]等价于obj.getitem
#三是len：获取数据集的大小，len(obj)等价于obj.len()
class Dataset(dataset.Dataset):
    def __init__(self, data,label, mask, mode, cropsize, scale, num,
                  red_solid, purple_dotted, gray_solid):   #, gray_solid, gray_dotted):   #purple_dotted
        self.data = data
        self.label = label
        self.mask = mask
        self.mode = mode
        self.cropsize = cropsize
        self.scale = scale
        self.red_solid = red_solid
        self.purple_dotted = purple_dotted
        self.gray_solid = gray_solid
        # self.gray_dotted = gray_dotted
        random1 = np.random.RandomState(0)
        self.seed = random1.randint(100000, size=num)

    def __len__(self):  #获取数据集的大小
        return len(self.seed)

    def __getitem__(self, index): #定义数据获取的方式，返回第index个样本的具体数据
        data = self.data
        label = self.label
        mask = self.mask
        scale = self.scale
        h, w = np.shape(data)
        red_solid = self.red_solid
        purple_dotted = self.purple_dotted
        gray_solid = self.gray_solid
        # gray_dotted = self.gray_dotted

        if self.mode == 'train':

            hstar = random.randint(0, h-self.cropsize*scale)
            wstar = random.randint(0, w-self.cropsize*scale)
            data = data[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            label = label[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            mask = mask[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            red_solid = red_solid[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            purple_dotted = purple_dotted[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            gray_solid = gray_solid[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            # gray_dotted = gray_dotted[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]

        else:
            random1 = np.random.RandomState(self.seed[index])
            hstar = random1.randint(0, h-self.cropsize*scale)
            wstar = random1.randint(0, w-self.cropsize*scale)
            data = data[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            label = label[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            mask = mask[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            red_solid = red_solid[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            purple_dotted = purple_dotted[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            gray_solid = gray_solid[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
            # gray_dotted = gray_dotted[hstar:hstar+self.cropsize*scale:scale, wstar:wstar+self.cropsize*scale:scale]
        data = np.reshape(data, (1, self.cropsize, self.cropsize)).astype(np.float32)
        label = (label * mask).astype(np.longlong)
        mask = np.reshape(mask, (1, self.cropsize, self.cropsize)).astype(np.float32)
        red_solid = np.reshape(red_solid, (1, self.cropsize, self.cropsize)).astype(np.float32)
        purple_dotted = np.reshape(purple_dotted, (1, self.cropsize, self.cropsize)).astype(np.float32)
        gray_solid = np.reshape(gray_solid, (1, self.cropsize, self.cropsize)).astype(np.float32)
        # gray_dotted = np.reshape(gray_dotted, (1, self.cropsize, self.cropsize)).astype(np.float32)

        return data, label, mask,red_solid, purple_dotted, gray_solid#, gray_dotted#, purple_dotted,
