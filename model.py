

import torch
from torch import nn

#基本残差块
class BasicBlock(nn.Module):
    #   第一个layer不要downsample,后面的统一在第一个block进行downsample
    def __init__(self,in_ch,block_ch,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        #   可能是虚线也可能是实线，通过downsample来调整
        self.downsample=downsample
        self.conv1=nn.Conv2d(in_ch,block_ch,kernel_size=3,stride=stride,padding=1,bias=False)
        #   使用BatchNorm bias设置为False
        self.bn1=nn.BatchNorm2d(block_ch)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(block_ch,block_ch,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(block_ch)
        self.relu2=nn.ReLU()

    def forward(self,x):
        identity=x
        #   如果是虚线，那就需要进行对其
        if self.downsample is not None:
            identity=self.downsample(x)

        #进行前向传播
        out=self.relu1(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=identity
        return self.relu2(out)

#瓶颈残差块
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, in_ch, block_ch, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        #   设置为1*1的卷积
        self.conv1 = nn.Conv2d(in_ch, block_ch, kernel_size=1, bias=False)
        #   使用BatchNorm bias设置为False
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_ch, block_ch, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(block_ch)
        self.relu2 = nn.ReLU()
        self.conv3=nn.Conv2d(block_ch,block_ch*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(block_ch*self.expansion)
        self.relu3=nn.ReLU()

    def forward(self, x):
        identity = x

        #   如果是虚线，那就需要进行对其
        if self.downsample is not None:
            identity = self.downsample(x)

        # 进行前向传播
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out=self.bn3(self.conv3(out))
        return self.relu3(out+identity)

class ResNet(nn.Module):
    #block_num 表示残差块怎样进行堆叠
    def __init__(self,in_ch=3,num_classes=100,block=Bottleneck,block_num=[3,4,6,3]):
        super(ResNet, self).__init__()
        #通过 in_ch 对输入通道进行跟踪
        self.in_ch=in_ch
        self.conv1=nn.Conv2d(in_ch,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.maxpool1=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.in_ch=64
        self.layer1=self._make_layer(block,64,block_num[0])
        self.layer2 = self._make_layer(block, 128, block_num[1],stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2],stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3],stride=2)
        self.fc_layer=nn.Sequential(
            nn.Linear(512*block.expansion*7*7,num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        print(f'out shape:{x.shape}')
        out = self.maxpool1(self.bn1(self.conv1(x)))  #(1,3,224,224)->(1,64,56,56)
        print(f'out shape:{out.shape}')
        out = self.layer1(out)
        print(f'out shape:{out.shape}')
        out = self.layer2(out)
        print(f'out shape:{out.shape}')
        out = self.layer3(out)
        print(f'out shape:{out.shape}')
        out = self.layer4(out)
        print(f'out shape:{out.shape}')
        out = out.reshape(out.shape[0],-1)
        print(f'out shape:{out.shape}')
        out = self.fc_layer(out)
        return out
    def _make_layer(self,block,block_ch,block_num,stride=1):
        layers=[]
        #downsample效果和block效果是一样的
        downsample=nn.Conv2d(self.in_ch,block_ch*block.expansion,kernel_size=1,stride=stride)
        layers+=[block(self.in_ch,block_ch,stride=stride,downsample=downsample)]
        self.in_ch=block_ch*block.expansion

        for _ in range(1,block_num):
            layers+=[block(self.in_ch,block_ch)]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    resnet50=ResNet(in_ch=3,num_classes=100,block=Bottleneck,block_num=[2,2,2,2])
    y=resnet50(x)
    print(y.shape)