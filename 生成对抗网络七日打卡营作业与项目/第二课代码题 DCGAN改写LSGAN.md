## 将第一节课实践的DCGAN代码改成lsgan的损失函数

可以看下有提示的地方。


```python
#导入一些必要的包
import os
import random
import paddle 
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.datasets as dset
import paddle.vision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

## 导入MNIST手写数据集，并做数据增强
第一次运行会通过paddle的API自动下载数据集


```python
dataset = paddle.vision.datasets.MNIST(mode='train', #取得是训练集
                                        transform=transforms.Compose([
                                        # resize ->(32,32)
                                        transforms.Resize((32,32)),
                                        # 归一化到-1~1
                                        transforms.Normalize([127.5], [127.5])#mean, std, 单通道
                                    ]))

dataloader = paddle.io.DataLoader(dataset, batch_size=32,
                                  shuffle=True, num_workers=4)
```

    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz 
    Begin to download
    
    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz 
    Begin to download
    ........
    Download finished


## 定义用于参数初始化的函数


```python
#参数初始化的模块
@paddle.no_grad()
def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)#正态分布采样
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def uniform_(x, a=-1., b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)#均匀分布采样
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def constant_(x, value):
    temp_value = paddle.full(x.shape, value, x.dtype)#设置为常数
    x.set_value(temp_value)
    return x

def weights_init(m):#权重初始化函数
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        normal_(m.weight, 1.0, 0.02)
        constant_(m.bias, 0)
```

## 生成器模型


```python
# Generator Code
class Generator(nn.Layer):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input is Z, [B, 100, 1, 1] -> [B, 64 * 4, 4, 4]
            nn.Conv2DTranspose(100, 64 * 4, 4, 1, 0, bias_attr=False),
            nn.BatchNorm2D(64 * 4),
            nn.ReLU(True),
            # state size. [B, 64 * 4, 4, 4] -> [B, 64 * 2, 8, 8]
            nn.Conv2DTranspose(64 * 4, 64 * 2, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(64 * 2),
            nn.ReLU(True),
            # state size. [B, 64 * 2, 8, 8] -> [B, 64, 16, 16]
            nn.Conv2DTranspose( 64 * 2, 64, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(True),
            # state size. [B, 64, 16, 16] -> [B, 1, 32, 32]
            nn.Conv2DTranspose( 64, 1, 4, 2, 1, bias_attr=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


netG = Generator()
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)
```

    Generator(
      (gen): Sequential(
        (0): Conv2DTranspose(100, 256, kernel_size=[4, 4], data_format=NCHW)
        (1): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        (2): ReLU(name=True)
        (3): Conv2DTranspose(256, 128, kernel_size=[4, 4], stride=[2, 2], padding=1, data_format=NCHW)
        (4): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (5): ReLU(name=True)
        (6): Conv2DTranspose(128, 64, kernel_size=[4, 4], stride=[2, 2], padding=1, data_format=NCHW)
        (7): BatchNorm2D(num_features=64, momentum=0.9, epsilon=1e-05)
        (8): ReLU(name=True)
        (9): Conv2DTranspose(64, 1, kernel_size=[4, 4], stride=[2, 2], padding=1, data_format=NCHW)
        (10): Tanh()
      )
    )


## 判别器模型


```python
class Discriminator(nn.Layer):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(

            # input [B, 1, 32, 32] -> [B, 64, 16, 16]
            nn.Conv2D(1, 64, 4, 2, 1, bias_attr=False),
            nn.LeakyReLU(0.2),

            # state size. [B, 64, 16, 16] -> [B, 128, 8, 8]
            nn.Conv2D(64, 64 * 2, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(64 * 2),
            nn.LeakyReLU(0.2),

            # state size. [B, 128, 8, 8] -> [B, 256, 4, 4]
            nn.Conv2D(64 * 2, 64 * 4, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(64 * 4),
            nn.LeakyReLU(0.2),

            # state size. [B, 256, 4, 4] -> [B, 1, 1, 1]
            nn.Conv2D(64 * 4, 1, 4, 1, 0, bias_attr=False),
            # 这里为需要改变的地方
            #nn.Sigmoid()
            #######删掉即行
        )

    def forward(self, x):
        return self.dis(x)

netD = Discriminator()
netD.apply(weights_init)
print(netD)
```

    Discriminator(
      (dis): Sequential(
        (0): Conv2D(1, 64, kernel_size=[4, 4], stride=[2, 2], padding=1, data_format=NCHW)
        (1): LeakyReLU(negative_slope=0.2)
        (2): Conv2D(64, 128, kernel_size=[4, 4], stride=[2, 2], padding=1, data_format=NCHW)
        (3): BatchNorm2D(num_features=128, momentum=0.9, epsilon=1e-05)
        (4): LeakyReLU(negative_slope=0.2)
        (5): Conv2D(128, 256, kernel_size=[4, 4], stride=[2, 2], padding=1, data_format=NCHW)
        (6): BatchNorm2D(num_features=256, momentum=0.9, epsilon=1e-05)
        (7): LeakyReLU(negative_slope=0.2)
        (8): Conv2D(256, 1, kernel_size=[4, 4], data_format=NCHW)
      )
    )


## 参数及优化器配置，训练并可视化结果


```python
# Initialize BCELoss function
# 这里为需要改变的地方
#loss = nn.BCELoss()
#改为Initialize MSELoss function
loss = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = paddle.randn([32, 100, 1, 1], dtype='float32')

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=0.0002, beta1=0.5, beta2=0.999)
optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=0.0002, beta1=0.5, beta2=0.999)

```


```python
losses = [[], []]
#plt.ion()
now = 0
for pass_id in range(20):#只训练了20个epoch，太多提交后可能无法展示
    for batch_id, (data, target) in enumerate(dataloader):
        ############################
        # (1) Update D network: min(D(x) - 1)^2 + (D(G(z)) - 0)^2
        ###########################

        optimizerD.clear_grad()#清理梯度，防止梯度累加
        real_img = data
        bs_size = real_img.shape[0]#batch_size
        label = paddle.full((bs_size, 1, 1, 1), real_label, dtype='float32')#全1label
        real_out = netD(real_img)
        #这里计算的是(D(x) - 1)^2
        errD_real = loss(real_out, label)
        errD_real.backward()

        noise = paddle.randn([bs_size, 100, 1, 1], 'float32')#随机采样噪声作为生成器输入
        fake_img = netG(noise)
        label = paddle.full((bs_size, 1, 1, 1), fake_label, dtype='float32')#全0label
        fake_out = netD(fake_img.detach())
        #这里计算的是D(G(z) - 0)^2
        errD_fake = loss(fake_out,label)
        errD_fake.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        errD = errD_real + errD_fake#总损失
        losses[0].append(errD.numpy()[0])

        ############################
        # (2) Update G network: min(D(G(z)) - 1)^2
        ###########################
        optimizerG.clear_grad()
        noise = paddle.randn([bs_size, 100, 1, 1],'float32')
        fake = netG(noise)
        label = paddle.full((bs_size, 1, 1, 1), real_label, dtype=np.float32,)
        output = netD(fake)
        errG = loss(output,label)
        errG.backward()
        optimizerG.step()
        optimizerG.clear_grad()

        losses[1].append(errG.numpy()[0])


        ############################
        # visualize
        ###########################
        if batch_id % 100 == 0:
            generated_image = netG(noise).numpy()
            imgs = []
            plt.figure(figsize=(15,15))
            try:
                for i in range(10):
                    image = generated_image[i].transpose()
                    image = np.where(image > 0, image, 0)
                    image = image.transpose((1,0,2))
                    plt.subplot(10, 10, i + 1)
                    
                    plt.imshow(image[...,0], vmin=-1, vmax=1)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplots_adjust(wspace=0.1, hspace=0.1)
                msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(pass_id, batch_id, errD.numpy()[0], errG.numpy()[0])
                print(msg)
                plt.suptitle(msg,fontsize=20)
                plt.draw()
                plt.savefig('{}/{:04d}_{:04d}.png'.format('work', pass_id, batch_id), bbox_inches='tight')
                plt.pause(0.01)
            except IOError:
                print(IOError)
    paddle.save(netG.state_dict(), "work/generator.params")
```

    Epoch ID=0 Batch ID=0 
    
     D-Loss=3.400453805923462 G-Loss=1.8688591718673706



![png](output_12_1.png)


    Epoch ID=0 Batch ID=100 
    
     D-Loss=2.6674020290374756 G-Loss=0.14542219042778015



![png](output_12_3.png)


    Epoch ID=0 Batch ID=200 
    
     D-Loss=1.0487353801727295 G-Loss=0.1208115741610527



![png](output_12_5.png)


    Epoch ID=0 Batch ID=300 
    
     D-Loss=0.13467498123645782 G-Loss=1.3784295320510864
    Epoch ID=0 Batch ID=400 
    
     D-Loss=0.338588684797287 G-Loss=0.380012571811676



![png](output_12_7.png)



![png](output_12_8.png)


    Epoch ID=0 Batch ID=500 
    
     D-Loss=0.09796582162380219 G-Loss=0.7588832378387451



![png](output_12_10.png)


    Epoch ID=0 Batch ID=600 
    
     D-Loss=0.09808957576751709 G-Loss=1.0171210765838623



![png](output_12_12.png)


    Epoch ID=0 Batch ID=700 
    
     D-Loss=0.12590503692626953 G-Loss=0.3874339461326599



![png](output_12_14.png)


    Epoch ID=0 Batch ID=800 
    
     D-Loss=0.1605311632156372 G-Loss=1.2766892910003662



![png](output_12_16.png)


    Epoch ID=0 Batch ID=900 
    
     D-Loss=0.1327817142009735 G-Loss=0.7916567325592041



![png](output_12_18.png)


    Epoch ID=0 Batch ID=1000 
    
     D-Loss=0.10024905204772949 G-Loss=1.2787437438964844



![png](output_12_20.png)


    Epoch ID=0 Batch ID=1100 
    
     D-Loss=0.1922587901353836 G-Loss=0.6846257448196411



![png](output_12_22.png)


    Epoch ID=0 Batch ID=1200 
    
     D-Loss=0.06421608477830887 G-Loss=1.077313780784607



![png](output_12_24.png)


    Epoch ID=0 Batch ID=1300 
    
     D-Loss=0.0964384377002716 G-Loss=0.9019376039505005



![png](output_12_26.png)


    Epoch ID=0 Batch ID=1400 
    
     D-Loss=0.059109386056661606 G-Loss=0.9217849969863892



![png](output_12_28.png)


    Epoch ID=0 Batch ID=1500 
    
     D-Loss=0.18823960423469543 G-Loss=0.5970481038093567



![png](output_12_30.png)


    Epoch ID=0 Batch ID=1600 
    
     D-Loss=0.07585460692644119 G-Loss=0.8025795817375183



![png](output_12_32.png)


    Epoch ID=0 Batch ID=1700 
    
     D-Loss=0.06141771376132965 G-Loss=0.7037197947502136



![png](output_12_34.png)


    Epoch ID=0 Batch ID=1800 
    
     D-Loss=0.29965734481811523 G-Loss=1.2362244129180908



![png](output_12_36.png)


    Epoch ID=1 Batch ID=0 
    
     D-Loss=0.14948074519634247 G-Loss=0.6934993863105774



![png](output_12_38.png)


    Epoch ID=1 Batch ID=100 
    
     D-Loss=0.09207949042320251 G-Loss=1.0403361320495605



![png](output_12_40.png)


    Epoch ID=1 Batch ID=200 
    
     D-Loss=0.19846245646476746 G-Loss=1.0208704471588135



![png](output_12_42.png)


    Epoch ID=1 Batch ID=300 
    
     D-Loss=0.1353323608636856 G-Loss=0.7848243713378906



![png](output_12_44.png)


    Epoch ID=1 Batch ID=400 
    
     D-Loss=0.14947620034217834 G-Loss=0.9446578025817871



![png](output_12_46.png)


    Epoch ID=1 Batch ID=500 
    
     D-Loss=0.11825098097324371 G-Loss=0.4883885979652405



![png](output_12_48.png)


    Epoch ID=1 Batch ID=600 
    
     D-Loss=0.19390927255153656 G-Loss=0.5661721229553223



![png](output_12_50.png)


    Epoch ID=1 Batch ID=700 
    
     D-Loss=0.4641159176826477 G-Loss=0.43232014775276184



![png](output_12_52.png)


    Epoch ID=1 Batch ID=800 
    
     D-Loss=0.32661551237106323 G-Loss=0.5200859308242798



![png](output_12_54.png)


    Epoch ID=1 Batch ID=900 
    
     D-Loss=0.35382080078125 G-Loss=0.2726805806159973



![png](output_12_56.png)


    Epoch ID=1 Batch ID=1000 
    
     D-Loss=0.3024030327796936 G-Loss=0.4103527367115021



![png](output_12_58.png)


    Epoch ID=1 Batch ID=1100 
    
     D-Loss=0.28779706358909607 G-Loss=0.5770471692085266



![png](output_12_60.png)


    Epoch ID=1 Batch ID=1200 
    
     D-Loss=0.24063196778297424 G-Loss=0.6318605542182922



![png](output_12_62.png)


    Epoch ID=1 Batch ID=1300 
    
     D-Loss=0.34834912419319153 G-Loss=0.2897428572177887



![png](output_12_64.png)


    Epoch ID=1 Batch ID=1400 
    
     D-Loss=0.21613949537277222 G-Loss=0.38999736309051514



![png](output_12_66.png)


    Epoch ID=1 Batch ID=1500 
    
     D-Loss=0.1689126044511795 G-Loss=0.920936107635498



![png](output_12_68.png)


    Epoch ID=1 Batch ID=1600 
    
     D-Loss=0.08762986958026886 G-Loss=0.514633297920227



![png](output_12_70.png)


    Epoch ID=1 Batch ID=1700 
    
     D-Loss=0.3318466544151306 G-Loss=0.46000903844833374



![png](output_12_72.png)


    Epoch ID=1 Batch ID=1800 
    
     D-Loss=0.16000398993492126 G-Loss=0.5974721908569336



![png](output_12_74.png)


    Epoch ID=2 Batch ID=0 
    
     D-Loss=0.4008271098136902 G-Loss=1.4240235090255737



![png](output_12_76.png)


    Epoch ID=2 Batch ID=100 
    
     D-Loss=0.19307160377502441 G-Loss=0.7800183296203613



![png](output_12_78.png)


    Epoch ID=2 Batch ID=200 
    
     D-Loss=1.5302236080169678 G-Loss=0.12722399830818176



![png](output_12_80.png)


    Epoch ID=2 Batch ID=300 
    
     D-Loss=0.3374469578266144 G-Loss=1.1945006847381592



![png](output_12_82.png)


    Epoch ID=2 Batch ID=400 
    
     D-Loss=0.15802739560604095 G-Loss=0.7431222200393677



![png](output_12_84.png)


    Epoch ID=2 Batch ID=500 
    
     D-Loss=0.17009487748146057 G-Loss=0.3035312294960022



![png](output_12_86.png)


    Epoch ID=2 Batch ID=600 
    
     D-Loss=0.18720513582229614 G-Loss=0.9995764493942261



![png](output_12_88.png)


    Epoch ID=2 Batch ID=700 
    
     D-Loss=0.32225435972213745 G-Loss=0.41888177394866943



![png](output_12_90.png)


    Epoch ID=2 Batch ID=800 
    
     D-Loss=0.19239817559719086 G-Loss=0.5756144523620605



![png](output_12_92.png)


    Epoch ID=2 Batch ID=900 
    
     D-Loss=0.18668469786643982 G-Loss=0.45021310448646545



![png](output_12_94.png)


    Epoch ID=2 Batch ID=1000 
    
     D-Loss=0.05977490544319153 G-Loss=1.0354499816894531



![png](output_12_96.png)


    Epoch ID=2 Batch ID=1100 
    
     D-Loss=0.21890032291412354 G-Loss=0.9511539936065674



![png](output_12_98.png)


    Epoch ID=2 Batch ID=1200 
    
     D-Loss=0.10253173112869263 G-Loss=0.8129938244819641



![png](output_12_100.png)


    Epoch ID=2 Batch ID=1300 
    
     D-Loss=0.09679090976715088 G-Loss=0.449867844581604



![png](output_12_102.png)


    Epoch ID=2 Batch ID=1400 
    
     D-Loss=0.18788622319698334 G-Loss=0.7797309160232544



![png](output_12_104.png)


    Epoch ID=2 Batch ID=1500 
    
     D-Loss=0.14497576653957367 G-Loss=0.7317827343940735



![png](output_12_106.png)


    Epoch ID=2 Batch ID=1600 
    
     D-Loss=0.07747182250022888 G-Loss=0.9700220823287964



![png](output_12_108.png)


    Epoch ID=2 Batch ID=1700 
    
     D-Loss=0.1422778069972992 G-Loss=0.6064671277999878



![png](output_12_110.png)


    Epoch ID=2 Batch ID=1800 
    
     D-Loss=0.3122670650482178 G-Loss=0.7185271978378296



![png](output_12_112.png)


    Epoch ID=3 Batch ID=0 
    
     D-Loss=0.3251939117908478 G-Loss=0.22684764862060547



![png](output_12_114.png)


    Epoch ID=3 Batch ID=100 
    
     D-Loss=0.1684044897556305 G-Loss=0.8138859272003174



![png](output_12_116.png)


    Epoch ID=3 Batch ID=200 
    
     D-Loss=0.12778352200984955 G-Loss=0.7529358863830566



![png](output_12_118.png)


    Epoch ID=3 Batch ID=300 
    
     D-Loss=0.27071255445480347 G-Loss=0.16342608630657196



![png](output_12_120.png)


    Epoch ID=3 Batch ID=400 
    
     D-Loss=0.21267016232013702 G-Loss=0.9033311009407043



![png](output_12_122.png)


    Epoch ID=3 Batch ID=500 
    
     D-Loss=0.21516525745391846 G-Loss=1.0814449787139893



![png](output_12_124.png)


    Epoch ID=3 Batch ID=600 
    
     D-Loss=0.18193352222442627 G-Loss=0.6492577791213989



![png](output_12_126.png)


    Epoch ID=3 Batch ID=700 
    
     D-Loss=0.1775263547897339 G-Loss=0.8630233407020569



![png](output_12_128.png)


    Epoch ID=3 Batch ID=800 
    
     D-Loss=0.5107330083847046 G-Loss=1.27354097366333



![png](output_12_130.png)


    Epoch ID=3 Batch ID=900 
    
     D-Loss=0.1570863276720047 G-Loss=0.6009745001792908



![png](output_12_132.png)


    Epoch ID=3 Batch ID=1000 
    
     D-Loss=0.3751717805862427 G-Loss=0.7341494560241699



![png](output_12_134.png)


    Epoch ID=3 Batch ID=1100 
    
     D-Loss=0.09024937450885773 G-Loss=0.43426060676574707



![png](output_12_136.png)


    Epoch ID=3 Batch ID=1200 
    
     D-Loss=0.07728732377290726 G-Loss=1.0359286069869995



![png](output_12_138.png)


    Epoch ID=3 Batch ID=1300 
    
     D-Loss=0.10974854975938797 G-Loss=0.5014258027076721



![png](output_12_140.png)


    Epoch ID=3 Batch ID=1400 
    
     D-Loss=0.11448624730110168 G-Loss=1.3021742105484009



![png](output_12_142.png)


    Epoch ID=3 Batch ID=1500 
    
     D-Loss=0.04680080711841583 G-Loss=0.9061709642410278



![png](output_12_144.png)


    Epoch ID=3 Batch ID=1600 
    
     D-Loss=0.09570781886577606 G-Loss=0.5970727801322937



![png](output_12_146.png)


    Epoch ID=3 Batch ID=1700 
    
     D-Loss=0.1585269719362259 G-Loss=1.6924397945404053



![png](output_12_148.png)


    Epoch ID=3 Batch ID=1800 
    
     D-Loss=0.1341010332107544 G-Loss=0.5747125148773193



![png](output_12_150.png)


    Epoch ID=4 Batch ID=0 
    
     D-Loss=0.08065986633300781 G-Loss=0.7642529010772705



![png](output_12_152.png)


    Epoch ID=4 Batch ID=100 
    
     D-Loss=0.14029547572135925 G-Loss=1.221071481704712



![png](output_12_154.png)


    Epoch ID=4 Batch ID=200 
    
     D-Loss=0.28818610310554504 G-Loss=0.9463022947311401



![png](output_12_156.png)


    Epoch ID=4 Batch ID=300 
    
     D-Loss=0.08052106946706772 G-Loss=0.54066401720047



![png](output_12_158.png)


    Epoch ID=4 Batch ID=400 
    
     D-Loss=0.16761843860149384 G-Loss=0.5517034530639648



![png](output_12_160.png)


    Epoch ID=4 Batch ID=500 
    
     D-Loss=0.18631701171398163 G-Loss=0.7319110631942749



![png](output_12_162.png)


    Epoch ID=4 Batch ID=600 
    
     D-Loss=0.10708343982696533 G-Loss=0.6096556782722473



![png](output_12_164.png)


    Epoch ID=4 Batch ID=700 
    
     D-Loss=0.2511407136917114 G-Loss=0.9017599821090698



![png](output_12_166.png)


    Epoch ID=4 Batch ID=800 
    
     D-Loss=0.08276492357254028 G-Loss=0.6764005422592163



![png](output_12_168.png)


    Epoch ID=4 Batch ID=900 
    
     D-Loss=0.09604914486408234 G-Loss=0.6661819219589233



![png](output_12_170.png)


    Epoch ID=4 Batch ID=1000 
    
     D-Loss=0.19617091119289398 G-Loss=0.5737773180007935



![png](output_12_172.png)


    Epoch ID=4 Batch ID=1100 
    
     D-Loss=0.16229304671287537 G-Loss=1.0229880809783936



![png](output_12_174.png)


    Epoch ID=4 Batch ID=1200 
    
     D-Loss=0.22940106689929962 G-Loss=1.4058852195739746



![png](output_12_176.png)


    Epoch ID=4 Batch ID=1300 
    
     D-Loss=0.15065108239650726 G-Loss=0.9587112665176392



![png](output_12_178.png)


    Epoch ID=4 Batch ID=1400 
    
     D-Loss=0.08075766265392303 G-Loss=1.2977062463760376



![png](output_12_180.png)


    Epoch ID=4 Batch ID=1500 
    
     D-Loss=0.04791607707738876 G-Loss=0.8896881341934204



![png](output_12_182.png)


    Epoch ID=4 Batch ID=1600 
    
     D-Loss=0.12828882038593292 G-Loss=0.762262225151062



![png](output_12_184.png)


    Epoch ID=4 Batch ID=1700 
    
     D-Loss=0.09904347360134125 G-Loss=1.193873643875122



![png](output_12_186.png)


    Epoch ID=4 Batch ID=1800 
    
     D-Loss=0.07540962100028992 G-Loss=0.813156008720398



![png](output_12_188.png)


    Epoch ID=5 Batch ID=0 
    
     D-Loss=0.07111786305904388 G-Loss=0.6032891273498535



![png](output_12_190.png)


    Epoch ID=5 Batch ID=100 
    
     D-Loss=0.06784143298864365 G-Loss=0.9104608297348022



![png](output_12_192.png)


    Epoch ID=5 Batch ID=200 
    
     D-Loss=0.0946667492389679 G-Loss=0.7436808347702026



![png](output_12_194.png)


    Epoch ID=5 Batch ID=300 
    
     D-Loss=0.1362549066543579 G-Loss=1.294539213180542



![png](output_12_196.png)


    Epoch ID=5 Batch ID=400 
    
     D-Loss=0.039412107318639755 G-Loss=1.1025499105453491



![png](output_12_198.png)


    Epoch ID=5 Batch ID=500 
    
     D-Loss=0.19508609175682068 G-Loss=0.7893275618553162



![png](output_12_200.png)


    Epoch ID=5 Batch ID=600 
    
     D-Loss=0.052869390696287155 G-Loss=1.4012413024902344



![png](output_12_202.png)


    Epoch ID=5 Batch ID=700 
    
     D-Loss=0.08730629086494446 G-Loss=0.7960569858551025



![png](output_12_204.png)


    Epoch ID=5 Batch ID=800 
    
     D-Loss=0.0934702455997467 G-Loss=0.4844265878200531



![png](output_12_206.png)


    Epoch ID=5 Batch ID=900 
    
     D-Loss=0.24576644599437714 G-Loss=0.5713326334953308



![png](output_12_208.png)


    Epoch ID=5 Batch ID=1000 
    
     D-Loss=0.11616192013025284 G-Loss=0.8180938363075256



![png](output_12_210.png)


    Epoch ID=5 Batch ID=1100 
    
     D-Loss=0.06129559129476547 G-Loss=1.238623857498169



![png](output_12_212.png)


    Epoch ID=5 Batch ID=1200 
    
     D-Loss=0.028729423880577087 G-Loss=0.8909373879432678



![png](output_12_214.png)


    Epoch ID=5 Batch ID=1300 
    
     D-Loss=0.03935423493385315 G-Loss=0.8337852954864502



![png](output_12_216.png)


    Epoch ID=5 Batch ID=1400 
    
     D-Loss=0.04269764944911003 G-Loss=1.0522851943969727



![png](output_12_218.png)


    Epoch ID=5 Batch ID=1500 
    
     D-Loss=0.07552240788936615 G-Loss=0.645479679107666



![png](output_12_220.png)


    Epoch ID=5 Batch ID=1600 
    
     D-Loss=0.05552205815911293 G-Loss=1.512845754623413



![png](output_12_222.png)


    Epoch ID=5 Batch ID=1700 
    
     D-Loss=0.04039156436920166 G-Loss=0.9314566850662231



![png](output_12_224.png)


    Epoch ID=5 Batch ID=1800 
    
     D-Loss=0.07221339643001556 G-Loss=1.005277156829834



![png](output_12_226.png)


    Epoch ID=6 Batch ID=0 
    
     D-Loss=0.023051436990499496 G-Loss=0.9035989046096802



![png](output_12_228.png)


    Epoch ID=6 Batch ID=100 
    
     D-Loss=0.04932551085948944 G-Loss=0.9383813738822937



![png](output_12_230.png)


    Epoch ID=6 Batch ID=200 
    
     D-Loss=0.05120714753866196 G-Loss=0.600116491317749



![png](output_12_232.png)


    Epoch ID=6 Batch ID=300 
    
     D-Loss=0.08469412475824356 G-Loss=1.1965411901474



![png](output_12_234.png)


    Epoch ID=6 Batch ID=400 
    
     D-Loss=0.03454137593507767 G-Loss=1.0143136978149414



![png](output_12_236.png)


    Epoch ID=6 Batch ID=500 
    
     D-Loss=0.9490438103675842 G-Loss=0.044926196336746216



![png](output_12_238.png)


    Epoch ID=6 Batch ID=600 
    
     D-Loss=0.06083190068602562 G-Loss=1.1394562721252441



![png](output_12_240.png)


    Epoch ID=6 Batch ID=700 
    
     D-Loss=0.26327240467071533 G-Loss=0.5323363542556763



![png](output_12_242.png)


    Epoch ID=6 Batch ID=800 
    
     D-Loss=0.1049802377820015 G-Loss=0.9928985238075256



![png](output_12_244.png)


    Epoch ID=6 Batch ID=900 
    
     D-Loss=0.06136822700500488 G-Loss=0.7781879305839539



![png](output_12_246.png)


    Epoch ID=6 Batch ID=1000 
    
     D-Loss=0.02565855160355568 G-Loss=1.1241681575775146



![png](output_12_248.png)


    Epoch ID=6 Batch ID=1100 
    
     D-Loss=0.150676429271698 G-Loss=1.252845287322998



![png](output_12_250.png)


    Epoch ID=6 Batch ID=1200 
    
     D-Loss=0.058903247117996216 G-Loss=0.8936097621917725



![png](output_12_252.png)


    Epoch ID=6 Batch ID=1300 
    
     D-Loss=0.11163600534200668 G-Loss=0.8819695711135864



![png](output_12_254.png)


    Epoch ID=6 Batch ID=1400 
    
     D-Loss=0.028707202523946762 G-Loss=0.8194321393966675



![png](output_12_256.png)


    Epoch ID=6 Batch ID=1500 
    
     D-Loss=0.057408981025218964 G-Loss=0.910914421081543



![png](output_12_258.png)


    Epoch ID=6 Batch ID=1600 
    
     D-Loss=0.25063619017601013 G-Loss=0.9224302172660828



![png](output_12_260.png)


    Epoch ID=6 Batch ID=1700 
    
     D-Loss=0.09244143217802048 G-Loss=0.4404316842556



![png](output_12_262.png)


    Epoch ID=6 Batch ID=1800 
    
     D-Loss=0.08348322659730911 G-Loss=0.4018116295337677



![png](output_12_264.png)


    Epoch ID=7 Batch ID=0 
    
     D-Loss=0.06502607464790344 G-Loss=0.7641855478286743



![png](output_12_266.png)


    Epoch ID=7 Batch ID=100 
    
     D-Loss=0.02509377896785736 G-Loss=1.1288483142852783



![png](output_12_268.png)


    Epoch ID=7 Batch ID=200 
    
     D-Loss=0.04155135527253151 G-Loss=1.2387171983718872



![png](output_12_270.png)


    Epoch ID=7 Batch ID=300 
    
     D-Loss=0.039614856243133545 G-Loss=1.0619723796844482



![png](output_12_272.png)


    Epoch ID=7 Batch ID=400 
    
     D-Loss=0.2714537978172302 G-Loss=0.5699805617332458



![png](output_12_274.png)


    Epoch ID=7 Batch ID=500 
    
     D-Loss=0.019326917827129364 G-Loss=1.0033655166625977



![png](output_12_276.png)


    Epoch ID=7 Batch ID=600 
    
     D-Loss=0.13523955643177032 G-Loss=1.2876100540161133



![png](output_12_278.png)


    Epoch ID=7 Batch ID=700 
    
     D-Loss=0.05695457383990288 G-Loss=0.7633079886436462



![png](output_12_280.png)


    Epoch ID=7 Batch ID=800 
    
     D-Loss=0.026625892147421837 G-Loss=0.8661825656890869



![png](output_12_282.png)


    Epoch ID=7 Batch ID=900 
    
     D-Loss=0.023699194192886353 G-Loss=0.8654183149337769



![png](output_12_284.png)


    Epoch ID=7 Batch ID=1000 
    
     D-Loss=0.13736486434936523 G-Loss=0.6476945877075195



![png](output_12_286.png)


    Epoch ID=7 Batch ID=1100 
    
     D-Loss=0.02586495131254196 G-Loss=1.4510107040405273



![png](output_12_288.png)


    Epoch ID=7 Batch ID=1200 
    
     D-Loss=0.060235895216464996 G-Loss=1.0627436637878418



![png](output_12_290.png)


    Epoch ID=7 Batch ID=1300 
    
     D-Loss=0.034548304975032806 G-Loss=1.1105105876922607



![png](output_12_292.png)


    Epoch ID=7 Batch ID=1400 
    
     D-Loss=0.17277486622333527 G-Loss=0.8932023048400879



![png](output_12_294.png)


    Epoch ID=7 Batch ID=1500 
    
     D-Loss=0.04175697639584541 G-Loss=1.358069658279419



![png](output_12_296.png)


    Epoch ID=7 Batch ID=1600 
    
     D-Loss=0.03940146043896675 G-Loss=0.5916641354560852



![png](output_12_298.png)


    Epoch ID=7 Batch ID=1700 
    
     D-Loss=0.11307286471128464 G-Loss=0.9825721979141235



![png](output_12_300.png)


    Epoch ID=7 Batch ID=1800 
    
     D-Loss=0.026517318561673164 G-Loss=1.1014139652252197



![png](output_12_302.png)


    Epoch ID=8 Batch ID=0 
    
     D-Loss=0.05528169870376587 G-Loss=1.0226975679397583



![png](output_12_304.png)


    Epoch ID=8 Batch ID=100 
    
     D-Loss=0.09683781862258911 G-Loss=1.4817187786102295



![png](output_12_306.png)


    Epoch ID=8 Batch ID=200 
    
     D-Loss=0.039385080337524414 G-Loss=1.484697937965393



![png](output_12_308.png)


    Epoch ID=8 Batch ID=300 
    
     D-Loss=0.0405496209859848 G-Loss=0.8884657621383667



![png](output_12_310.png)


    Epoch ID=8 Batch ID=400 
    
     D-Loss=0.020274754613637924 G-Loss=1.0697402954101562



![png](output_12_312.png)


    Epoch ID=8 Batch ID=500 
    
     D-Loss=0.07571655511856079 G-Loss=0.7738865613937378



![png](output_12_314.png)


    Epoch ID=8 Batch ID=600 
    
     D-Loss=0.06420592963695526 G-Loss=1.3238577842712402



![png](output_12_316.png)


    Epoch ID=8 Batch ID=700 
    
     D-Loss=0.05730278044939041 G-Loss=1.0491816997528076



![png](output_12_318.png)


    Epoch ID=8 Batch ID=800 
    
     D-Loss=0.014839617535471916 G-Loss=0.6380182504653931



![png](output_12_320.png)


    Epoch ID=8 Batch ID=900 
    
     D-Loss=0.09166599810123444 G-Loss=0.9901611804962158



![png](output_12_322.png)


    Epoch ID=8 Batch ID=1000 
    
     D-Loss=0.09490203857421875 G-Loss=0.7502714395523071



![png](output_12_324.png)


    Epoch ID=8 Batch ID=1100 
    
     D-Loss=0.02090262435376644 G-Loss=0.9230260848999023



![png](output_12_326.png)


    Epoch ID=8 Batch ID=1200 
    
     D-Loss=0.0736616924405098 G-Loss=1.039686679840088



![png](output_12_328.png)


    Epoch ID=8 Batch ID=1300 
    
     D-Loss=0.03678358718752861 G-Loss=0.9158564209938049



![png](output_12_330.png)


    Epoch ID=8 Batch ID=1400 
    
     D-Loss=0.13977819681167603 G-Loss=0.846415102481842



![png](output_12_332.png)


    Epoch ID=8 Batch ID=1500 
    
     D-Loss=0.06585482507944107 G-Loss=0.9492319226264954



![png](output_12_334.png)


    Epoch ID=8 Batch ID=1600 
    
     D-Loss=0.03129959478974342 G-Loss=0.9231626987457275



![png](output_12_336.png)


    Epoch ID=8 Batch ID=1700 
    
     D-Loss=0.06321538984775543 G-Loss=1.1762652397155762



![png](output_12_338.png)


    Epoch ID=8 Batch ID=1800 
    
     D-Loss=0.05268007516860962 G-Loss=1.033692479133606



![png](output_12_340.png)


    Epoch ID=9 Batch ID=0 
    
     D-Loss=0.03255988284945488 G-Loss=1.277975082397461



![png](output_12_342.png)


    Epoch ID=9 Batch ID=100 
    
     D-Loss=0.04266148805618286 G-Loss=0.773383617401123



![png](output_12_344.png)


    Epoch ID=9 Batch ID=200 
    
     D-Loss=0.12089435011148453 G-Loss=0.7138911485671997



![png](output_12_346.png)


    Epoch ID=9 Batch ID=300 
    
     D-Loss=0.047555893659591675 G-Loss=1.0659096240997314



![png](output_12_348.png)


    Epoch ID=9 Batch ID=400 
    
     D-Loss=0.06500237435102463 G-Loss=0.9656267166137695



![png](output_12_350.png)


    Epoch ID=9 Batch ID=500 
    
     D-Loss=0.030526919290423393 G-Loss=0.9634817838668823



![png](output_12_352.png)


    Epoch ID=9 Batch ID=600 
    
     D-Loss=0.07268200069665909 G-Loss=1.2991876602172852



![png](output_12_354.png)


    Epoch ID=9 Batch ID=700 
    
     D-Loss=0.1784868687391281 G-Loss=0.7933263778686523



![png](output_12_356.png)


    Epoch ID=9 Batch ID=800 
    
     D-Loss=0.012506200931966305 G-Loss=0.8704217076301575



![png](output_12_358.png)


    Epoch ID=9 Batch ID=900 
    
     D-Loss=0.15610362589359283 G-Loss=0.789577841758728



![png](output_12_360.png)


    Epoch ID=9 Batch ID=1000 
    
     D-Loss=0.12328260391950607 G-Loss=0.7022613286972046



![png](output_12_362.png)


    Epoch ID=9 Batch ID=1100 
    
     D-Loss=0.03957776725292206 G-Loss=1.0865304470062256



![png](output_12_364.png)


    Epoch ID=9 Batch ID=1200 
    
     D-Loss=0.014272814616560936 G-Loss=1.1932828426361084



![png](output_12_366.png)


    Epoch ID=9 Batch ID=1300 
    
     D-Loss=0.09236739575862885 G-Loss=1.3127777576446533



![png](output_12_368.png)


    Epoch ID=9 Batch ID=1400 
    
     D-Loss=0.09535202383995056 G-Loss=0.7039675712585449



![png](output_12_370.png)


    Epoch ID=9 Batch ID=1500 
    
     D-Loss=0.04047500342130661 G-Loss=1.013313889503479



![png](output_12_372.png)


    Epoch ID=9 Batch ID=1600 
    
     D-Loss=0.04688654839992523 G-Loss=0.7144192457199097



![png](output_12_374.png)


    Epoch ID=9 Batch ID=1700 
    
     D-Loss=0.05292198434472084 G-Loss=1.3458869457244873



![png](output_12_376.png)


    Epoch ID=9 Batch ID=1800 
    
     D-Loss=0.014810319058597088 G-Loss=1.042931079864502



![png](output_12_378.png)


    Epoch ID=10 Batch ID=0 
    
     D-Loss=0.013736434280872345 G-Loss=1.1734867095947266



![png](output_12_380.png)


    Epoch ID=10 Batch ID=100 
    
     D-Loss=0.09149311482906342 G-Loss=0.9720432758331299



![png](output_12_382.png)


    Epoch ID=10 Batch ID=200 
    
     D-Loss=0.029284454882144928 G-Loss=0.8518458604812622



![png](output_12_384.png)


    Epoch ID=10 Batch ID=300 
    
     D-Loss=0.04419095441699028 G-Loss=0.5280283093452454



![png](output_12_386.png)


    Epoch ID=10 Batch ID=400 
    
     D-Loss=0.030283695086836815 G-Loss=1.0074868202209473



![png](output_12_388.png)


    Epoch ID=10 Batch ID=500 
    
     D-Loss=0.012887491844594479 G-Loss=1.0935996770858765



![png](output_12_390.png)


    Epoch ID=10 Batch ID=600 
    
     D-Loss=0.09736743569374084 G-Loss=0.5753271579742432



![png](output_12_392.png)


    Epoch ID=10 Batch ID=700 
    
     D-Loss=0.04101863503456116 G-Loss=1.1583034992218018



![png](output_12_394.png)


    Epoch ID=10 Batch ID=800 
    
     D-Loss=0.022214341908693314 G-Loss=0.8220953941345215



![png](output_12_396.png)


    Epoch ID=10 Batch ID=900 
    
     D-Loss=0.0298161581158638 G-Loss=0.8397170305252075



![png](output_12_398.png)


    Epoch ID=10 Batch ID=1000 
    
     D-Loss=0.0169465821236372 G-Loss=0.8080704808235168



![png](output_12_400.png)


    Epoch ID=10 Batch ID=1100 
    
     D-Loss=0.06586981564760208 G-Loss=0.890291690826416



![png](output_12_402.png)


    Epoch ID=10 Batch ID=1200 
    
     D-Loss=0.03429468721151352 G-Loss=1.1347477436065674



![png](output_12_404.png)


    Epoch ID=10 Batch ID=1300 
    
     D-Loss=0.15205460786819458 G-Loss=0.4656068682670593



![png](output_12_406.png)


    Epoch ID=10 Batch ID=1400 
    
     D-Loss=0.04593486711382866 G-Loss=1.4135264158248901



![png](output_12_408.png)


    Epoch ID=10 Batch ID=1500 
    
     D-Loss=0.030772563070058823 G-Loss=1.215019941329956



![png](output_12_410.png)


    Epoch ID=10 Batch ID=1600 
    
     D-Loss=0.028221700340509415 G-Loss=0.9588998556137085



![png](output_12_412.png)


    Epoch ID=10 Batch ID=1700 
    
     D-Loss=0.14609044790267944 G-Loss=1.3747750520706177



![png](output_12_414.png)


    Epoch ID=10 Batch ID=1800 
    
     D-Loss=0.05106296017765999 G-Loss=0.8774070143699646



![png](output_12_416.png)


    Epoch ID=11 Batch ID=0 
    
     D-Loss=0.008485984988510609 G-Loss=0.6645403504371643



![png](output_12_418.png)


    Epoch ID=11 Batch ID=100 
    
     D-Loss=0.023848500102758408 G-Loss=0.9010822176933289



![png](output_12_420.png)


    Epoch ID=11 Batch ID=200 
    
     D-Loss=0.10844381898641586 G-Loss=1.1585981845855713



![png](output_12_422.png)


    Epoch ID=11 Batch ID=300 
    
     D-Loss=0.03230839967727661 G-Loss=0.8572943806648254



![png](output_12_424.png)


    Epoch ID=11 Batch ID=400 
    
     D-Loss=0.026261182501912117 G-Loss=0.9442567825317383



![png](output_12_426.png)


    Epoch ID=11 Batch ID=500 
    
     D-Loss=0.019559629261493683 G-Loss=1.0034030675888062



![png](output_12_428.png)


    Epoch ID=11 Batch ID=600 
    
     D-Loss=0.1333739459514618 G-Loss=0.830806314945221



![png](output_12_430.png)


    Epoch ID=11 Batch ID=700 
    
     D-Loss=0.11405257880687714 G-Loss=0.681114912033081



![png](output_12_432.png)


    Epoch ID=11 Batch ID=800 
    
     D-Loss=0.055268336087465286 G-Loss=0.7620543837547302



![png](output_12_434.png)


    Epoch ID=11 Batch ID=900 
    
     D-Loss=0.044203177094459534 G-Loss=0.9317896962165833



![png](output_12_436.png)


    Epoch ID=11 Batch ID=1000 
    
     D-Loss=0.08671022951602936 G-Loss=1.22116219997406



![png](output_12_438.png)


    Epoch ID=11 Batch ID=1100 
    
     D-Loss=0.017287416383624077 G-Loss=0.9373984336853027



![png](output_12_440.png)


    Epoch ID=11 Batch ID=1200 
    
     D-Loss=0.014872762374579906 G-Loss=1.0028871297836304



![png](output_12_442.png)


    Epoch ID=11 Batch ID=1300 
    
     D-Loss=0.04694719985127449 G-Loss=1.1546978950500488



![png](output_12_444.png)


    Epoch ID=11 Batch ID=1400 
    
     D-Loss=0.08510241657495499 G-Loss=0.7892489433288574



![png](output_12_446.png)


    Epoch ID=11 Batch ID=1500 
    
     D-Loss=0.01469055749475956 G-Loss=0.9724133014678955



![png](output_12_448.png)


    Epoch ID=11 Batch ID=1600 
    
     D-Loss=0.02631019614636898 G-Loss=0.8872658014297485



![png](output_12_450.png)


    Epoch ID=11 Batch ID=1700 
    
     D-Loss=0.018791642040014267 G-Loss=1.1208257675170898



![png](output_12_452.png)


    Epoch ID=11 Batch ID=1800 
    
     D-Loss=0.06353272497653961 G-Loss=1.166917324066162



![png](output_12_454.png)


    Epoch ID=12 Batch ID=0 
    
     D-Loss=0.05737417936325073 G-Loss=1.0923876762390137



![png](output_12_456.png)


    Epoch ID=12 Batch ID=100 
    
     D-Loss=0.04356764256954193 G-Loss=0.677922785282135



![png](output_12_458.png)


    Epoch ID=12 Batch ID=200 
    
     D-Loss=0.09979402273893356 G-Loss=0.7346960306167603



![png](output_12_460.png)


    Epoch ID=12 Batch ID=300 
    
     D-Loss=0.00731640262529254 G-Loss=0.9640225172042847



![png](output_12_462.png)


    Epoch ID=12 Batch ID=400 
    
     D-Loss=0.02008197456598282 G-Loss=0.7820202112197876



![png](output_12_464.png)


    Epoch ID=12 Batch ID=500 
    
     D-Loss=0.050277478992938995 G-Loss=0.9416468739509583



![png](output_12_466.png)


    Epoch ID=12 Batch ID=600 
    
     D-Loss=0.027854129672050476 G-Loss=0.9149248600006104



![png](output_12_468.png)


    Epoch ID=12 Batch ID=700 
    
     D-Loss=0.031850822269916534 G-Loss=1.2152236700057983



![png](output_12_470.png)


    Epoch ID=12 Batch ID=800 
    
     D-Loss=0.03547179326415062 G-Loss=1.3103255033493042



![png](output_12_472.png)


    Epoch ID=12 Batch ID=900 
    
     D-Loss=0.06504305452108383 G-Loss=0.9864102602005005



![png](output_12_474.png)


    Epoch ID=12 Batch ID=1000 
    
     D-Loss=0.04375023394823074 G-Loss=0.9804983735084534



![png](output_12_476.png)


    Epoch ID=12 Batch ID=1100 
    
     D-Loss=0.024055790156126022 G-Loss=0.9786320924758911



![png](output_12_478.png)


    Epoch ID=12 Batch ID=1200 
    
     D-Loss=0.0386868417263031 G-Loss=1.1964354515075684



![png](output_12_480.png)


    Epoch ID=12 Batch ID=1300 
    
     D-Loss=0.02395559288561344 G-Loss=0.8887919187545776



![png](output_12_482.png)


    Epoch ID=12 Batch ID=1400 
    
     D-Loss=0.02258831076323986 G-Loss=0.8444098234176636



![png](output_12_484.png)


    Epoch ID=12 Batch ID=1500 
    
     D-Loss=0.013297803699970245 G-Loss=0.9802988767623901



![png](output_12_486.png)


    Epoch ID=12 Batch ID=1600 
    
     D-Loss=0.010012472048401833 G-Loss=1.172366976737976



![png](output_12_488.png)


    Epoch ID=12 Batch ID=1700 
    
     D-Loss=0.0812346413731575 G-Loss=0.6482234597206116



![png](output_12_490.png)


    Epoch ID=12 Batch ID=1800 
    
     D-Loss=0.02551959827542305 G-Loss=0.7600587010383606



![png](output_12_492.png)


    Epoch ID=13 Batch ID=0 
    
     D-Loss=0.021334370598196983 G-Loss=1.3096060752868652



![png](output_12_494.png)


    Epoch ID=13 Batch ID=100 
    
     D-Loss=0.0058733997866511345 G-Loss=0.9685430526733398



![png](output_12_496.png)


    Epoch ID=13 Batch ID=200 
    
     D-Loss=0.015706021338701248 G-Loss=0.8786799907684326



![png](output_12_498.png)


    Epoch ID=13 Batch ID=300 
    
     D-Loss=0.009392207488417625 G-Loss=1.080885410308838



![png](output_12_500.png)


    Epoch ID=13 Batch ID=400 
    
     D-Loss=0.05245854705572128 G-Loss=0.7606619596481323



![png](output_12_502.png)


    Epoch ID=13 Batch ID=500 
    
     D-Loss=0.014419103041291237 G-Loss=0.8637226819992065



![png](output_12_504.png)


    Epoch ID=13 Batch ID=600 
    
     D-Loss=0.04640379548072815 G-Loss=1.126063346862793



![png](output_12_506.png)


    Epoch ID=13 Batch ID=700 
    
     D-Loss=0.016927724704146385 G-Loss=0.7829092144966125



![png](output_12_508.png)


    Epoch ID=13 Batch ID=800 
    
     D-Loss=0.03855689615011215 G-Loss=1.0829648971557617



![png](output_12_510.png)


    Epoch ID=13 Batch ID=900 
    
     D-Loss=0.009615739807486534 G-Loss=0.9894837141036987



![png](output_12_512.png)


    Epoch ID=13 Batch ID=1000 
    
     D-Loss=0.03506877273321152 G-Loss=0.7578850984573364



![png](output_12_514.png)


    Epoch ID=13 Batch ID=1100 
    
     D-Loss=0.17952242493629456 G-Loss=0.5755703449249268



![png](output_12_516.png)


    Epoch ID=13 Batch ID=1200 
    
     D-Loss=0.050386376678943634 G-Loss=0.8706139326095581



![png](output_12_518.png)


    Epoch ID=13 Batch ID=1300 
    
     D-Loss=0.021014612168073654 G-Loss=1.0625035762786865



![png](output_12_520.png)


    Epoch ID=13 Batch ID=1400 
    
     D-Loss=0.010086712427437305 G-Loss=1.259331226348877



![png](output_12_522.png)


    Epoch ID=13 Batch ID=1500 
    
     D-Loss=0.006751023232936859 G-Loss=1.1902475357055664



![png](output_12_524.png)


    Epoch ID=13 Batch ID=1600 
    
     D-Loss=0.011212462559342384 G-Loss=0.834634006023407



![png](output_12_526.png)


    Epoch ID=13 Batch ID=1700 
    
     D-Loss=0.012468907982110977 G-Loss=0.8682239651679993



![png](output_12_528.png)


    Epoch ID=13 Batch ID=1800 
    
     D-Loss=0.01437994185835123 G-Loss=1.1317532062530518



![png](output_12_530.png)


    Epoch ID=14 Batch ID=0 
    
     D-Loss=0.13489755988121033 G-Loss=0.640828013420105



![png](output_12_532.png)


    Epoch ID=14 Batch ID=100 
    
     D-Loss=0.04921326786279678 G-Loss=1.1263641119003296



![png](output_12_534.png)


    Epoch ID=14 Batch ID=200 
    
     D-Loss=0.03866671770811081 G-Loss=0.722066342830658



![png](output_12_536.png)


    Epoch ID=14 Batch ID=300 
    
     D-Loss=0.018411453813314438 G-Loss=0.9039878845214844



![png](output_12_538.png)


    Epoch ID=14 Batch ID=400 
    
     D-Loss=0.04890851676464081 G-Loss=1.1053879261016846



![png](output_12_540.png)


    Epoch ID=14 Batch ID=500 
    
     D-Loss=0.015291792340576649 G-Loss=1.0680078268051147



![png](output_12_542.png)


    Epoch ID=14 Batch ID=600 
    
     D-Loss=0.009760105982422829 G-Loss=1.244476079940796



![png](output_12_544.png)


    Epoch ID=14 Batch ID=700 
    
     D-Loss=0.11625095456838608 G-Loss=1.0039366483688354



![png](output_12_546.png)


    Epoch ID=14 Batch ID=800 
    
     D-Loss=0.05591531842947006 G-Loss=0.9867346286773682



![png](output_12_548.png)


    Epoch ID=14 Batch ID=900 
    
     D-Loss=0.06189713627099991 G-Loss=1.4777164459228516



![png](output_12_550.png)


    Epoch ID=14 Batch ID=1000 
    
     D-Loss=0.0306638702750206 G-Loss=1.1347860097885132



![png](output_12_552.png)


    Epoch ID=14 Batch ID=1100 
    
     D-Loss=0.015138054266571999 G-Loss=1.1350228786468506



![png](output_12_554.png)


    Epoch ID=14 Batch ID=1200 
    
     D-Loss=0.03573378175497055 G-Loss=1.111714482307434



![png](output_12_556.png)


    Epoch ID=14 Batch ID=1300 
    
     D-Loss=0.32203811407089233 G-Loss=0.5850235223770142



![png](output_12_558.png)


    Epoch ID=14 Batch ID=1400 
    
     D-Loss=0.016808606684207916 G-Loss=1.0518829822540283



![png](output_12_560.png)


    Epoch ID=14 Batch ID=1500 
    
     D-Loss=0.01841554045677185 G-Loss=1.264991044998169



![png](output_12_562.png)


    Epoch ID=14 Batch ID=1600 
    
     D-Loss=0.019394373521208763 G-Loss=1.0465118885040283



![png](output_12_564.png)


    Epoch ID=14 Batch ID=1700 
    
     D-Loss=0.011514825746417046 G-Loss=0.8751041293144226



![png](output_12_566.png)


    Epoch ID=14 Batch ID=1800 
    
     D-Loss=0.00979947205632925 G-Loss=0.793493390083313



![png](output_12_568.png)


    Epoch ID=15 Batch ID=0 
    
     D-Loss=0.03710101544857025 G-Loss=1.1231540441513062



![png](output_12_570.png)


    Epoch ID=15 Batch ID=100 
    
     D-Loss=0.0158816147595644 G-Loss=0.9689079523086548



![png](output_12_572.png)


    Epoch ID=15 Batch ID=200 
    
     D-Loss=0.00820118747651577 G-Loss=0.709984302520752



![png](output_12_574.png)


    Epoch ID=15 Batch ID=300 
    
     D-Loss=0.3711306154727936 G-Loss=0.11997513473033905



![png](output_12_576.png)


    Epoch ID=15 Batch ID=400 
    
     D-Loss=0.03582117706537247 G-Loss=0.8844955563545227



![png](output_12_578.png)


    Epoch ID=15 Batch ID=500 
    
     D-Loss=0.020968865603208542 G-Loss=0.9413135051727295



![png](output_12_580.png)


    Epoch ID=15 Batch ID=600 
    
     D-Loss=0.05489595979452133 G-Loss=1.2058885097503662



![png](output_12_582.png)


    Epoch ID=15 Batch ID=700 
    
     D-Loss=0.01904579997062683 G-Loss=1.0416147708892822



![png](output_12_584.png)


    Epoch ID=15 Batch ID=800 
    
     D-Loss=0.015480084344744682 G-Loss=1.0226290225982666



![png](output_12_586.png)


    Epoch ID=15 Batch ID=900 
    
     D-Loss=0.03582639992237091 G-Loss=1.1369235515594482



![png](output_12_588.png)


    Epoch ID=15 Batch ID=1000 
    
     D-Loss=0.0319383405148983 G-Loss=1.1117968559265137



![png](output_12_590.png)


    Epoch ID=15 Batch ID=1100 
    
     D-Loss=0.027321739122271538 G-Loss=0.6875720620155334



![png](output_12_592.png)


    Epoch ID=15 Batch ID=1200 
    
     D-Loss=0.026702895760536194 G-Loss=1.1100668907165527



![png](output_12_594.png)


    Epoch ID=15 Batch ID=1300 
    
     D-Loss=0.02179907262325287 G-Loss=1.2371208667755127



![png](output_12_596.png)


    Epoch ID=15 Batch ID=1400 
    
     D-Loss=0.030951514840126038 G-Loss=1.0998725891113281



![png](output_12_598.png)


    Epoch ID=15 Batch ID=1500 
    
     D-Loss=0.07565324008464813 G-Loss=1.2171316146850586



![png](output_12_600.png)


    Epoch ID=15 Batch ID=1600 
    
     D-Loss=0.30191606283187866 G-Loss=0.21065306663513184



![png](output_12_602.png)


    Epoch ID=15 Batch ID=1700 
    
     D-Loss=0.021762453019618988 G-Loss=0.8384331464767456



![png](output_12_604.png)


    Epoch ID=15 Batch ID=1800 
    
     D-Loss=0.012137710116803646 G-Loss=1.0413389205932617



![png](output_12_606.png)


    Epoch ID=16 Batch ID=0 
    
     D-Loss=0.08940083533525467 G-Loss=1.246382713317871



![png](output_12_608.png)


    Epoch ID=16 Batch ID=100 
    
     D-Loss=0.00595524488016963 G-Loss=1.0791873931884766



![png](output_12_610.png)


    Epoch ID=16 Batch ID=200 
    
     D-Loss=0.04576326906681061 G-Loss=0.8725364804267883



![png](output_12_612.png)


    Epoch ID=16 Batch ID=300 
    
     D-Loss=0.024855967611074448 G-Loss=1.4568026065826416



![png](output_12_614.png)


    Epoch ID=16 Batch ID=400 
    
     D-Loss=0.013464889489114285 G-Loss=1.0847828388214111



![png](output_12_616.png)


    Epoch ID=16 Batch ID=500 
    
     D-Loss=0.025286495685577393 G-Loss=0.9237557649612427



![png](output_12_618.png)


    Epoch ID=16 Batch ID=600 
    
     D-Loss=0.04706696420907974 G-Loss=1.149672508239746



![png](output_12_620.png)


    Epoch ID=16 Batch ID=700 
    
     D-Loss=0.022225532680749893 G-Loss=1.539246916770935



![png](output_12_622.png)


    Epoch ID=16 Batch ID=800 
    
     D-Loss=0.06284595280885696 G-Loss=1.0552350282669067



![png](output_12_624.png)


    Epoch ID=16 Batch ID=900 
    
     D-Loss=0.008757307194173336 G-Loss=1.1260571479797363



![png](output_12_626.png)


    Epoch ID=16 Batch ID=1000 
    
     D-Loss=0.13475219905376434 G-Loss=1.2498724460601807



![png](output_12_628.png)


    Epoch ID=16 Batch ID=1100 
    
     D-Loss=0.09791979193687439 G-Loss=1.2777860164642334



![png](output_12_630.png)


    Epoch ID=16 Batch ID=1200 
    
     D-Loss=0.006418653763830662 G-Loss=1.0181670188903809



![png](output_12_632.png)


    Epoch ID=16 Batch ID=1300 
    
     D-Loss=0.008762186393141747 G-Loss=1.119043231010437



![png](output_12_634.png)


    Epoch ID=16 Batch ID=1400 
    
     D-Loss=0.1391017735004425 G-Loss=1.0624394416809082



![png](output_12_636.png)


    Epoch ID=16 Batch ID=1500 
    
     D-Loss=0.02144749090075493 G-Loss=0.8286222219467163



![png](output_12_638.png)


    Epoch ID=16 Batch ID=1600 
    
     D-Loss=0.02630341798067093 G-Loss=1.1668835878372192



![png](output_12_640.png)


    Epoch ID=16 Batch ID=1700 
    
     D-Loss=0.017184089869260788 G-Loss=1.066572666168213



![png](output_12_642.png)


    Epoch ID=16 Batch ID=1800 
    
     D-Loss=0.06852399557828903 G-Loss=1.0586047172546387



![png](output_12_644.png)


    Epoch ID=17 Batch ID=0 
    
     D-Loss=0.01469020638614893 G-Loss=1.3009811639785767



![png](output_12_646.png)


    Epoch ID=17 Batch ID=100 
    
     D-Loss=0.03129599988460541 G-Loss=0.6971345543861389



![png](output_12_648.png)


    Epoch ID=17 Batch ID=200 
    
     D-Loss=0.015110412612557411 G-Loss=0.9680709838867188



![png](output_12_650.png)


    Epoch ID=17 Batch ID=300 
    
     D-Loss=0.017164025455713272 G-Loss=0.8187944889068604



![png](output_12_652.png)


    Epoch ID=17 Batch ID=400 
    
     D-Loss=0.017464037984609604 G-Loss=1.14751398563385



![png](output_12_654.png)


    Epoch ID=17 Batch ID=500 
    
     D-Loss=0.008688803762197495 G-Loss=1.1322588920593262



![png](output_12_656.png)


    Epoch ID=17 Batch ID=600 
    
     D-Loss=0.021506739780306816 G-Loss=0.8890673518180847



![png](output_12_658.png)


    Epoch ID=17 Batch ID=700 
    
     D-Loss=0.009543513879179955 G-Loss=0.7415221929550171



![png](output_12_660.png)


    Epoch ID=17 Batch ID=800 
    
     D-Loss=0.04036320000886917 G-Loss=0.8845856785774231



![png](output_12_662.png)


    Epoch ID=17 Batch ID=900 
    
     D-Loss=0.03510428965091705 G-Loss=1.589906096458435



![png](output_12_664.png)


    Epoch ID=17 Batch ID=1000 
    
     D-Loss=0.8983469605445862 G-Loss=0.2306852489709854



![png](output_12_666.png)


    Epoch ID=17 Batch ID=1100 
    
     D-Loss=0.049820296466350555 G-Loss=1.1257182359695435



![png](output_12_668.png)


    Epoch ID=17 Batch ID=1200 
    
     D-Loss=0.21566073596477509 G-Loss=1.4497628211975098



![png](output_12_670.png)


    Epoch ID=17 Batch ID=1300 
    
     D-Loss=0.03303813934326172 G-Loss=1.1585707664489746



![png](output_12_672.png)


    Epoch ID=17 Batch ID=1400 
    
     D-Loss=0.009288710542023182 G-Loss=0.9799796342849731



![png](output_12_674.png)


    Epoch ID=17 Batch ID=1500 
    
     D-Loss=0.012703772634267807 G-Loss=0.8854556679725647



![png](output_12_676.png)


    Epoch ID=17 Batch ID=1600 
    
     D-Loss=0.010062536224722862 G-Loss=0.9225185513496399



![png](output_12_678.png)


    Epoch ID=17 Batch ID=1700 
    
     D-Loss=0.019230835139751434 G-Loss=0.9723936915397644



![png](output_12_680.png)


    Epoch ID=17 Batch ID=1800 
    
     D-Loss=0.021573029458522797 G-Loss=1.0475833415985107



![png](output_12_682.png)


    Epoch ID=18 Batch ID=0 
    
     D-Loss=0.00831496063619852 G-Loss=1.2016165256500244



![png](output_12_684.png)


    Epoch ID=18 Batch ID=100 
    
     D-Loss=0.122499980032444 G-Loss=1.2958002090454102



![png](output_12_686.png)


    Epoch ID=18 Batch ID=200 
    
     D-Loss=0.017702534794807434 G-Loss=0.9980553388595581



![png](output_12_688.png)


    Epoch ID=18 Batch ID=300 
    
     D-Loss=0.01579098589718342 G-Loss=1.2356436252593994



![png](output_12_690.png)


    Epoch ID=18 Batch ID=400 
    
     D-Loss=0.05345870554447174 G-Loss=0.7738733291625977



![png](output_12_692.png)


    Epoch ID=18 Batch ID=500 
    
     D-Loss=0.03131207823753357 G-Loss=1.2653734683990479



![png](output_12_694.png)


    Epoch ID=18 Batch ID=600 
    
     D-Loss=0.005701130721718073 G-Loss=1.1378059387207031



![png](output_12_696.png)


    Epoch ID=18 Batch ID=700 
    
     D-Loss=0.03607829287648201 G-Loss=0.7844630479812622



![png](output_12_698.png)


    Epoch ID=18 Batch ID=800 
    
     D-Loss=0.022687029093503952 G-Loss=1.1807827949523926



![png](output_12_700.png)


    Epoch ID=18 Batch ID=900 
    
     D-Loss=0.03580748289823532 G-Loss=1.2078893184661865



![png](output_12_702.png)


    Epoch ID=18 Batch ID=1000 
    
     D-Loss=0.009585171937942505 G-Loss=1.2164472341537476



![png](output_12_704.png)


    Epoch ID=18 Batch ID=1100 
    
     D-Loss=0.14483755826950073 G-Loss=0.690606951713562



![png](output_12_706.png)


    Epoch ID=18 Batch ID=1200 
    
     D-Loss=0.011224604211747646 G-Loss=1.1804523468017578



![png](output_12_708.png)


    Epoch ID=18 Batch ID=1300 
    
     D-Loss=0.021867044270038605 G-Loss=0.6840731501579285



![png](output_12_710.png)


    Epoch ID=18 Batch ID=1400 
    
     D-Loss=0.054914407432079315 G-Loss=2.029837131500244



![png](output_12_712.png)


    Epoch ID=18 Batch ID=1500 
    
     D-Loss=0.018098164349794388 G-Loss=0.8608964085578918



![png](output_12_714.png)


    Epoch ID=18 Batch ID=1600 
    
     D-Loss=0.023989910259842873 G-Loss=1.0636391639709473



![png](output_12_716.png)


    Epoch ID=18 Batch ID=1700 
    
     D-Loss=0.016940651461482048 G-Loss=1.305191993713379



![png](output_12_718.png)


    Epoch ID=18 Batch ID=1800 
    
     D-Loss=0.03707289695739746 G-Loss=0.9396777153015137



![png](output_12_720.png)


    Epoch ID=19 Batch ID=0 
    
     D-Loss=0.06535699963569641 G-Loss=0.9958704710006714



![png](output_12_722.png)


    Epoch ID=19 Batch ID=100 
    
     D-Loss=0.14628159999847412 G-Loss=1.2112295627593994



![png](output_12_724.png)


    Epoch ID=19 Batch ID=200 
    
     D-Loss=0.014682949520647526 G-Loss=0.6637448072433472



![png](output_12_726.png)


    Epoch ID=19 Batch ID=300 
    
     D-Loss=0.02715861052274704 G-Loss=0.8515108823776245



![png](output_12_728.png)


    Epoch ID=19 Batch ID=400 
    
     D-Loss=0.006648050621151924 G-Loss=1.0534456968307495



![png](output_12_730.png)


    Epoch ID=19 Batch ID=500 
    
     D-Loss=0.012569384649395943 G-Loss=0.9365555644035339



![png](output_12_732.png)


    Epoch ID=19 Batch ID=600 
    
     D-Loss=0.06364822387695312 G-Loss=0.8991942405700684



![png](output_12_734.png)


    Epoch ID=19 Batch ID=700 
    
     D-Loss=0.029317526146769524 G-Loss=1.401364803314209



![png](output_12_736.png)


    Epoch ID=19 Batch ID=800 
    
     D-Loss=0.0174681656062603 G-Loss=0.7191983461380005



![png](output_12_738.png)


    Epoch ID=19 Batch ID=900 
    
     D-Loss=0.0077944565564394 G-Loss=0.8570947647094727



![png](output_12_740.png)


    Epoch ID=19 Batch ID=1000 
    
     D-Loss=0.07618646323680878 G-Loss=1.0524706840515137



![png](output_12_742.png)


    Epoch ID=19 Batch ID=1100 
    
     D-Loss=0.015550179407000542 G-Loss=0.7084706425666809



![png](output_12_744.png)


    Epoch ID=19 Batch ID=1200 
    
     D-Loss=0.008733240887522697 G-Loss=0.9141517877578735



![png](output_12_746.png)


    Epoch ID=19 Batch ID=1300 
    
     D-Loss=0.03501037508249283 G-Loss=1.1494534015655518



![png](output_12_748.png)


    Epoch ID=19 Batch ID=1400 
    
     D-Loss=0.009255552664399147 G-Loss=1.0666455030441284



![png](output_12_750.png)


    Epoch ID=19 Batch ID=1500 
    
     D-Loss=0.029576070606708527 G-Loss=0.9208664894104004



![png](output_12_752.png)


    Epoch ID=19 Batch ID=1600 
    
     D-Loss=0.07134019583463669 G-Loss=1.0054757595062256



![png](output_12_754.png)


    Epoch ID=19 Batch ID=1700 
    
     D-Loss=0.04515087231993675 G-Loss=1.223494529724121



![png](output_12_756.png)


    Epoch ID=19 Batch ID=1800 
    
     D-Loss=0.03286606818437576 G-Loss=0.9200620055198669



![png](output_12_758.png)


请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
