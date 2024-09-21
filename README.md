# Calm
序言：首先非常感谢Dian团队能够提供这次参与招新的机会，在为期一周的学习、实践、遇挫、debug、优化等等的过程中，我能够清晰的感到我自身学习能力的提升，从拿到题目时的迷茫，到一步步学习相关知识，运用所掌握工具搭建网络，最终实现一个个小目标。虽然对于某些原理我仍然处于使用“黑匣子”的状态，但在日后的学习中，我将逐步挖掘其中更深层的原理。

# Day1
## 1.1 题目整体理解
本次考核共有四大题，其中有一个附加题
- 第一题是关于一个基础网络的实现问题:`RNN`循环神经网络
  
  主要任务可分解为：

  1.搭建`RNN`网络
  
  2.实现`fashion_minist`数据集的分类任务

  3.运用自己的指标计算函数可视化并评价训练过程

- 第二题是关于`transformer`架构中的第一步：位置编码
  
  主要任务可分解为：

  1.通过自学理解题目所提供的公式实现绝对位置编码

  2.自己提供一个测试用例，用来测试绝对位置编码

  3.通过自学理解题目所提供的公式实现旋转位置编码

  4.自己提供一个测试用例，用来测试旋转位置编码

- 第三题是关于`transformer`架构中的`Multi-Head Attention`部分
  
  主要任务可分解为：

  1.理解`Multi-Head Attention`的计算过程与数学推导

  2.通过代码实现`Multi-Head Attention`

  3.自己给定一个随机矩阵，通过`Multi-Head Attention`计算注意力权重

- 附加题是关于通过复现提供的论文中的`DDPM`模型
  
  主要任务可分解为：

  1.理解论文中模型的主要实现思想+数学公式推导过程

  2.通过自学以及论文中对模型主要思想的理解，通过代码实现

  3.通过自己的方式复现结果



# Day2 & Day3
## 2.1 第一题的实现
- 通过学习对`RNN`模型的理解：


  <img width="503" alt="b3cf076c47e35c8b046e83d2b4b7176" src="https://github.com/user-attachments/assets/d7574b9a-dd83-4d03-9250-5cfa6b3f23a6">


  1.应用层面：`RNN` 是一种擅长处理序列数据的神经网络，通常用于`NLP`领域

  2.网络特点：
  
   记忆：普通的神经网络看待每个数据点是独立的，但`RNN`具有“记忆性”。它通过一个循环结构把之前的信息“记住”，然后用这些信息来帮助理解当前的数据。这就像是你在读书时记住之前的故事情节来理解新的一页。
   
   更新：RNN 通过在每一步接收新的输入，并更新它的“记忆”，来处理序列。每个时间步骤的输出不仅依赖于当前的输入，还依赖于之前的状态。

   <img width="638" alt="80bbf319b24b6e63e79d3a3b0fe6f59" src="https://github.com/user-attachments/assets/038e78ca-3a87-480e-8889-15d263480fed">


   如上图所示，该图展示出`RNN`网络的隐藏层工作原理

   其中`W_hh`参数是将前一个隐藏状态`H_(t-1)`转换到当前隐藏状态`H_t`的转换矩阵,主要通过这个过程来实现‘记忆性’

   `W_xh`参数是将当前输入`x_t`转换到当前隐藏状态`H_t`的转换矩阵，这是对当前时间步输入信息的处理

   `b_h`是偏置项
   
   这三个参数都是可以学习的

   其中`fai`函数采用的是激活函数`tanh`，原因主要是可以缓解梯度消失问题， ⁡`tanh` 函数的导数范围在 0 到 1 之间，相对平滑这有助于梯度下降算法在训练过程中更有效地更新权重。尽管梯度在输入值很大或很小时可能会接近于 0（导致梯度消 
   失）但是相较于某些其他函数，`tanh` 的梯度消失问题较为缓解，关于这个问题后续还会采用梯度裁剪法继续实现

   <img width="733" alt="d561d6413ec63e02ca7ac8b52087c05" src="https://github.com/user-attachments/assets/5c8510b1-1621-4115-8322-3ab8948ea5bc">


   上图是`RNN`的计算图

   <img width="590" alt="950a12ad0ff749841a19d1d1327d099" src="https://github.com/user-attachments/assets/f842a94e-bd83-4c6a-b5bd-ccaa4facb002">


   上图是根据计算图进行反向传播，得出的损失函数`L`对于`W_hh`和`W_hx`的偏导，因为后续的`updater`优化器采用的是`torch.optim.SGD`，所以求出损失函数对于待学习超参数的梯度十分重要。

   同时，根据计算得出的梯度结果可以看出，梯度随时间`t`呈指数变化，这样就容易出现梯度消失或者梯度爆炸的问题，解决方案在上文提及，这里不再赘述。

- 通过代码对`RNN`网络进行实现（详情见[Dian秋招第一题.ipynb](https://github.com/Calm-tech-hub/Dian-24-Autumn/blob/master/Dian%E7%A7%8B%E6%8B%9B%E7%AC%AC%E4%B8%80%E9%A2%98.ipynb)）

   1.定义`RNN`的类：

   首先在`__init__`部分，需要定义三个线性层，其中两个分别是`W_hh`以及`W_hx`对应的`nn.Linear`层，第三个是将信息整合输出的线性层`self.hidden_to_output`

   然后再`forward`部分，根据时间步按上述步骤进行时序循环，最终输出结果

   2.权重初始化：采用`Xavier`方法进行初始化

   3.定义评价标准：联想到机器学习中的评价指标--混淆矩阵，并通过学习了解，设置以下指标进行评价：`loss`,`accuracy`,`precision`,`F1-score`,`recall`这几个指标对`RNN`网络进行评估

   4.训练过程可视化：参考《动手学深度学习》这本书里提供的`Animator`类以及`Accumulator`类进行动态训练过程可视化

   动态可视化最终结果如下图所示：

   ![训练指标变化](https://github.com/user-attachments/assets/402e437f-24ce-48ef-87a9-c41ae637d3c6)

   ![优化过程图像](https://github.com/user-attachments/assets/6576ce2a-e13e-43b7-ab58-f3311aefac6e)



   5.构建训练函数：主要参数：训练网络  训练集 测试集 损失函数 训练次数 优化器

   6.进行训练前后对比：通过测试集的分类结果进行训练前后的比对，表现出训练的高效性,对比结果如下图所示：

   ![训练前分类](https://github.com/user-attachments/assets/64456c44-148e-41b2-a440-f96d2252f677)

   ![训练后分类](https://github.com/user-attachments/assets/e0285766-83cd-494c-8568-a8c481fbb8df)



## 2.2 第二题的实现
- 通过学习两种位置编码，首先对这两种编码在`transformer`中的作用进行一个整体的理解，通过一个简单的没有位置编码的transformer架构的NLP模型进行分析，发现其中两个字交换位置并没有造成最终输出的改变，比如‘我爱你’和‘你爱我’是两种截然不同的意思，但由于没有对其在整体中的位置进行描述，故而出现这种错误，然后通过数学推导理解这两种位置编码，特别是`RoPE`编码的实现过程，如下图：
  
  ![678c4b0a90808eeaf81b2a6599f1282](https://github.com/user-attachments/assets/65c37f80-9158-408c-add8-57b6f275aa0e)

  
  1.绝对位置编码：顾名思义，在绝对位置编码的公式定义中与自身在所给数据中有关的变量有两个：一个是`pos`,一个是`i`,其中`pos`指的是对应的第几号样本，`i`反映的是该变量在该样本中的第几个特征,而且在公式中并未涉及其他变量的位置，所以仅由自身在样本整体中的位置决定，故称为绝对位置编码，最终将公式的结果在附加到原来的特征信息上即可

  2.代码实现：通过定义一个`AbsPosEncoding`的类，在里面通过`pytorch`自带的数学函数实现上述数学公式，并用一个随机矩阵进行测试，详情见[Dian秋招第二题.ipynb](https://github.com/Calm-tech-hub/Dian-24-Autumn/blob/master/Dian%E7%A7%8B%E6%8B%9B%E7%AC%AC%E4%BA%8C%E9%A2%98.ipynb)

  3.旋转位置编码：这个相较于绝对位置编码要复杂的多，所以先从简单的二维情况开始考虑，首先从结果出发，将其与绝对位置编码进行比较，它在结果中引入`（m-n）`的项，以及`theta`中有`i`的元素，所以里面含有绝对位置信息与相对位置信息，下面将二维扩展至多维，如下图所示：
  
  <img width="587" alt="dbf86f2fbd16ba8786b0060b93253e0" src="https://github.com/user-attachments/assets/92401355-d113-427d-ab5c-9a1f566e9c05">


  这样就与题目所提供的公式形式基本保持一致了，至此旋转编码的公式推导也已经基本完成。

  4.代码实现：仿照绝对位置编码的格式，先定义一个RotaryPosEncoding的类，通过Pytorch以及math自带的数学公式实现旋转位置编码，在`forward`部分在通过根据公式输出结果，最后给出一个随机矩阵进行测试，其中部分`cos_pos_encoding`和`sin_pos_encoding`结果存放于
  [cos_pos_encoding](https://github.com/Calm-tech-hub/Dian-24-autumn-recruitment/blob/master/cos_pos_encoding.xlsx)
  以及[sin_pos_encoding](https://github.com/Calm-tech-hub/Dian-24-autumn-recruitment/blob/master/sin_pos_encoding.xlsx)中

## 2.3第三题的实现
-通过学习对`Multi-Head Attention`的理解：

<img width="600" alt="bfb105aef4e78248a3027fb436c3936" src="https://github.com/user-attachments/assets/8106e001-8e83-4949-b3b5-4fe3ac22e504">

这个注意力机制是transformer架构中的核心部分，其主要组成部分有两个：一个是多头，另一个是注意力机制，其主要工作机制是帮助模型在处理输入时，能够动态地集中关注输入序列中最重要的部分。它允许模型根据上下文信息自动调整权重，优先处理对当前任务最相关的信息，而忽略不相关或次要的部分，而多头的作用是可以从不同的角度、不同的子空间来捕捉输入序列中的特征。每个头关注的内容可能不同，这有助于模型捕获输入数据中更多样化的信息。

首先，对该机制处理的三个对象进行理解，即`Q、K、V`矩阵。拿在NLP细分领域机器翻译举例，Q代表的是目标语言单词的特征，K表示源语言的单词的特征，V表示K中特征包含的具体含义信息。

- 上图中的左图主要通过缩放点积注意力模型对`Q、K、V`矩阵进行处理，得出他们之间的注意力权重系数矩阵，详细步骤为：

  1.将Q矩阵乘上K矩阵的转置，这一步得到初步的注意力权重关系矩阵，反映的是`Q`与`K`之间的相关性，下面的步骤就是对这个矩阵进行处理

  2.将这个矩阵除以根号下`dk`,以此来进行缩放，根据中央极限定理，当维度`dk`增大时，点积的分布趋向于正态分布，其均值与标准差受到维度的影响。通过除以根号`dk`，我们可以将点积结果的方差调整到一个较为标准化的水平，使得其方差约为1,从而缓解梯度消失以及梯度爆炸的问题

  3.运用`mask`遮挡，`mask`遮挡主要分为填充遮挡和因果遮挡

  4.用softmax函数上述结果进行归一化处理

  5.将上述得到结果乘上V矩阵得到最终结果

- 上图中的右图描述的多头注意力机制的全部实现过程，详细步骤为：

  1.首先对`Q、K、V`矩阵根据‘头’的数量进行重塑

  2.将重塑后的`Q、K、V`矩阵分别通过一个`Linear`层，进行第一步处理

  3.通过上面的缩放点积注意力模型得出多个头的注意力权重矩阵

  4.将多个头的权重注意力矩阵进行`concat`操作，这一步在实际 代码中通过`reshape`实现

  5.将结果通过一个`Linear`层进行输出

- 代码实现部分：（详情见[Dian秋招第三题.ipynb](https://github.com/Calm-tech-hub/Dian-24-Autumn/blob/master/Dian%E7%A7%8B%E6%8B%9B%E7%AC%AC%E4%B8%89%E9%A2%98.ipynb)）

  1.根据定义，定义一个`MultiHeadAttention`的类

  2.用Xavier方法初始化权重

  3.初始化一个`Q、K、V`矩阵，运用自注意力机制，即`Q、K、V`三者相同，并进行测试

  4.运用`plt`中的热力图对注意力权重系数矩阵进行可视化，如下图所示：

  <img width="603" alt="6f8ce5ec1abb1901f1da1071243dbbc" src="https://github.com/user-attachments/assets/d1413bb8-af82-424e-8ce9-fa3d22c8aaf6">



##  2.4第四题的初步学习
- 首先建立对`DDPM`模型的初步理解：
  
  `DDPM`主要分为两大过程:
  
  1.前向传播过程：向原始提供的数据集的图片中不断添加服从高斯分布的噪音

  2.反向传播过程：通过预测该时间步的噪音，进而一步步反推，复原图片

- 对论文的初印象：

  该论文主要介绍了`DDPM`的工作机制，其大篇幅被图片占据，除此之外，便是大片的数学公式推导，故而明天的任务是复现论文中的数学公式推导


# Day4

-今天是满课的一天 时间不多 只有晚上的部分时间

  利用课余时间初步完成了数学公式的复现（有些细节并未理解清楚），如下图所示：

<img width="495" alt="d71113494d02074f57d7c75ed41cd00" src="https://github.com/user-attachments/assets/383bddec-e193-4086-a3fc-df92e94c6b9a">

<img width="464" alt="7eb0d7b8e08c556b622d43fd16589eb" src="https://github.com/user-attachments/assets/069d8e2e-8872-40d4-b44c-d4188a89569f">

<img width="443" alt="84432fe08449db464df59cc99c3b223" src="https://github.com/user-attachments/assets/d3ba82cd-0eda-48d6-9275-13cd13d64e76">

<img width="491" alt="4f483f7aa2b6b5cfc6a88c9b3af03eb" src="https://github.com/user-attachments/assets/e3f8c2f6-40aa-41e8-80ce-509a02722feb">

  回顾数学推导的过程，整体的宏观目标是确定损失函数：MSE。

  整个推导过程中始终保持的假设是马尔科夫
  
  首先通过前向传播过程根据前向概率分布公式的不断迭代，得出`x_t`与`x_0`的关系，然后进入反向扩散过程，给出反向概率分布函数，其中方差固定，均值需要训练，下面运用负对数极大似然估计进行推导，首先通过放缩得到变分下界，方法是：在后面加上前向传播概率分布函数以及 
  反向传播两者之间的KL散度，衡量两者之间的相似度，这一项恒非负，对该项通过贝叶斯公式进行化简，确定最终的变分下界，在通过定义进一步化简，展开成三项，其中第二项的分子方差过大，于是决定引入`x_0`（`x_0`可以推出`x_t`），以此来增加稳定性,再用贝叶斯公式，替换掉这 
  个分子，得到最终的三项形式，下面分别对这三项进行进一步的分析。

  其中第一项没有可学习参数，可以忽略。

  第二项是两个概率分布函数之间的KL散度，其中的方差是固定的值，因此目标是让两者之间的均值尽可能的小，通过变换最终得到关于`epsilon`的MSE，并且前面还有一个系数，论文研究发现，忽略系数效果更佳，于是选择忽略前面的常数系数。

  最后第三项通过在t=1的时间步，不添加噪声，也去掉了。在一步步推导之后，得到一个关于噪音的`MSE`函数,作为最终的损失函数。

# Day 5 & Day 6
- `DDPM`的代码实现,详情见[Dian秋招附加题_1.ipynb](https://github.com/Calm-tech-hub/Dian-24-Autumn/blob/master/Dian%E7%A7%8B%E6%8B%9B%E9%99%84%E5%8A%A0%E9%A2%98%E2%80%94%E2%80%94RNN.ipynb)
以及[Dian秋招附加题_2.ipynb](https://github.com/Calm-tech-hub/Dian-24-Autumn/blob/master/Dian%E7%A7%8B%E6%8B%9B%E9%99%84%E5%8A%A0%E9%A2%98%E2%80%94%E2%80%94Unet.ipynb)

  代码的初步思路为：

  1.根据对题目的理解初步搭建代码的框架，根据`DDPM`模型的定义，搭建一个`DDPM`的类，里面包含定义前向过程的数学表达式的函数，以及反向的采样过程。并导入`fashion_minist`的数据集，对其进行训练，生成训练后的图像。

  2.对于题目中提到利用`RNN`网络，就用`RNN`网络进行噪声预测即可

  3.训练函数，对`RNN`预测噪声的模型进行训练以及优化，损失函数定为`MSE`函数，优化器初步采用随机梯度下降优化器

  4.评估函数：实时更新两者的`Loss`函数，并对RNN采用`accuracy`进行评估。

  5.训练过程可视化：将噪声以及去噪的图像，在一整轮的时间步的迭代过程后展示出来。

  开始具体实现：(具体代码详见[Dian秋招附加题.ipynb](https://github.com/Calm-tech-hub/Dian-24-autumn-recruitment/blob/master/dain%E7%A7%8B%E6%8B%9B%E9%99%84%E5%8A%A0%E9%A2%98.ipynb))

  实验结果：

  ![DDPM_RNN](https://github.com/user-attachments/assets/1449e613-d126-4983-a892-f3e9b8a5f298)

  可见，可以出现一些大致的轮廓，但是效果很差。（不过该说不说，是挺快）

  于是开始反思，根据之前的了解，`RNN`网络的主要应用领域在于`NLP`领域，其对这种图像特征提取的工作效果并不好，于是开始查阅资料，最终确定用`Unet`网络代替`RNN`网络进行噪声预测.

  `Unet`网络的主要组成部分有：编码器（双层卷积层，池化层），以及译码器（双层卷积层，上采样层（即转置卷积层））组成，并通过跳跃连接，最终尽可能全面的提取图片中的特征。


  实现过程中遇到的困难：

  1.第一次用`minist`数据集进行实验，发现效果很差，如下图所示：

  <img width="648" alt="b627675ecf0830c9f6e4367f220d00b" src="https://github.com/user-attachments/assets/f78836d0-7a3e-4320-a8d4-4c800fd74721">

  于是开始反思代码，将问题锁定到优化器上，通过查阅资料，决定将`optim.SGD`换成`optim.Adam`，再次进行训练，训练前与训练后的对比图如下：

  <img width="595" alt="e3390c41bca8c950c00e9e193579896" src="https://github.com/user-attachments/assets/65c3ff58-7926-4d13-901f-5feba9784b8f">

  <img width="599" alt="ede76c43d981e7ad3d9d311fd29ee72" src="https://github.com/user-attachments/assets/92ed20b0-a51b-4aba-8624-fe0df3ed6926">

  比之前效果好点，但还不是很明显

  2.下面对`fashion_minist`数据集进行训练，最终得到的训练结果如下图所示（训练时间明显比`minist`数据集长，训练过程更加复杂）：

  <img width="597" alt="dbd3e24523607cd5c6549f927ebd451" src="https://github.com/user-attachments/assets/47c4877c-76b2-40c0-8ab0-b9cb003eba9c">

  <img width="595" alt="89d538afe45823c97f9e05d529b5925" src="https://github.com/user-attachments/assets/284eba8d-530f-4dd8-bff4-54baaa03e9f9">

  在训练44次之后，效果较为明显，比上面的效果好一点

  3.开始反思如何优化图像复原效果：
  
  首先图像复原的关键环节在于如何更准确的预测噪声。而这个问题就归结于`Unet`层的性能，所以第一个着手点就在`Unet`层，先通过以下手段进行改进：

  使用 `Leaky ReLU` 激活函数：增强非线性性质

  添加残差连接

  首先对`Fashion_Minist`数据集进行训练，训练一次与训练多次的对比结果如下：

  ![DDPM1](https://github.com/user-attachments/assets/f826be64-0d62-4bb0-a891-36bb599f6130)

  ![DDPM2](https://github.com/user-attachments/assets/7caa9085-5cb4-42b2-804f-402ceaf09cb6)

  效果有所改善

  再对`Minist`数据集进行训练，结果如下：

  ![DDPM4](https://github.com/user-attachments/assets/2c3f44d9-8963-4435-b44e-6ddf3f1f91625)

  也还行，比之前好,基本可以辨认出来是什么数字了

# Day 7

要准备计算机三级了，要来不及了，目前就先这样吧。
