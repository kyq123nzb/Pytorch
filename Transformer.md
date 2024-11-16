# 一、Transformer基本介绍

​	Transformer是一种用于**自然语言处理（NLP）**和其他序列到序列（sequence-to-sequence）任务的深度学习模型架构，它在2017年由Vaswani等人首次提出。Transformer架构引入了**自注意力机制**（self-attention mechanism），这是一个关键的创新，使其在处理序列数据时表现出色。

Transformer的一些重要组成部分和特点：

​	**自注意力机制**（Self-Attention）：这是Transformer的核心概念之一，它使模型能够**同时考虑**输入序列中的**所有位置**，而不是像循环神经网络（RNN）或卷积神经网络（CNN）一样逐步处理。自注意力机制允许模型根据输入序列中的**不同部分来赋予不同的注意权重**，从而更好地捕捉语义关系。
​	**多头注意力**（Multi-Head Attention）：Transformer中的自注意力机制被扩展为多个注意力头，每个头可以学习不同的注意权重，以更好地捕捉不同类型的关系。多头注意力**允许模型并行处理不同的信息子空间**。
​	**堆叠层**（Stacked Layers）：Transformer通常由多个相同的编码器和解码器层堆叠而成。这些堆叠的层有助于模型学习复杂的特征表示和语义。
​	**位置编码**（Positional Encoding）：由于Transformer没有内置的序列位置信息，它需要**额外的位置编码**来表达输入序列中单词的位置顺序。
​	**残差连接和层归一化**（Residual Connections and Layer Normalization）：这些技术有助于减轻训练过程中的梯度消失和爆炸问题，使模型更容易训练。
​	**编码器和解码器**：Transformer通常包括一个编码器用于处理输入序列和一个解码器用于生成输出序列，这使其适用于序列到序列的任务，如机器翻译。

## 1-1、Transformer的结构

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3319e3d6922a2e7f2499a3130d3b5925.png#pic_center)

​	**Nx = 6，Encoder block由6个encoder堆叠而成，图中的一个框代表的是一个encoder的内部结构，一个Encoder是由Multi-Head Attention和全连接神经网络Feed Forward Network构成。**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/20c5baff36eedc6100d9f107e4fe3c95.png#pic_center)

（每一个编码器都对应上面的一个Encoder）

​	Transformer的编码组件是由**6个编码器**叠加在一起组成的，解码器同样如此。所有的编码器在结构上是相同的，但是它们之间并没有共享参数。编码器的简略结构如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/630deb7da181d99eb9dd7d70f6b4da98.png#pic_center)

​	从编码器输入的句子首先会经过一个**自注意力层**，这一层帮助编码器在对每个单词编码的时候时刻**关注句子的其它单词**。解码器中的解码注意力层的作用是**关注输入句子的相关部分**，类似于seq2seq（序列到序列学习）的注意力。

​	原结构中使用到的是多头注意力机制（Multi-Head Attention）



## 1-2、自注意力机制

​	自注意力的作用：随着模型处理输入序列的每个单词，自注意力会**关注整个输入序列的所有单词**，帮助模型对本单词更好地进行编码。在处理过程中，自注意力机制会将**对所有相关单词的理解**融入到我们正在处理的单词中。

​	更具体的功能如下：

​	序列建模：自注意力可以用于**序列数据（例如文本、时间序列、音频等）的建模**。它可以捕捉序列中不同位置的依赖关系，从而更好地理解上下文。这对于机器翻译、文本生成、情感分析等任务非常有用。
​	并行计算：自注意力可以**并行计算**，这意味着可以有效地在现代硬件上进行加速。相比于RNN和CNN等序列模型，它**更容易在GPU和TPU等硬件上进行高效的训练和推理**。（因为在自注意力中可以并行的计算得分）
​	长距离依赖捕捉：传统的循环神经网络（RNN）在处理长序列时可能面临梯度消失或梯度爆炸的问题。自注意力可以**更好地处理长距离依赖关系**，因为它**不需要按顺序处理输入序列**。

### 1、自注意力的结构

（缩放点积注意力：一种常用的自注意力机制,用于在深度学习中对序列数据进行建模）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/49c88545fae57dbf255c8ab9fd279110.png#pic_center)

### 2、自注意力的计算

#### 	（1）生成三个向量

从每个编码器的输入向量（每个单词的**词向量**，即Embedding，可以是任意形式的词向量，比如说word2vec，GloVe，one-hot编码）中生成三个向量，即**查询向量、键向量和一个值向量**。（这三个向量是通过词嵌入与三个权重矩阵即
$$
W^Q、  W^K、  W^V
$$
**相乘**后创建出来的），**新向量在维度上往往比词嵌入向量更低**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8d3240e15889168ca8c37b09684eed67.png#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cc00beb97c344a486d07e3d9e8a58f06.png#pic_center)

​		更一般的，将以上所得到的**查询向量、键向量、值向量**组合起来就可以得到三个向量矩阵Query、Keys、Values。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d027e1a13de1965169178cb2e5eb9ec1.png#pic_center)

#### 	（2）计算得分

​	假设我们在为这个例子中的第一个词“Thinking”计算自注意力向量，我们需要拿输入句子中的每个单词对“Thinking”打分。这些分数是通过所有输入句子的单词的键向量与“Thinking”的查询向量**相点积**来计算的。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/59ee8dec83e584b860869ef701a3a4e3.png)

#### 	（3&4）将分数除以8

​	8是论文中使用的键向量的维数64的平方根，这会**让梯度更稳定**。这里也可以使用其它值，8只是默认值，这样做是为了防止内积过大，然后**通过softmax传递结果**。随着模型处理输入序列的每个单词，自注意力会关注整个输入序列的所有单词，帮助模型对本单词更好地进行编码。softmax的作用是**使所有单词的分数归一化**，得到的分数都是正值且和为1。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bc1a451b1bb9c568a382f9d08fa28341.png)

​	这个softmax分数决定了**每个单词对编码当下位置（“Thinking”）的贡献**。显然，已经在这个位置上的单词将获得最高的softmax分数，

#### 	（5）将每个值向量乘以softmax分数

​	为了准备之后将它们求和，这里的直觉是希望关注语义上相关的单词，并**弱化不相关的单词**(例如，让它们乘以0.001这样的小数)。

​	**Softmax函数**：或称**归一化指数函数**，将**每一个元素的范围都压缩到(0，1)之间**，并且**所有元素的和为1**

#### 	（6）对加权值向量求和

​	即**得到自注意力层在该位置的输出**(在我们的例子中是对于第一个单词)。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/00994ceb6bf9e66db19611c496463364.png#pic_center)

整体的计算图如图所示：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e976d386a1aad85c2efb7fc965099c27.png#pic_center)

​	最终得到了自注意力，并将得到的向量**传递给前馈神经网络**。

### 3、自注意力层的完善——“多头”注意力机制

对应整体结构图中的**Multi——Head Attention**

​	1、扩展了模型专注于不同位置的能力。
​	2、有**多个查询/键/值权重矩阵集合**，（Transformer使用八个注意力头）并且每一个都是**随机初始化**的。和上边一样，用矩阵X乘以WQ、WK、WV来产生查询、键、值矩阵。
​	3、self-attention只是使用了一组WQ、WK、WV来进行变换得到查询、键、值矩阵，而Multi-Head Attention**使用多组WQ，WK，WV得到多组查询、键、值矩阵**，然后每组分别计算得到一个Z矩阵。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a923f7bb907110650448d4b773bf0671.png#pic_center)

​	前馈层只需要**一个矩阵**，则把得到的8个矩阵拼接在一起，然后用一个附加的权重矩阵
$$
W^O
$$
与它们相乘。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fd6af04ca65df88d3f13f4aaf987b0f3.png#pic_center)

**总结整个流程：**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0b8dacfc201e24ef7dc0e690b41b998c.png#pic_center)



**编码it一词时，不同注意力的头集中在哪里，当我们编码it这个单词时：（图中只列举出了两个注意力）：**

- 其中一个注意力头集中在The animal
- 另一个注意力头集中在tire上。即形象解释it代指的是animal和tire。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f4767d9bd1b2d19029a45867f92efb12.png#pic_center)

## 1-3、使用位置编码表示序列的顺序

​	为什么要用位置编码？

​	如果不添加位置编码，那么无论单词在什么位置，它的**注意力分数都是确定的**。这不是我们想要的。
为了**理解单词顺序**，Transformer为**每个输入的词嵌入添加了一个向量**，这样能够更好的表达词与词之间的关系。词嵌入与位置编码**相加**，而不是拼接，他们的效率差不多，但是拼接的话维度会变大，所以不考虑。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/06e447ba79993632bed97672d4578c21.png)

​	为了让模型理解单词的顺序，我们添加了位置编码向量，这些向量的值**遵循特定的模式**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/beeb9fb0c7074051b1a064edb19a73bc.png)

## 1-4、Add&Normalize

​	在经过多头注意力机制得到矩阵Z之后，并没有直接传入全连接神经网络，而是经过了一步**Add&Normalize**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/29a24a78b70aa77ffd41b5ae2bfdc5e7.png#pic_center)

Add & Norm 层由 **Add 和 Norm 两部分**组成，其计算公式如下：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6818e843a25c1db5e970b5c7e470af8b.png#pic_center)

​	其中 X表示 Multi-Head Attention 或者 Feed Forward 的**输入**，MultiHeadAttention(X) 和 FeedForward(X) 表示**输出** (输出与输入 X 维度是一样的，所以可以相加)。

### 1、Add

​	Add，就是在z的基础上**加了一个残差块X**，加入残差块的目的是为了**防止在深度神经网络的训练过程中发生退化的问题**，退化的意思就是深度神经网络通过**增加网络的层数**，Loss逐渐减小，然后**趋于稳定达到饱和**，然后**再继续增加网络层数，Loss反而增大**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b9951837dd639046ca37c9fba8b4efc5.png#pic_center)

### 2、ResNet残差神经网络

​	为了了解残差块，我们引入ResNet残差神经网络，神经网络退化指的是在**达到最优网络层数之后**，神经网络**还在继续训练导致Loss增大**，对于多余的层，我们需要保证**多出来的网络进行恒等映射**。只有进行了恒等映射之后才能保证这多出来的神经网络**不会影响到模型的效果**。残差连接主要是为了**防止网络退化**。
​	![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/df2011b93622dc7da054b8853b16bafd.png#pic_center)

​	上图就是构造的一个残差块，X是输入值，F（X）是经过第一层线性变换后并且激活的输出，在第二层线性变化之后，激活之前，F（X）加入了这一层输入值X，然后再进行激活后输出。

​	要恒等映射，我们只需要让F(X)=0就可以了。x经过线性变换（随机初始化权重一般偏向于0），输出值明显会偏向于0，而且经过激活函数Relu会将负数变为0，过滤了负数的影响。

​	这样当网络自己决定哪些网络层为冗余层时，使用ResNet的网络很大程度上解决了学习恒等映射的问题，用学习残差F(x)=0更新该冗余层的参数来代替学习h(x)=x更新冗余层的参数。

### 3、Normalize

​	归一化目的：
​		1、加快训练速度
​		2、提高训练的稳定性
​	使用到的归一化方法是**Layer Normalization**。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cc6ff426fbe027c7998efd23a8e0a833.png#pic_center)

​	**LN**是在**同一个样本**中**不同神经元**之间进行归一化，而BN是在**同一个batch**中**不同样本**之间的**同一位置**的神经元之间进行归一化。
​	**BN**是对于**相同的维度**进行归一化，但是咱们NLP中输入的都是词向量，一个300维的词向量，单独去分析它的每一维是没有意义地，在每一维上进行归一化也是适合地，因此这里选用的是LN。

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/cd46c3d200f9126819446f66c24ef099.png#pic_center)

## 1-5、全连接层Feed Forward

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/98f4c96bfa951f24b2d1b8c8686582bd.png)

​	全连接层是一个**两层的神经网络**，先**线性变换**，然后**ReLU非线性**，再**线性变换**。
​	这两层网络就是为了**将输入的Z映射到更加高维的空间**中然后通过非线性函数ReLU进行筛选，筛选完后再变回原来的维度。
​	经过6个encoder后输入到decoder中。

【ReLU非线性：https://blog.csdn.net/lph159/article/details/138535817?fromshare=blogdetail&sharetype=blogdetail&sharerId=138535817&sharerefer=PC&sharesource=2301_80864377&sharefrom=from_link】

## 1-6、Decoder整体结构

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/da0673543394c7623497447b1211054d.png#pic_center)

​	和Encoder Block一样，Decoder也是由**6个decoder**堆叠而成的，Nx=6。包含两个 Multi-Head Attention 层。第一个 Multi-Head Attention 层采用了 Masked 操作。第二个 Multi-Head Attention 层的K, V矩阵使用 Encoder 的编码信息矩阵C进行计算，而Q使用上一个 Decoder block 的输出计算。

### Masked Multi-Head Attention

​	与Encoder的Multi-Head Attention计算原理一样，只是多加了一个mask码。mask 表示掩码，它**对某些值进行掩盖，使其在参数更新时不产生效果。**Transformer 模型里面涉及两种 mask，分别是 padding mask 和 sequence mask。

#### 	（1）padding mask

​	因为**每个批次输入序列长度**是不一样的，所以我们要对输入序列**进行对齐**。具体来说，就是给在较短的序列后面填充 0。但是如果输入的序列太长，则是截取左边的内容，把多余的直接舍弃。因为这些填充的位置，其实是没什么意义的，所以我们的attention机制不应该把注意力放在这些位置上，所以我们需要进行一些处理。

​	具体的做法：把这些位置的值加上一个**非常大的负数(负无穷)**，这样的话，经过 softmax，这些位置的概率就会接近0！

#### 	（2）sequence mask

​	sequence mask 是为了使得 decoder **不能看见未来的信息**。对于一个序列，在 time_step 为 t 的时刻，我们的解码输出应该**只能依赖于 t 时刻之前的输出，而不能依赖 t 之后的输出**。因此我们需要想一个办法，把 t 之后的信息给隐藏起来。这在训练的时候有效，因为训练的时候每次我们是将target数据完整输入进decoder中地，预测时不需要，预测的时候我们只能得到前一时刻预测出的输出。

​	具体做法：产生一个上三角矩阵，**上三角的值全为0**。把这个矩阵**作用在每一个序列上**，就可以达到我们的目的。

**【注意】**

​	1、在Encoder中的Multi-Head Attention也是需要进行mask的，只不过**Encoder中只需要padding mask即可**，**而Decoder中需要padding mask和sequence mask**。

​	2、Encoder中的Multi-Head Attention是基于Self-Attention的，Decoder中的**第二个Multi-Head Attention就只是基于Attention**，它的输入Quer来自于Masked Multi-Head Attention的输出，Keys和Values来自于Encoder中最后一层的输出。

## 1-7、输出

​	Output如图中所示，首先**经过一次线性变换**（线性变换层是一个**简单的全连接神经网络**，它可以把解码组件产生的向量投射到一个比它大得多的，被称为对数几率的向量里），然后**Softmax得到输出的概率分布**（softmax层会把向量变成概率），然后**通过词典，输出概率最大的对应的单词作为我们的预测输出**。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/39496009ca4e4e4b42cf8f6b946af914.png#pic_center)

【全连接神经网络：https://blog.csdn.net/qq_43276566/article/details/139709553?fromshare=blogdetail&sharetype=blogdetail&sharerId=139709553&sharerefer=PC&sharesource=2301_80864377&sharefrom=from_link】

## 1-8、transformer的优缺点：

**优点：**
	1、效果好
	2、可以并行训练，速度快
	3、很好的解决了长距离依赖的问题

**缺点：**
	完全基于self-attention，对于**词语位置之间的信息有一定的丢失**，虽然加入了positional encoding来解决这个问题，但也还存在着可以优化的地方。



# 二、Self-Attention的实现

## 2-0、过程

- 准备输入
- 初始化参数
- 获取key，query和value
- 给input1计算attention score
- 计算softmax
- 给value乘上score
- 给value加权求和获取output1
- 重复步骤4-7，获取output2，output3

## 2-1、准备输入（词嵌入向量）

```python
import torch
x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)
print(x)
```

**输出：**

```python
tensor([[1., 0., 1., 0.],
        [0., 2., 0., 2.],
        [1., 1., 1., 1.]])
```

## 2-2、初始化参数（Q、K、V矩阵）

**Note**： Q、K、V矩阵在神经网络初始化的过程中，一般都是**随机采样**完成并且比较小，可以根据**想要输出的维度**来确定 Q、K、V矩阵的维度。

```python
w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

print("Weights for key: \n", w_key)
print("Weights for query: \n", w_query)
print("Weights for value: \n", w_value)

```

**输出：**

```
Weights for key: 
 tensor([[0., 0., 1.],
        [1., 1., 0.],
        [0., 1., 0.],
        [1., 1., 0.]])
Weights for query: 
 tensor([[1., 0., 1.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 1., 1.]])
Weights for value: 
 tensor([[0., 2., 0.],
        [0., 3., 0.],
        [1., 0., 3.],
        [1., 1., 0.]])
```

## 2-3、获取key，query和value

```python
keys = x @ w_key
# 如果x是一个形状为(m, n)的矩阵，而w_key是一个形状为(n, p)的矩阵，那么x @ w_key的结果将是一个形状为(m, p)的矩阵，其元素是x的行与w_key的列的点积
querys = x @ w_query
values = x @ w_value

print("Keys: \n", keys)
# tensor([[0., 1., 1.],
#         [4., 4., 0.],
#         [2., 3., 1.]])

print("Querys: \n", querys)
# tensor([[1., 0., 2.],
#         [2., 2., 2.],
#         [2., 1., 3.]])
print("Values: \n", values)
# tensor([[1., 2., 3.],
#         [2., 8., 0.],
#         [2., 6., 3.]])
```

**下图为得到的key，query和value**：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/284847367ce4318319ddcdb90c397a81.png)

## 2-4、计算注意力分数

```python
attn_scores = querys @ keys.T
print(attn_scores)
```

**输出：**

```python
tensor([[ 2.,  4.,  4.],
        [ 4., 16., 12.],
        [ 4., 12., 10.]])
```

## 2-5、计算softmax

```python
from torch.nn.functional import softmax

attn_scores_softmax = softmax(attn_scores, dim=-1)
print(attn_scores_softmax)

# tensor([[6.3379e-02, 4.6831e-01, 4.6831e-01],
#         [6.0337e-06, 9.8201e-01, 1.7986e-02],
#         [2.9539e-04, 8.8054e-01, 1.1917e-01]])

# 为了使得后续方便，这里简略将计算后得到的分数赋予了一个新的值
# For readability, approximate the above as follows

attn_scores_softmax = [
  [0.0, 0.5, 0.5],
  [0.0, 1.0, 0.0],
  [0.0, 0.9, 0.1]
]
attn_scores_softmax = torch.tensor(attn_scores_softmax)
print(attn_scores_softmax)
```

**输出：**

```python
tensor([[0.0000, 0.5000, 0.5000],
        [0.0000, 1.0000, 0.0000],
        [0.0000, 0.9000, 0.1000]])
```

## 2-6、给value乘上score

```python
weighted_values = values[:,None] * attn_scores_softmax.T[:,:,None]
print(weighted_values)
```

​	**`values[:,None]`**：这里`values`是一个多维数组（可能是二维的，表示不同位置的向量值）。`[:,None]`的作用是在`values`数组的第二个维度上增加一个大小为1的新维度，这通常是为了让`values`的形状与后续操作的数组形状兼容。如果`values`的形状原本是`(N, D)`（N是序列长度，D是特征维度），那么`values[:,None]`的形状会变成`(N, 1, D)`。	

​	**`attn_scores_softmax.T[:,:,None]`**：`attn_scores_softmax`是注意力分数经过softmax函数处理后的结果，通常表示不同位置之间的相关性或重要性。`.T`表示对`attn_scores_softmax`进行转置。如果`attn_scores_softmax`的形状原本是`(N, N)`（表示N个位置两两之间的注意力分数），那么转置后的形状是`(N, N)`（虽然数值没有变化，但行列位置互换了）。接着，`[:,:,None]`在转置后的数组的第三个维度上增加一个大小为1的新维度，使其形状变为`(N, N, 1)`。	

​	**`values[:,None] \* attn_scores_softmax.T[:,:,None]`**：这一步是元素级别的乘法（也称为Hadamard积），发生在`values`扩展后的形状`(N, 1, D)`和`attn_scores_softmax.T`扩展后的形状`(N, N, 1)`之间。由于`attn_scores_softmax.T`的第三个维度是1，而`values`的第二个维度也是1，这两个维度上的1使得乘法可以沿着这两个维度进行广播（Broadcasting），最终得到一个形状为`(N, N, D)`的数组。这个数组中的每个元素都是原始`values`中的向量值经过相应注意力分数加权后的结果。

​	**`print(weighted_values)`**：最后，打印出加权后的值`weighted_values`，其形状为`(N, N, D)`，表示每个位置对其他位置向量值的加权和，其中权重由注意力分数决定。

**输出：**

```python
tensor([[[0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000]],

        [[1.0000, 4.0000, 0.0000],
         [2.0000, 8.0000, 0.0000],
         [1.8000, 7.2000, 0.0000]],

        [[1.0000, 3.0000, 1.5000],
         [0.0000, 0.0000, 0.0000],
         [0.2000, 0.6000, 0.3000]]])
```

##### **【具体如何实现？】**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2e387d94c0263153f6f9ab9c9176da20.png)

## 2-7、给value加权求和获取output(得到input1的结果向量)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b021027fb8b935d6f320a54b5ae19abb.png)

**重复步骤4-7，获取到input2、input3的结果向量**





# 三、Stochastic Depth

Stochastic Depth指在**网络训练**时**随机丢弃**一部分网络结构，而在**测试时使用完整的网络结构**。

```
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224):
        super().__init__()
 
        ...
 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # stochastic depth decay rule
        
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
```

**这段代码定义了一个包含多个预测模块Block的神经网络模型。**

1. **类定义**：`VisionTransformer`类继承自`nn.Module`，这是PyTorch中所有神经网络模块的基类。通过继承这个基类，`VisionTransformer`可以获得构建神经网络所需的基本功能。
2. **初始化函数** (`__init__`)：这个函数是类的构造函数，用于初始化Vision Transformer模型。它接受一个参数`img_size`，默认为224，表示输入图像的尺寸。
3. **省略部分** (`...`)：在初始化函数的中间部分，省略了其他重要的初始化步骤，如定义嵌入维度（`embed_dim`）、头数（`num_heads`）、MLP比率（`mlp_ratio`）、QKV偏置（`qkv_bias`）、QK缩放因子（`qk_scale`）、丢弃率（`drop_rate`、`attn_drop_rate`）、随机深度丢弃率（`drop_path_rate`）、深度（`depth`）、归一化层（`norm_layer`）、激活层（`act_layer`）等参数，以及可能的其他网络层或组件。
4. **随机深度衰减规则**：`dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]`这行代码生成一个列表`dpr`，用于在Transformer块中应用随机深度（Stochastic Depth）技术。`torch.linspace(0, drop_path_rate, depth)`生成一个从0到`drop_path_rate`均匀分布的`depth`个值的张量，然后通过`.item()`方法将其转换为Python列表中的标量值。这个列表`dpr`中的每个值对应于每个Transformer块中的**随机深度丢弃率**，**随着深度的增加，丢弃率逐渐增大，这有助于防止过拟合并提高训练效率**。
5. **Transformer块**：`self.blocks`是一个由多个`Block`对象组成的序列，每个`Block`代表Vision Transformer中的一个Transformer块。这些块通过`nn.Sequential`串联起来，形成一个完整的Transformer架构。每个`Block`的构造依赖于多个参数，包括嵌入维度、头数、MLP比率、QKV偏置、QK缩放因子、丢弃率、注意力丢弃率、随机深度丢弃率、归一化层和激活层等。
6. **循环创建块**：通过列表推导式和`for i in range(depth)`循环，为每个深度级别创建一个`Block`实例，并将它们添加到`self.blocks`序列中。`dpr[i]`确保每个块根据其深度级别具有不同的随机深度丢弃率。

​	综上所述，这段代码是Vision Transformer模型的核心部分，它定义了模型的架构和参数，包括如何应用随机深度技术来优化训练过程。



# 四、丢弃率

## 4-1、定义与原理

1. 定义

   ​	丢弃率是指在神经网络的训练过程中，按照一定的概率**随机丢弃**网络中的神经元（或其连接），使其**暂时不参与前向传播和反向传播**的过程。

2. 原理

   ​	丢弃率通过**随机性**引入了一种**正则化**效果，有助于防止神经网络在训练过程中**过度依赖某些特定的神经元或特征，**从而减轻过拟合现象。

   ​	当丢弃率较高时，网络需要学习**如何以不同的方式组合剩余的神经元来做出预测**，这增**强了网络的泛化能力**。

## 4-2、应用方式

1. 在训练阶段

   ​	在每次训练迭代中，根据丢弃率随机选择一部分神经元**将其输出置为零**，这些神经元在当前的迭代中不参与计算。

   ​	由于**丢弃是随机**的，因此每次迭代中实际参与计算的神经元集合都是不同的，这迫使网络学习更加鲁棒的特征表示。

2. 在测试阶段

   ​	**在测试或推理阶段，通常不使用丢弃率，即所有神经元都参与计算。**

   ​	为了补偿训练阶段由于丢弃率引入的额外信息，通常会将训练好的模型权重按照丢弃率进行缩放（例如，如果丢弃率是0.5，则在测试阶段将权重乘以2）。

## 4-3、选择与调整

1. 选择丢弃率

   ​	丢弃率的选择取决于具体的任务、数据集和网络架构。

   ​	常用的丢弃率值包括0.2、0.5等，但最佳值通常需要通过实验来确定。

2. 调整丢弃率

   ​	在训练过程中，可以通过网格搜索、随机搜索或贝叶斯优化等方法来调整丢弃率，以找到最佳的模型性能。

   ​	需要注意的是，过高的丢弃率可能会导致模型欠拟合，而过低的丢弃率则可能不足以起到正则化的效果。

## 4-4、变体与方法

1. 标准Dropout

   ​	最基本的Dropout方法，随机丢弃神经元。

2. 变体Dropout

   ​	如DropConnect，它随机丢弃神经元之间的连接而不是神经元本身。

   ​	还有空间Dropout（Spatial Dropout），特别适用于卷积神经网络（CNN）中的卷积层。

3. 其他正则化方法

   ​	除了Dropout之外，还有其他正则化方法如L1正则化、L2正则化和权重衰减等，它们可以与Dropout结合使用以增强模型的泛化能力。

## 4-5、结论

​	丢弃率是深度学习中一种简单而有效的正则化技术，通过随机丢弃网络中的神经元来防止过拟合。在选择和调整丢弃率时，需要考虑具体的任务、数据集和网络架构，并通过实验来确定最佳值。同时，还可以结合其他正则化方法来进一步增强模型的性能。



# 五、代码示例

```
def drop_path(x, drop_prob: float = 0., training: bool = False):

    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    
    每个样本的下降路径（随机深度）（当应用于残差块的主路径时）。
	这与我为EfficientNet等网络创建的DropConnect impl相同，但是，
	原始名称具有误导性，因为“Drop Connect”是另一篇论文中的不同形式的dropout。。。
	见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择了
	将图层和参数名称更改为“drop-path”，而不是将DropConnect混合作为图层名称并使用
	以“存活率”为论据。
    """
    
    if drop_prob == 0. or not training:
        return x
        
    keep_prob = 1 - drop_prob
    
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    # work with diff dim tensors, not just 2D ConvNets(使用diff-dim张量，而不仅仅是2D卷积网络)
    
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    
    random_tensor.floor_()  # binarize(二值化)
    
    output = x.div(keep_prob) * random_tensor
    
    return output
```

## drop_path

​	这段代码定义了一个名为 **`drop_path` 的函数**，该函数实现了随机深度（Stochastic Depth）的一种形式，即在**残差块（residual blocks）的主路径上按样本丢弃路径**。这种方法是深度学习中**用于正则化**的一种技术，特别是在**训练非常深的神经网络时**。以下是对该函数的详细解释：

**函数参数**

- `x`: 输入张量，通常是网络中的特征图或激活值。
- `drop_prob`: **丢弃概率**，一个浮点数。它决定了**每个路径（或神经元连接）被丢弃的概率**。
- `training`: 一个**布尔值**，指示当前**是否处于训练模式**。如果不在训练模式，则不进行丢弃。

**函数逻辑**

1. **检查丢弃概率和训练模式**

   ​	如果 `drop_prob` 为 0 或不在训练模式，则直接返回输入 `x`，不进行任何操作。

2. **计算保留概率**

   ​	`keep_prob = 1 - drop_prob`，即**每个路径被保留的概率**。

3. **生成随机张量**

   ​	`shape = (x.shape[0],) + (1,) * (x.ndim - 1)`：创建一个与输入 `x` 形状相匹配的随机张量形状，但除了第一个维度（通常是批次大小）外，其他维度都是1。这样做是为了**确保随机性可以广播到 `x` 的所有元素上**。

   ​	`random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)`：生成一个与 `x` 数据类型和设备相同的随机张量，并将其值范围调整为 `[keep_prob, 1 + keep_prob)`。

4. **二值化随机张量**

   ​	`random_tensor.floor_()`：将随机张量**向下取整**到最接近的整数，实际上是将值转换为0或1（因为值要么在 `[0, keep_prob)` 范围内被取整为0，要么在 `[keep_prob, 1)` 范围内被取整为1）。

5. **计算输出**

   ​	`output = x.div(keep_prob) * random_tensor`：首先，将 `x` 除以 `keep_prob` 以补偿在训练过程中被丢弃的路径。然后，将这个结果乘以 `random_tensor`，以根据随机张量的值（0或1）丢弃或保留路径。

**注意事项**

- 这个函数通常用在残差块中，特别是当构建非常深的网络时（如ResNet的变体或Vision Transformer）。
- 随机深度的引入有助于减轻深度网络中的梯度消失问题，并提高训练效率。
- 在测试时，通常不使用随机深度（即 `drop_prob` 设置为0），并且模型权重在推理时不需要进行任何特殊的缩放，因为训练时的权重已经通过除以 `keep_prob` 进行了调整。然而，在实际应用中，有时会在训练结束后将权重重新缩放回原始比例（即不除以 `keep_prob`），但这取决于具体的实现和训练策略。

总的来说，`drop_path` 函数是实现随机深度正则化的一种有效方法，有助于提升深度神经网络的性能和稳定性。



```
class DropPath(nn.Module):

    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    每个样本的下降路径（随机深度）（当应用于残差块的主路径时）。
    """
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
```

## DropPath

**类定义**

- `class DropPath(nn.Module)`: 定义一个名为 `DropPath` 的新类，它继承自 PyTorch 的 `nn.Module` 类。

**初始化方法**

- ```
  def __init__(self, drop_prob=None)
  ```

  类的初始化方法。

  ​	`self.drop_prob = drop_prob`: 将传入的 `drop_prob`（丢弃概率）保存到类的实例变量中。如果没有提供 `drop_prob`，则默认为 `None`（但在实际使用中，通常会在实例化时提供一个具体的值）。

**前向传播方法**

- ```
  def forward(self, x)
  ```

  ​	定义了模块的**前向传播逻辑**。

  ​	`return drop_path(x, self.drop_prob, self.training)`: 调用 `drop_path` 函数，将输入张量 `x`、丢弃概率 `self.drop_prob` 和当前模块的训练模式 `self.training` 作为参数传入。然后返回 `drop_path` 函数的输出。

**注意事项**

1. **`drop_path` 函数**：在代码中，`drop_path` 函数被当作一个已经定义好的函数来使用。确保在您的代码库中已经定义了 `drop_path` 函数，或者从其他模块中正确导入了它。
2. **训练模式**：`self.training` 是 `nn.Module` 类的一个属性，它根据模块的 `train()` 和 `eval()` 方法来设置。当调用 `module.train()` 时，`module.training` 被设置为 `True`；当调用 `module.eval()` 时，它被设置为 `False`。这允许 `DropPath` 模块在训练时应用丢弃，而在评估（或推理）时不应用。
3. **默认参数**：虽然 `drop_prob` 的默认值是 `None`，但在实际使用中，您可能需要为其提供一个具体的数值（通常在0到1之间），以控制丢弃的概率。
4. **随机性**：由于 `drop_path` 函数内部使用了随机性（通常是通过调用 `torch.rand` 或类似函数），因此每次在训练模式下对同一个输入调用 `forward` 方法时，可能会得到不同的输出。这是期望的行为，因为它有助于正则化模型并减少过拟合。
5. **性能影响**：在推理时（即 `self.training` 为 `False` 时），`DropPath` 模块不会对输入进行任何修改，因此不会对模型的推理性能产生负面影响。



```
class PatchEmbed(nn.Module):

    """
    2D Image to Patch Embedding
    二维图像到补丁嵌入
    """
    
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
```

## PatchEmbed

​	这段代码定义了一个名为`PatchEmbed`的类，它是`nn.Module`的子类，用于**将2D图像分割成小块**（patches），并**将这些小块嵌入到一个更高维度的空间中**。这是Vision Transformer（ViT）等基于Transformer的图像处理模型中的一个关键步骤。下面是对这个类的详细解释：

**初始化方法 `__init__`**

- `img_size=224`: 输入图像的默认大小，这里假设图像是正方形的，宽度和高度都是224像素。
- `patch_size=16`: 每个小块的大小，默认是16x16像素。
- `in_c=3`: 输入图像的通道数，对于RGB图像来说是3。
- `embed_dim=768`: 嵌入向量的维度，即将每个小块映射到的向量空间的维度。
- `norm_layer=None`: 用于嵌入向量之后进行归一化的层，如果传入`None`，则不进行**归一化**，使用`nn.Identity()`作为占位符。

​	在初始化方法中，首先确保`img_size`和`patch_size`是**元组**形式，表示宽度和高度。然后计算网格大小（即图像被分割成多少行和列的小块），以及总的小块数量。接着，使用二维卷积`nn.Conv2d`来执行投影操作，将每个小块映射到指定的嵌入维度。如果提供了归一化层，则对嵌入后的向量进行归一化。

**前向传播方法 `forward`**

- 输入`x`的形状是`[B, C, H, W]`，其中`B`是批次大小，`C`是通道数，`H`和`W`分别是图像的高度和宽度。
- 首先检查输入图像的大小是否与模型预期的`img_size`相匹配。
- 使用`self.proj`（即二维卷积）对输入图像进行投影，然后使用`flatten(2)`将每个小块的嵌入向量从形状`[B, embed_dim, H//patch_size, W//patch_size]`扁平化为`[B, embed_dim, (H//patch_size)*(W//patch_size)]`，最后使用`transpose(1, 2)`将其转换为`[B, (H//patch_size)*(W//patch_size), embed_dim]`，以便每个小块的嵌入向量成为一行。
- 应用归一化层（如果有的话）。
- 返回处理后的嵌入向量。

​	总的来说，`PatchEmbed`类**将输入的2D图像转换为一个序列的嵌入向量**，这些向量可以进一步用于Transformer模型中的**自注意力机制**等处理。



```python
class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

## Attention

​	这段代码定义了一个名为`Attention`的类，它继承自`nn.Module`，是PyTorch深度学习框架中用于**构建神经网络模型**的基础类。`Attention`类实现了一个**多头自注意力**（Multi-Head Self-Attention）机制，这是Transformer架构的核心组件之一。下面是对这个类的详细解释：

**初始化方法 `__init__`**

- `dim`: 输入**token的维度**。
- `num_heads`: 多头注意力的头数。
- `qkv_bias`: 在qkv线性变换中**是否添加偏置项**。
- `qk_scale`: 查询（query）和键（key）的**缩放因子**，用于调节点积注意力分数的尺度。
- `attn_drop_ratio`: 注意力分数上的**dropout比率**。
- `proj_drop_ratio`: 投影后的dropout比率。

​	在初始化方法中，首先计算了**每个头的维度`head_dim`**（假设`dim`能被`num_heads`整除），然后设置了**缩放因子`self.scale`**（如果未指定，则使用`head_dim`的负平方根）。接着，定义了**qkv线性变换**（`self.qkv`），它将输入映射到三倍于输入维度的空间，以分别获取查询（query）、键（key）和值（value）的表示。还定义了**注意力分数上的dropout**（`self.attn_drop`）和**投影后的dropout**（`self.proj_drop`），以及一个用于最终投影的线性层（`self.proj`）。

**前向传播方法 `forward`**

- `x`: 输入数据，形状为`[batch_size, num_patches + 1, total_embed_dim]`，其中`num_patches + 1`通常表示图像**被分割成的小块数量加上一个类token**（class token）。

​	在前向传播方法中，首先**通过qkv线性变换得到查询、键和值的表示**，并将它们重塑和置换维度以匹配多头注意力的要求。然后，计算查询和键的点积注意力分数，应用缩放因子，并**通过softmax函数得到归一化的注意力权重**。接着，应用注意力dropout，并将**注意力权重与值相乘得到加权后的值**。最后，将这些加权后的值重塑回原始维度，通过**投影线性层**，并**应用投影dropout**，得到最终的输出。

**注意事项**

- 代码中使用了`qkv[0]`, `qkv[1]`, `qkv[2]`来分别获取查询、键和值的表示，这是为了让代码兼容TorchScript（PyTorch的一个子集，用于模型的序列化和部署）。在纯PyTorch代码中，可以直接使用元组解包来获取这些值。
- 注意力机制中的缩放因子`self.scale`是为了防止点积结果过大，导致softmax函数进入饱和区，从而影响梯度传播。
- 投影后的dropout层（`self.proj_drop`）是可选的，但在一些模型中用于正则化，防止过拟合。



```
class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

## Mlp

​	这段代码定义了一个名为`Mlp`的类，它是`nn.Module`的子类，用于**实现多层感知机（MLP）结构**，这种结构在Vision Transformer、MLP-Mixer以及相关的网络中得到了广泛应用。下面是对这个类的详细解释：

**初始化方法 `__init__`**

- `in_features`: 输入特征的维度。
- `hidden_features=None`: 隐藏层特征的维度。如果未指定，则默认与输入特征的维度相同。
- `out_features=None`: 输出特征的维度。如果未指定，则默认与输入特征的维度相同。
- `act_layer=nn.GELU`: 激活层，默认使用高斯误差线性单元（GELU）。
- `drop=0.`: Dropout比率，用于防止过拟合。默认是不进行dropout。

​	在初始化方法中，首先根据**输入参数设置了输出特征维度**`out_features`和**隐藏层特征维度**`hidden_features`的默认值（如果未指定的话）。然后，定义了**第一个全连接层**`self.fc1`，它将输入特征从`in_features`维映射到`hidden_features`维。接着，定义了激活层`self.act`，它是一个GELU激活函数（或者根据`act_layer`参数指定的其他激活函数）。之后，定义了第二个全连接层`self.fc2`，它将隐藏层特征从`hidden_features`维映射到`out_features`维。最后，定义了一个dropout层`self.drop`，用于在训练过程中随机丢弃一部分神经元的输出，以防止模型过拟合。

**前向传播方法 `forward`**

- 输入`x`是**待处理的数据**。

​	在前向传播方法中，数据首先通过**第一个全连接层**`self.fc1`，然后经过激活层`self.act`进行**非线性变换**，接着通过dropout层`self.drop`进行**随机丢弃**（仅在训练过程中生效）。之后，数据再通过第二个全连接层`self.fc2`，最后再次通过dropout层`self.drop`得到最终的输出。注意，这里的dropout层在第二次使用时可能会对输出特征进行进一步的随机丢弃，但这也取决于`drop`参数的值以及模型是否处于训练模式。

​	这个类实现的是一个简单的**两层MLP结构**，它可以被用于构建更复杂的网络模型，如Vision Transformer和MLP-Mixer等。在这些模型中，MLP通常用于**处理序列或图像块嵌入后的特征表示**，以捕捉**特征之间的非线性关系**。



```
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

## Block

​	这段代码定义了一个名为`Block`的类，它继承自`nn.Module`，是构建Transformer或类似架构中的基本块。这个基本块包含了归一化层、注意力机制层、随机深度丢弃层（或恒等操作层）、以及一个多层感知机（MLP）。下面是对这个类的详细解释：

**初始化方法 `__init__`**

- `dim`: 输入/输出的维度。
- `num_heads`: 注意力机制中头（head）的数量。
- `mlp_ratio=4.`: MLP中隐藏层维度与输入/输出维度的比例。
- `qkv_bias=False`: 在查询（Q）、键（K）、值（V）的线性变换中是否添加偏置项。
- `qk_scale=None`: 用于缩放查询和键的点积结果的缩放因子。
- `drop_ratio=0.`: MLP和注意力机制后的丢弃比率。
- `attn_drop_ratio=0.`: 注意力权重上的丢弃比率。
- `drop_path_ratio=0.`: 随机深度丢弃比率，用于正则化。
- `act_layer=nn.GELU`: MLP中的激活函数层。
- `norm_layer=nn.LayerNorm`: 归一化层，默认使用层归一化。

​	在初始化方法中，首先定义了两个归一化层`self.norm1`和`self.norm2`，它们分别位于注意力机制和MLP之前。接着，定义了一个注意力机制层`self.attn`，它包含了多头注意力机制以及可能的丢弃层。如果`drop_path_ratio`大于0，则定义一个随机深度丢弃层`self.drop_path`，否则定义一个恒等操作层（即不改变输入）。然后，根据`mlp_ratio`计算出MLP中隐藏层的维度，并定义一个MLP层`self.mlp`。

**前向传播方法 `forward`**

- 输入`x`是待处理的数据。

​	在前向传播方法中，数据首先经过第一个归一化层`self.norm1`，然后送入注意力机制层`self.attn`。注意力机制的输出经过随机深度丢弃层（或恒等操作层）后，与原始输入`x`进行残差连接。接着，数据经过第二个归一化层`self.norm2`，然后送入MLP层`self.mlp`。MLP的输出同样经过随机深度丢弃层（或恒等操作层）后，再次与之前的残差连接结果进行相加，得到最终的输出。

​	这种结构允许网络**通过残差连接学习输入和输出之间的差异**，同时通过**归一化层和丢弃层**来提高模型的泛化能力。随机深度丢弃层是一种正则化技术，它通过随机丢弃一部分路径（即一部分层的输出）来防止模型过拟合。在这个类中，**随机深度丢弃层是可选的**，取决于`drop_path_ratio`的值。



```
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_c (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            
Args：
img_size（int，tuple）：输入图像大小
patch_size（int，tuple）：补丁大小
inc（int）：输入通道数
num_classes（int）：分类头的类数
embed_dim（int）：嵌入维度
depth（int）：变压器的深度
num_heads（int）：关注头的数量
mlp_aratio（int）：mlp隐藏dim与嵌入dim的比率
qkv_bias（bool）：如果为True，则启用qkv的偏置
qk_scale（float）：覆盖head_dim**的默认qk scale，如果设置为-0.5
representation_size（可选[int]）：启用表示层（pre-logits）并将其设置为该值（如果已设置）
districted（bool）：与DeiT模型一样，模型包括一个蒸馏令牌和头部
drop_ratio（浮动）：辍学率
attn_drop_ratio（float）：注意力丢失率
drop_path_ratio（float）：随机深度率
embed_layer（nn.Module）：补丁嵌入层
norm_layer：（nn.Module）：规范化层

        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer	(表示层)
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)	(分级机机头)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init	(重量初始化)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x
```

## VisionTransformer

​	这段代码定义了一个名为`VisionTransformer`（ViT）的类，它继承自`nn.Module`，并实现了Vision Transformer模型的结构。Vision Transformer是一种基于Transformer架构的**图像分类模型**，它将图像分割成一系列小块（patches），然后将这些小块作为序列输入到Transformer中进行处理。以下是对这个类的详细解释：

**初始化方法 `__init__`**

- `img_size`: 输入图像的尺寸。
- `patch_size`: 图像被分割成的小块（patch）的尺寸。
- `in_c`: 输入图像的通道数。
- `num_classes`: 分类头的输出类别数。
- `embed_dim`: 嵌入维度，即Transformer中每个token的维度。
- `depth`: Transformer的层数（即Block的数量）。
- `num_heads`: 多头注意力机制中头的数量。
- `mlp_ratio`: MLP中隐藏层维度与嵌入维度的比例。
- `qkv_bias`: 在查询（Q）、键（K）、值（V）的线性变换中是否添加偏置项。
- `qk_scale`: 用于缩放查询和键的点积结果的缩放因子。
- `representation_size`: 表示层的维度（如果设置），用于预logits层。
- `distilled`: 是否为蒸馏模型，即是否包含蒸馏token和头（如DeiT模型）。
- `drop_ratio`: dropout比率。
- `attn_drop_ratio`: 注意力机制上的dropout比率。
- `drop_path_ratio`: 随机深度丢弃比率。
- `embed_layer`: 嵌入层，默认为`PatchEmbed`，用于将图像分割成小块并嵌入到向量空间中。
- `norm_layer`: 归一化层，默认为`nn.LayerNorm`。
- `act_layer`: 激活函数层，默认为`nn.GELU`。

​	在初始化方法中，首先设置了模型的一些基本属性，如类别数、嵌入维度等。然后定义了嵌入层`self.patch_embed`，用于将输入图像分割成小块并嵌入到向量空间中。接着定义了类token（`self.cls_token`）和蒸馏token（如果`distilled`为True，则`self.dist_token`），以及位置嵌入（`self.pos_embed`）。还定义了位置dropout层`self.pos_drop`。

​	接下来，根据`depth`创建了多个`Block`实例，并将它们顺序地添加到`self.blocks`中。每个`Block`都包含了Transformer的一个层，包括归一化层、注意力机制层、MLP层以及可能的随机深度丢弃层。

​	如果设置了`representation_size`，则创建一个表示层`self.pre_logits`，用于在分类头之前对特征进行进一步的处理。最后，定义了分类头`self.head`，如果`distilled`为True，则还定义一个蒸馏头`self.head_dist`。

​	在初始化方法的最后部分，对**位置嵌入、类token和蒸馏token（如果存在）进行了权重初始化**，并应用了自定义的权重初始化函数`_init_vit_weights`（尽管这个函数在代码片段中没有给出）。

**前向传播方法 `forward` 和 `forward_features`**

- `forward_features(x)`: 这个方法计算模型的前向传播，但不包括最终的分类头。它首先通过嵌入层将输入图像**分割成小块**并嵌入到向量空间中，然后添加类token（和蒸馏token，如果存在），接着添加位置嵌入并通过位置dropout层。之后，数据通过Transformer的多个层（即`self.blocks`），最后**通过归一化层**。如果设置了表示层，则还对其进行处理。如果模型是蒸馏模型，则返回类token和蒸馏token的表示；否则，只返回类token的表示。
- `forward(x)`: 这个方法是模型的主要前向传播方法。它首先调用`forward_features`方法获取表示，然后根据模型是否为蒸馏模型以及是否处于训练模式，决定是返回两个分类头的输出（在训练模式下）还是它们的平均值（在推理模式下）。如果模型不是蒸馏模型，则只返回分类头的输出。

​	这个Vision Transformer类提供了一个灵活的框架，可以通过调整各种参数来构建不同规模和复杂度的模型，以适应不同的图像分类任务。



```
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
```

## _init_vit_weights

​	这段代码定义了一个名为 `_init_vit_weights` 的函数，用于**初始化Vision Transformer（ViT）模型中的权重**。这个函数接收一个参数 `m`，代表模型中的一个模块（module）。根据模块的类型，它会采用不同的权重初始化方法。下面是对这个函数及其各个部分的解释：

1. **线性层（`nn.Linear`）的初始化**

   ​	如果模块是一个线性层（`nn.Linear`），则使用截断正态分布（`trunc_normal_`）初始化其权重，标准差设置为0.01。截断正态分布是一种正态分布，但在一定范围之外的值会被丢弃并重新抽取，这有助于避免极端值的出现。

   ​	如果线性层有偏置项（`bias`），则将偏置项初始化为0（`zeros_`）。

2. **卷积层（`nn.Conv2d`）的初始化**

   ​	如果模块是一个二维卷积层（`nn.Conv2d`），则使用Kaiming初始化（也称为He初始化，`kaiming_normal_`）来初始化其权重。Kaiming初始化考虑了前向传播和反向传播中激活值和梯度的方差，特别适用于ReLU激活函数。`mode="fan_out"`表示根据输出单元的数量来缩放权重。

   ​	如果卷积层有偏置项，同样将偏置项初始化为0。

3. **层归一化（`nn.LayerNorm`）的初始化**

   ​	如果模块是层归一化（`nn.LayerNorm`），则将偏置项初始化为0，权重初始化为1（`ones_`）。层归一化是一种用于改善神经网络训练稳定性的技术，通过对层的输入进行归一化来实现。

​	这个函数的设计目的是为了在构建Vision Transformer模型时，能够**自动地为不同类型的层应用适当的权重初始化策略**。正确的权重初始化对于模型的训练效果和收敛速度至关重要，因为它可以帮助避免梯度消失或爆炸等问题。



```
def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model
```

## vit_base_patch16_224

​	这段代码定义了一个名为 `vit_base_patch16_224` 的函数，用于创建并返回一个Vision Transformer（ViT）基础模型配置，该配置遵循原始论文（https://arxiv.org/abs/2010.11929）中的描述。这个特定的模型配置被称为ViT-B/16，表示它是一个基础（Base）模型，使用16x16的图像块（patch）大小，并在ImageNet-1k数据集上以224x224的输入分辨率进行了预训练。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为1000，对应于ImageNet-1k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。

函数内部：

- 创建了一个`VisionTransformer`类的实例，该类应该在其他地方定义（这里没有给出）。这个实例被配置为具有特定的参数，这些参数定义了ViT模型的架构：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=16`: 将图像分割成的小块（patch）的尺寸。

  ​	`embed_dim=768`: 嵌入向量的维度，即每个图像块被转换成的向量的大小。

  ​	`depth=12`: Transformer编码器的层数。

  ​	`num_heads=12`: 在多头自注意力机制中使用的头的数量。

  ​	`representation_size=None`: 表示大小，如果设置，则会在分类头之前从Transformer输出中截取一个特定大小的表示。在这个配置中，它被设置为`None`，意味着不使用这个特性（或者使用默认值，取决于`VisionTransformer`类的实现）。

  ​	`num_classes=num_classes`: 分类头的输出类别数，根据函数参数设置。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接和密码是用于下载预训练权重的，但这通常不是在实际代码库或生产环境中分享权重的方式。在实际应用中，权重通常会通过更安全的途径（如模型库、云存储等）提供，并且不会包含密码。这里的链接和密码可能是为了示例或演示目的而提供的，不应在实际项目中使用。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。



```
def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    
    原始论文中的ViT-Base模型（ViT-B/16）(https://arxiv.org/abs/2010.11929).
ImageNet-21k权重@224x224，源https://github.com/google-research/vision_transformer.
权重从官方的Google JAX-inmpl移植：
https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model
```

## vit_base_patch16_224_in21k

​	这段代码定义了一个名为 `vit_base_patch16_224_in21k` 的函数，用于创建并返回一个Vision Transformer（ViT）基础模型配置，该配置与原始论文（https://arxiv.org/abs/2010.11929）中的描述相符。这个特定的模型配置是在ImageNet-21k数据集上进行了预训练，该数据集包含约21,843个类别，并且输入图像的分辨率为224x224。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为21843，对应于ImageNet-21k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。
- `has_logits`: 一个布尔值，指定模型是否应该包含一个用于分类的logits层。如果设置为`True`（默认值），则`representation_size`将被设置为`768`（与`embed_dim`相同），这意味着在Transformer编码器的输出之后将有一个额外的线性层来产生logits。如果设置为`False`，则模型将不会在Transformer输出之后添加logits层，而是直接返回Transformer的最终输出作为表示。

函数内部：

- 创建了一个`VisionTransformer`类的实例，该类应该在其他地方定义（这里没有给出）。这个实例被配置为具有特定的参数，这些参数定义了ViT模型的架构：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=16`: 将图像分割成的小块（patch）的尺寸。

  ​	`embed_dim=768`: 嵌入向量的维度。

  ​	`depth=12`: Transformer编码器的层数。

  ​	`num_heads=12`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=768 if has_logits else None`: 表示大小。如果`has_logits`为`True`，则设置与`embed_dim`相同的值，以便在Transformer输出之后添加logits层。如果为`False`，则不添加logits层，并可能返回Transformer的最终输出作为特征表示。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接是用于下载在ImageNet-21k上预训练的权重的。在实际应用中，你需要确保这个链接是有效的，并且你有权限访问这些权重。通常，这些权重文件会通过更正式的渠道（如模型库、云存储等）提供。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。此外，由于这里提到了特定的权重文件，你可能还需要在代码中添加加载这些权重的逻辑（尽管这部分代码没有在这里给出）。



```
def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model
```

## vit_base_patch32_224

​	这段代码定义了一个名为 `vit_base_patch32_224` 的函数，用于创建并返回一个Vision Transformer（ViT）基础模型配置，但这次使用的是32x32的图像块（patch）大小，而不是之前提到的16x16。这个特定的模型配置仍然遵循原始论文（https://arxiv.org/abs/2010.11929）中的描述，并在ImageNet-1k数据集上以224x224的输入分辨率进行了预训练。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为1000，对应于ImageNet-1k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。

函数内部：

- 创建了一个`VisionTransformer`类的实例，并配置了以下参数：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=32`: 将图像分割成的小块（patch）的尺寸。这是与之前的配置不同的地方，之前使用的是16x16的patch大小。

  ​	`embed_dim=768`: 嵌入向量的维度。

  ​	`depth=12`: Transformer编码器的层数。

  ​	`num_heads=12`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=None`: 表示大小。这里设置为`None`，意味着不使用特定的表示大小，或者将使用`VisionTransformer`类的默认值（如果有的话）。在实际应用中，这通常意味着Transformer编码器的输出将直接传递给分类头（如果存在的话）。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接和密码是用于下载预训练权重的，但这通常不是在实际代码库或生产环境中分享权重的方式。在实际应用中，权重通常会通过更安全的途径（如模型库、云存储等）提供，并且不会包含密码。这里的链接和密码可能是为了示例或演示目的而提供的，不应在实际项目中使用。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。此外，由于这里提到了特定的权重文件，你可能还需要在代码中添加加载这些权重的逻辑（尽管这部分代码没有在这里给出）。在实际应用中，你应该从可靠的来源获取预训练权重，并确保它们与你的模型架构相匹配。



```
def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model
```

## vit_base_patch32_224_in21k

​	这段代码定义了一个名为 `vit_base_patch32_224_in21k` 的函数，用于创建并返回一个Vision Transformer（ViT）基础模型配置。这个配置遵循原始论文（https://arxiv.org/abs/2010.11929）中的描述，但使用32x32的图像块（patch）大小，并在ImageNet-21k数据集上以224x224的输入分辨率进行了预训练。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为21843，对应于ImageNet-21k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。
- `has_logits`: 一个布尔值，指定模型是否应该包含一个用于分类的logits层。如果设置为`True`（默认值），则模型将包含一个logits层，该层将Transformer编码器的输出转换为类别分数。如果设置为`False`，则模型将不会包含logits层，而是直接返回Transformer的最终输出作为特征表示。

函数内部：

- 创建了一个`VisionTransformer`类的实例，并配置了以下参数：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=32`: 将图像分割成的小块（patch）的尺寸。

  ​	`embed_dim=768`: 嵌入向量的维度。

  ​	`depth=12`: Transformer编码器的层数。

  ​	`num_heads=12`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=768 if has_logits else None`: 表示大小。如果`has_logits`为`True`，则设置与`embed_dim`相同的值，以便在Transformer输出之后添加logits层。如果为`False`，则不添加logits层。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接是用于下载在ImageNet-21k上预训练的权重的。在实际应用中，你需要确保这个链接是有效的，并且你有权限访问这些权重。通常，这些权重文件会通过模型库、云存储或其他正式渠道提供。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。此外，由于这里提到了特定的权重文件，你可能还需要在代码中添加加载这些权重的逻辑。这通常涉及使用类似`torch.load`的函数来加载权重文件，并将其应用到模型实例上。
- 代码中未包含加载权重的具体实现，因为这通常取决于你的项目结构和需求。在实际项目中，你应该确保在模型实例化后正确加载预训练权重。



```
def vit_large_patch16_224(num_classes: int = 1000):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=None,
                              num_classes=num_classes)
    return model
```

## vit_large_patch16_224

​	这段代码定义了一个名为 `vit_large_patch16_224` 的函数，用于创建并返回一个Vision Transformer（ViT）大型模型配置。这个配置遵循原始论文（https://arxiv.org/abs/2010.11929）中的描述，使用16x16的图像块（patch）大小，并在ImageNet-1k数据集上以224x224的输入分辨率进行了预训练。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为1000，对应于ImageNet-1k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。

函数内部：

- 创建了一个`VisionTransformer`类的实例，并配置了以下参数：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=16`: 将图像分割成的小块（patch）的尺寸。这是与函数名中提到的“patch16”相对应的。

  ​	`embed_dim=1024`: 嵌入向量的维度。对于ViT-Large模型，这通常比ViT-Base模型（768维）要大。

  ​	`depth=24`: Transformer编码器的层数。对于ViT-Large模型，这通常比ViT-Base模型（12层）要深。

  ​	`num_heads=16`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=None`: 表示大小。这里设置为`None`，意味着不使用特定的表示大小，或者将使用`VisionTransformer`类的默认值（如果有的话）。在实际应用中，这通常意味着Transformer编码器的输出将直接传递给分类头（如果存在的话）。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接和密码是用于下载预训练权重的，但这通常不是在实际代码库或生产环境中分享权重的方式。在实际应用中，权重通常会通过更安全的途径（如模型库、云存储等）提供，并且不会包含密码。这里的链接和密码可能是为了示例或演示目的而提供的，不应在实际项目中使用。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。
- 由于这里提到了特定的权重文件，你可能还需要在代码中添加加载这些权重的逻辑（尽管这部分代码没有在这里给出）。在实际应用中，你应该从可靠的来源获取预训练权重，并确保它们与你的模型架构相匹配。加载权重的代码通常涉及使用类似`torch.load`的函数，并可能需要处理权重文件中的特定格式或结构。
- 函数名中的“vit_large_patch16_224”清晰地表明了模型的配置：大型（Large）ViT模型，使用16x16的patch大小，并在224x224的图像尺寸上进行训练和评估。



```
def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model
```

## vit_large_patch16_224_in21k

​	这段代码定义了一个名为 `vit_large_patch16_224_in21k` 的函数，用于创建并返回一个Vision Transformer（ViT）大型模型配置，该配置在ImageNet-21k数据集上进行了预训练。这个函数遵循了原始论文（https://arxiv.org/abs/2010.11929）中的描述，并使用了特定的参数设置。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为21843，对应于ImageNet-21k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。
- `has_logits`: 一个布尔值，指定模型是否应该包含一个用于分类的logits层。如果设置为`True`（默认值），则模型将包含一个logits层，该层将Transformer编码器的输出转换为类别分数。如果设置为`False`，则模型将不会包含logits层，而是直接返回Transformer的最终输出作为特征表示。

函数内部：

- 创建了一个`VisionTransformer`类的实例，并配置了以下参数：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=16`: 将图像分割成的小块（patch）的尺寸。

  ​	`embed_dim=1024`: 嵌入向量的维度。对于ViT-Large模型，这是标准的维度设置。

  ​	`depth=24`: Transformer编码器的层数。对于ViT-Large模型，这是标准的深度设置。

  ​	`num_heads=16`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=1024 if has_logits else None`: 表示大小。如果`has_logits`为`True`，则设置与`embed_dim`相同的值，以便在Transformer输出之后添加logits层。如果为`False`，则不添加logits层，并且此参数为`None`。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接是用于下载在ImageNet-21k上预训练的权重的。在实际应用中，你需要确保这个链接是有效的，并且你有权限访问这些权重。通常，这些权重文件会通过模型库、云存储或其他正式渠道提供。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。
- 由于这里提到了特定的权重文件，你可能还需要在代码中添加加载这些权重的逻辑（尽管这部分代码没有在这里给出）。在实际应用中，你应该使用类似`torch.load`的函数来加载权重文件，并将其应用到模型实例上。加载权重的代码通常涉及指定权重文件的路径，并可能需要处理权重文件中的特定格式或结构。
- 函数名中的“vit_large_patch16_224_in21k”清晰地表明了模型的配置：大型（Large）ViT模型，使用16x16的patch大小，在224x224的图像尺寸上进行训练和评估，并且使用了ImageNet-21k数据集的预训练权重。



```
def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)
    return model
```

## vit_large_patch32_224_in21k

​	这段代码定义了一个名为 `vit_large_patch32_224_in21k` 的函数，用于创建并返回一个Vision Transformer（ViT）大型模型配置，该配置在ImageNet-21k数据集上进行了预训练，并且使用了32x32的图像块（patch）大小。这个函数遵循了原始论文（https://arxiv.org/abs/2010.11929）中的描述，并使用了特定的参数设置。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为21843，对应于ImageNet-21k数据集的类别数。如果你打算在另一个具有不同类别数的数据集上使用此模型，你需要相应地更改这个参数。
- `has_logits`: 一个布尔值，指定模型是否应该包含一个用于分类的logits层。如果设置为`True`（默认值），则模型将包含一个logits层，该层将Transformer编码器的输出转换为类别分数。如果设置为`False`，则模型将不会包含logits层，而是直接返回Transformer的最终输出作为特征表示。

函数内部：

- 创建了一个`VisionTransformer`类的实例，并配置了以下参数：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=32`: 将图像分割成的小块（patch）的尺寸。这是与函数名中的“patch32”相对应的。

  ​	`embed_dim=1024`: 嵌入向量的维度。对于ViT-Large模型，这是标准的维度设置。

  ​	`depth=24`: Transformer编码器的层数。对于ViT-Large模型，这是标准的深度设置。

  ​	`num_heads=16`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=1024 if has_logits else None`: 表示大小。如果`has_logits`为`True`，则设置与`embed_dim`相同的值，以便在Transformer输出之后添加logits层。如果为`False`，则不添加logits层，并且此参数为`None`。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 代码中提到的权重链接是用于下载在ImageNet-21k上预训练的权重的，并且这些权重是针对32x32的patch大小进行训练的。在实际应用中，你需要确保这个链接是有效的，并且你有权限访问这些权重。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。
- 由于这里提到了特定的权重文件，你可能还需要在代码中添加加载这些权重的逻辑（尽管这部分代码没有在这里给出）。在实际应用中，你应该使用类似`torch.load`的函数来加载权重文件，并将其应用到模型实例上。加载权重的代码通常涉及指定权重文件的路径，并可能需要处理权重文件中的特定格式或结构。
- 函数名中的“vit_large_patch32_224_in21k”清晰地表明了模型的配置：大型（Large）ViT模型，使用32x32的patch大小，在224x224的图像尺寸上进行训练和评估，并且使用了ImageNet-21k数据集的预训练权重。



```
def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(img_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              depth=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)
    return model
```

## vit_huge_patch14_224_in21k

​	这段代码定义了一个名为 `vit_huge_patch14_224_in21k` 的函数，用于创建并返回一个Vision Transformer（ViT）巨型（Huge）模型配置，该配置在ImageNet-21k数据集上进行了预训练，并且使用了14x14的图像块（patch）大小。这个函数遵循了原始论文（https://arxiv.org/abs/2010.11929）中的描述，但需要注意的是，这里提到的预训练权重由于文件过大，目前并不在GitHub上公开提供。

函数参数：

- `num_classes`: 一个整数，指定模型输出层的类别数。默认为21843，对应于ImageNet-21k数据集的类别数。
- `has_logits`: 一个布尔值，指定模型是否应该包含一个用于分类的logits层。如果设置为`True`（默认值），则模型将包含一个logits层；如果设置为`False`，则模型将不包含logits层。

函数内部：

- 创建了一个`VisionTransformer`类的实例，并配置了以下参数：

  ​	`img_size=224`: 输入图像的尺寸。

  ​	`patch_size=14`: 将图像分割成的小块（patch）的尺寸。这是与函数名中的“patch14”相对应的。

  ​	`embed_dim=1280`: 嵌入向量的维度。对于ViT-Huge模型，这是比ViT-Large模型更高的维度设置。

  ​	`depth=32`: Transformer编码器的层数。对于ViT-Huge模型，这是比ViT-Large模型更深的设置。

  ​	`num_heads=16`: 多头自注意力机制中使用的头的数量。

  ​	`representation_size=1280 if has_logits else None`: 表示大小。如果`has_logits`为`True`，则设置与`embed_dim`相同的值，以便在Transformer输出之后添加logits层；如果为`False`，则不添加logits层。

  ​	`num_classes=num_classes`: 分类头的输出类别数。

返回值：

- 函数返回一个配置好的 `VisionTransformer` 模型实例。

注意：

- 由于ViT-Huge模型的权重文件非常大，因此目前并不在GitHub上公开提供。如果你需要使用这些权重，你可能需要联系作者或查找其他途径来获取它们。
- 要使这段代码正常工作，你需要有 `VisionTransformer` 类的实现，以及任何必要的依赖项（如PyTorch库）。
- 在实际应用中，如果你打算使用预训练的权重，你需要确保有权访问这些权重，并且你的硬件（如GPU）有足够的内存来加载和运行这个大型模型。
- 函数名中的“vit_huge_patch14_224_in21k”清晰地表明了模型的配置：巨型（Huge）ViT模型，使用14x14的patch大小，在224x224的图像尺寸上进行训练和评估，并且原本计划使用ImageNet-21k数据集的预训练权重（尽管这些权重目前并不公开提供）。