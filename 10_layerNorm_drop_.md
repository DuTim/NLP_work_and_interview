# 1 pre_norm  and post_norm

[Meta最新模型LLaMA细节与代码详解_常鸿宇的博客-CSDN博客](https://blog.csdn.net/weixin_44826203/article/details/129255185)

[pre_norm and post norm ref:](https://zhuanlan.zhihu.com/p/494661681)

<img src="assets/img/2023-07-29-22-07-52-image.png" title="" alt="" width="324">

<img title="" src="assets/img/2023-07-29-19-51-46-image.png" alt="" width="328">

<img title="" src="assets/img/2023-07-29-19-52-06-image.png" alt="" width="331">

同一设置之下，Pre Norm结构往往更容易训练，但最终效果通常不如Post Norm。Pre Norm更容易训练好理解，因为它的恒等路径更突出.

Pre Norm和Post Norm的式子分别如下：

$$
Pre Norm:  \quad \boldsymbol{x}_{t+1}=\boldsymbol{x}_{t}+F_{t}\left(\operatorname{Norm}\left(\boldsymbol{x}_{t}\right)\right) \\

Post Norm:  \quad \boldsymbol{x}_{t+1}=\operatorname{Norm}\left(\boldsymbol{x}_{t}+F_{t}\left(\boldsymbol{x}_{t}\right)\right) 
$$

在Transformer中，这里Norm的主要指Layer Normalization，但在一般的模型中，它也可以是Batch Normalization、Instance Normalization等，相关结论本质上是通用的。

显示Post Norm优于Pre Norm的工作有两篇，一篇是**[《Understanding the Difficulty of Training Transformers》](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2004.08249)**，一篇是**[《RealFormer: Transformer Likes Residual Attention》](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2012.11747)。

Post Norm的结构迁移性能更加好，也就是说在Pretraining中，Pre Norm和Post Norm都能做到大致相同的结果，但是Post Norm的Finetune效果明显更好。

#### 原因

pre Norm的深度有“水分”！也就是说，一个层的Pre Norm模型，其实际等效层数不如层的Post Norm模型，而层数少了导致效果变差了。

$$
\begin{aligned}
\boldsymbol{x}_{t+1} & =\boldsymbol{x}_{t}+F_{t}\left(\operatorname{Norm}\left(\boldsymbol{x}_{t}\right)\right) \\
& =\boldsymbol{x}_{t-1}+F_{t-1}\left(\operatorname{Norm}\left(\boldsymbol{x}_{t-1}\right)\right)+F_{t}\left(\operatorname{Norm}\left(\boldsymbol{x}_{t}\right)\right) \\
& =\cdots \\
& =\boldsymbol{x}_{0}+F_{0}\left(\left(\operatorname{Norm}\left(\boldsymbol{x}_{0}\right)\right)+\cdots+F_{t-1}\left(\operatorname{Norm}\left(\boldsymbol{x}_{t-1}\right)\right)+F_{t}\left(\operatorname{Norm}\left(\boldsymbol{x}_{t}\right)\right)\right.
\end{aligned}
$$

其中<mark>每一项都是同一量级</mark>的，那么有$x_{t+1} = \theta(x_{t})$，也就是说第t+1层跟第t层的差别就相当于t+1与t的差别(ep: 19 与18的差别)，当较大时，两者的相对差别是很小的，因此

$$
F_{t}({Norm}\left(\boldsymbol{x}_{t}\right))+F_{t+1}({Norm}\left(\boldsymbol{x}_{t+1}\right))\\
 \approx F_{t}({Norm}\left(\boldsymbol{x}_{t}\right))+F_{t}({Norm}\left(\boldsymbol{x}_{t}\right))\\
= (F_t\oplus F_t )(F_{t}({Norm}\left(\boldsymbol{x}_{t}\right))
$$

这个意思是说，当比t较大时，$x_t$与$x_{t+1}$相差较小，所以$F_{t}({Norm}\left(\boldsymbol{x}_{t}\right))$与$F_{t+1}({Norm}\left(\boldsymbol{x}_{t+1}\right))$很接近，因此原本一个t层的模型与t+1层的和，近似等效于一个更宽的层模型，所以在Pre Norm中多层叠加的结果更多是增加宽度而不是深度，层数越多，这个层就越“虚”。

说白了，Pre Norm结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。而Post Norm刚刚相反，在**[《浅谈Transformer的初始化、参数化与标准化》](https://link.zhihu.com/?target=https%3A//kexue.fm/archives/8620)**中我们就分析过，它每Norm一次就削弱一次恒等分支的权重，所以Post Norm反而是更突出残差分支的，因此Post Norm中的层数更加“足秤”，一旦训练好之后效果更优。



post-norm和pre-norm其实各有优势，post-norm在残差之后做归一化，对参数正则化的效果更强，进而模型的鲁棒性也会更好；pre-norm相对于post-norm，因为有一部分参数直接加在了后面，不需要对这部分参数进行正则化，正好可以防止模型的梯度爆炸或者梯度消失，因此，这里笔者可以得出的一个结论是如果层数少post-norm的效果其实要好一些，如果要把层数加大，为了保证模型的训练，pre-norm显然更好一些。

### RMSNorm

<img src="assets/img/2023-08-20-19-45-50-image.png" title="" alt="" width="787">

RMSNorm是对LayerNorm的一个改进，没有做re-center操作（移除了其中的均值项），可以看作LayerNorm在均值为0时的一个特例。论文通过实验证明，re-center操作不重要。
RMSNorm 也是一种标准化方法，但与 LayerNorm 不同，它不是使用整个样本的均值和方差，而是使用平方根的均值来归一化，这样做可以降低噪声的影响。
<mark>这里的 ai与Layer Norm中的 x 等价，作者认为这种模式在简化了Layer Norm的同时，可以在各个模型上减少约 7%∼64% 的计算时间</mark>





### transformer 中关于norm使用的的例子:

1. 对主分支进行 dropout 

2. 再对post-norm

```python
class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x  ##残差保留
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x   ##残差保留
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x
```

#### pytorch 中实现 layernorm

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out
```

# 2. drop out

ref: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html?highlight=dropout#torch.nn.Dropout

<img src="assets/img/2023-07-29-22-10-13-image.png" title="" alt="" width="713">

# 1 batch norm 的顺序问题

## 核心 ref :  [(94条消息) batch normalize、relu、dropout 等的相对顺序_batchnormalization层、convolution层和relu层_littlepineapple的博客-CSDN博客](https://blog.csdn.net/dunlongzun8445/article/details/107612252)

应为 Batch Normalization 可以激活层提供期望的分布  所以: 因此<mark> Batch Normalization 层恰恰插入在 Conv 层或全连接层之后</mark>，而在 ReLU等激活层之前。而对于 dropout 则应当置于 activation layer 之后。

例如 通过BN 后 可以避免落入 sigmoid 的两端 梯度较小的区域

#### FC ==> BN ==> Relu/Sigmoid ==> dropout
