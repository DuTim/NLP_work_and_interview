# 位置编码

## ref:  https://www.bilibili.com/video/BV1rK4y1X7xd/

ref: [Meta最新模型LLaMA细节与代码详解_常鸿宇的博客-CSDN博客](https://blog.csdn.net/weixin_44826203/article/details/129255185)

![](assets/img/2023-08-20-19-58-29-image.png)

RoPE（Rotary Position Embedding）旋转位置编码，是苏剑林老师提出的一种旋转位置编码方法，其思想是采用绝对位置编码的形式，实现相对位置编码。

而RoPE的巧妙之处在于，它既保留了绝对位置编码中的绝对位置信息，又保留了在内积运算下，对位置信息的相对性。

<img title="" src="assets/img/2023-08-20-20-04-34-image.png" alt="" width="1237">

<img title="" src="assets/img/2023-08-20-20-04-54-image.png" alt="" width="1237">
