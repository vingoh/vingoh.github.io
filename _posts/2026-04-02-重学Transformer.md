---
layout:     post
title:      "重学Transformer"
subtitle:   "LLM界自己的Hello World"
date:       2026-04-02
author:     vingo
catalog:    true
tags:
  - Transformer
  - LLM
  - 学习笔记
---

## 前言
这篇主要根据[Hello-Agents第三章](https://datawhalechina.github.io/hello-agents/#/./chapter3/%E7%AC%AC%E4%B8%89%E7%AB%A0%20%E5%A4%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80)总结而来，也小小的参考了一下[HuggingFase Agent Course Unit1.2](https://huggingface.co/learn/agents-course/zh-CN/unit1/what-are-llms)。总体来说算是对LLM的一些补课，因为有深度学习的基础，所以太基础的部分就直接跳过了

## RNN和LTMS
其实前面还有一部分马尔科夫链的讲解，但是现在实在用的太少，就略过了，直接从RNN开始复习。

简单来说，RNN与传统神经网络最大的区别是通过**hidden state**形成所谓记忆，其将上一时刻的输出融合进当前时刻的输入，使得不同时刻或者说顺序的输入之间产生了联系，这恰恰符合了语言的特性。

然而，类似于ResNet出现之前的神经网络，RNN容易出现梯度爆炸/消失，导致长期记忆效果不好，为解决这一问题，诞生了LSTM。

LSTM在RNN基础上加入了information highway和遗忘/输入/输出门，来决定哪些信息被保留/遗弃或快速传递。

>写到这里真的想感叹一句，在5年前第一次在Introduction to Deep Learning课上接触到LSTM的时候，还算是时兴热门。现在不说日薄西山也可以说是查无此人了吧，AI发展的速度真的太快了。

## Transformer
噔噔咚！LLM界的Hello World，还是很多年前的I2DL课上，transformer刚刚问世，简单学过一下，今天在来仔细复习一遍。

### 整体结构
初始的transformer是一个宏观上的`encoder-decoder`结构，encoder最基础的一层是由`Multi-Head Attention + FFN`构成，decoder则是由`Masked Multi-Head Attention + Cross Attention + FFN`组成，同时每个子层后面跟一层`Add + Norm`。

在训练时，Encoder输入为训练集中的对话输入，Decoder输入则为右移一位的训练集中的对话结果，最终输出为概率，通过训练使得对话结果中的下一token获得最大预测概率。

推理时，Encoder输入不变，Decoder输入则为之前时刻的模型输出token拼接而成。

<div class="note insight">

💡 我的理解

这里的输入输出第一次看会有点难理解，因为输入实际上一个"输入-输出对"，或者叫"任务-结果对"会更好理解，而模型的最终输出是每个token的概率。比如对于一个翻译任务，encoder输入是"我爱你"，decoder对应的输入是右移一位后的翻译结果，即"I love”。此时模型应该尽量最大化you的概率。
    
因为输入里面还有一个输出，很多教程会把他和最终的概率输出统称为输出，造成困扰。这里后面把输入中的统称为任务和结果，以和最终输出进行区分。
</div>

与RNN一个很大的区别是，虽然推理时都是串行输出，但transformer在训练的是可以并行训练的，这也是得益于其attention机制。

### Attention

每个输入的句子，经过tokenizer变成一系列的token后，再转换为对应的embedding向量，以此作为初始的输入，[positional encoding](#Positional_Encoding)的事一会再说。

每个向量需要和三个内部权重不同但形状均为$d_model*d_model$的线性模型相乘，得到大名鼎鼎的QKV向量。文中这样解释：
>查询 (Query, Q)：代表当前词元，它正在主动地“查询”其他词元以获取信息。
键 (Key, K)：代表句子中可被查询的词元“标签”或“索引”。
值 (Value, V)：代表词元本身所携带的“内容”或“信息”。

<div class="note insight">

💡 我的理解
    
个人认为都更像先射箭再画靶的强行解释，可能这里只有三个矩阵，还勉强可以解释一下。
我的想法是：
Q是表示token要与外界查询交互的部分，类似类中的一个public查询接口
K则是每个token的对外展示，类似类的public变量
V是是从scroe转换回信息本身所用，所以承载token实际语义且不对外，类似类中的private变量
</div>

Attention中具体要做的则是以下几步：
* $Q * K^T$，相当于计算sequence中各token之间的关系度score，得分越高相关性越强，类似余弦距离
* $\frac{Q K^T}{\sqrt{d_K}}$，对score进行缩放，防止进入softmax的极限位置，导致梯度过小
* softmax，计算结果归一
* 结果乘以$V$，刚刚计算得到的所有结果可以视为一个`score'`，乘以V才真正得到了最终的信息。换句话说，不乘以V得到的只是代表相关性的分数，乘以V得到的才是最终的概率也即最终输出的词汇。

### Multi-Head Attention
简单来说就是把每个QKV向量分成`head_num`份，每份单独跑一遍刚才的流程再相加。这样可以让每一个head学到不同纬度的信息，有些教程中会距离说不同头在学习不同的语言部分，如名次/动词/情绪。
但实际上这种说法是十分不准确的，中attention中的高维度中，很难具体的说每个部分有着怎样的semantic的含义了。

注意，multi-head并不会增加参数，只是在原基础上做了分割。

### Masked Attention
由于模型是并行训练，不同时刻的sequence会被一同输入到模型中进行训练，可能会导致偷看之后的结果，影响训练效果。
举个例子，以一个翻译任务举例，输入"我爱你"得到"I love you"。但在训练中，不同任务与结果的pair会一起被输入，也就是模型会提前看到"I love you"的句子，这样后面在训练"I love"作为结果的这组数据的时候，就会产生偷跑。

为了解决这个问题，Masked Attention被提出。想法也非常简单粗暴，既然不想让你看到不应该看到的部分，那就不给你看。在Softmax之前，mask会将当前时刻之后对应的所有token设为负无穷，这样经过softmax对应位置的概率就接近0，也就不会关注后续的本不应看到的token。

### Position-wise Feed Forward Network
这是一个相对简单的网络，如其名，每个token都要单独输入进这个网络，使用共享权重得到结果。通常结构是d_model-d_ff-d_model，且通常d_ff > d_model。

好处是共享权重减少参数，扩大后收缩，学习高维知识。

### Add & Norm
在每个Attention与FF之后，都会接一组Add & Norm。其中，Add的效果和ResNet以及LSTM中的information highway作用一致，减少梯度消失。Norm也是常见操作，保持数据稳定性，避免发生漂移导致梯度爆炸/消失。

### Positional Encoding <a id="Positional_Encoding"></a>
基本的结构和组成部分都梳理完了，但还漏了一点。之前有说过，凭借attention机制，transformer可以并行训练，但这其实也带来了问题，就是忽略了词序。一组"我爱你"和另一组"你爱我" ~~和一组"蜜雪冰城甜蜜蜜"~~ 在transformer眼里是没有区别的，因为他只关注了句子里有什么，而忘记了每个token出现的位置。

Positional Encoding就是在给每个位置计算一个独一无二的位置向量，加入到embedding中，使得其蕴含位置信息。具体公式忽略不看，只需知道他是一个只与模型维度`d_model`，token位置`pos`以及位置向量位置索引`i`相关的值，因而当模型参数选定后，此矩阵就也随之固定，可以提前计算好而不包含可训练参数。

### Decoder Only
通常的解释是，encoder负责理解，decoder负责生成，而GPT开发团队觉得模型本身不需要理解，只需要预测就够了，所以直接拿掉了encoder的部分。

<div class="note question">

❓ 我的疑问
    
最初看到这里有个疑惑，之前相当于同时给任务和结果进行训练，现在只给结果，就算能预测出来，又怎么知道这个结果是对应什么任务的呢？

后来看了下论文发现，在decoder only中，任务与结果被拼在一起给了decoder。

我悟了。这不就是相当于prompt engineering。
    
</div>

### 一些参数
#### temperature
对softmax的`exp()`内的参数除以T，T越小曲线越陡峭，原本高概率的地方会变得更高，使得输出更为保守，适合理性工作，如数学推导/现实问题；相反则曲线平滑，各token的输出概率更接近，输出结果更多样，适合感性问题，如文学创作。

#### Top K
计算出概率后，取前K个最高的，归一化后按照各自概率随机抽选。

#### Top P
同样按概率排序，累加前n个的概率，直到超过P。

## 结尾
到这里基本上整个transformer的结构和各处细节就梳理完成了，这次就不手撕代码了，参数太多本地应该也跑不动。课程里后续还有一些prompting相关的内容，但后面章节还有agent更相关的类似内容，这里就看一下带过了。不过，既然说到参数多，不如最后推算一下transformer结构会有多少参数。

<div class="note insight">

💡 理解

首先看一下可训练部分都有哪些，以一个decoder-only为例，姑且跳过token-embedding计算：
* Masked-Attention QKV三个$d_{model} * d_{model}$，后面接一个总的线性$d_{model} * d_{model}$
* Position-wise Feed Forward是$2*d_{model} * d_{ff}$
* 以上每个后面加一个Norm，也就是$2*d_{model}$
    
通常使用$d_{ff}=4 * d_{model}$，总层数为$n$，那最终的结果就是
    $$n*(12*d_{model}^2+2*d_{model})\approx n*12*d_{model}^2$$

作为对比，一个encoder-decoder结构的则会多出一个与上面相同的结构，以及一个cross attention，也就是会多出$(12 + 4)*d_{model}^2$，会让模型参数量翻倍都不止，如此看来decoder only带来的降低参数数量的效果还是很惊人的。
    
</div>
