![Authors](https://github.com/errorplayer/AI_snippets/blob/master/pics/GetToThePointer_Pointer-Generator_Networks.JPG)

### 文本摘要的两种主流手段：  
1. Extractive -- 直接从原文摘取(一般是整句话)，相对容易，而且直接从原文选取句子，可以保证基本的语法正确性和观点正确性。  
2. Abstractive -- 会使用新词汇，更接近人类写的摘要，可能具备更高级的语言能力，比如转述/概括/整合现实知识等，这些是抽取式方法无法做到的。  

### 传统 neural seq2seq 模型在做文本摘要时存在的一些问题：  
1. 产生对事实错误的描述。按照我的理解，就是以往的模型并不理解他所描述的东西，只是简单的去猜下一个词而生成句子。这就很容易出现错误，或者说与原文不符的描述。  
2. 容易重复自己之前的话。因为传统的模型还没有特别强调去记忆之前概括过的内容，是否重复可能存在随机性。  
3. 出现OOV。Out-Of-Vocabulary: 测试文本出现了训练数据集中没有出现过的单词。传统的模型只能从基于训练数据的词库中找词，如果测试文本含有一些训练词库中没有的词，那么它可能会使用UNK来代替，造成不理想的结果。  

### 文中给出的解决思路  
1. Pointer -- 复制词：从输入文本中找词拿过来，这样做可以提高准确性，特别是能够避免OOV。  
2. Generator -- 生成词：从它的extended vocabulary里选一个词。  
3. Coverage machanism -- 记录 $t$ 时刻前的attention累和，用作Loss计算依据，控制模型生成时对输入文本的内容覆盖。  
4. Pointer和Generator在实现过程中是借助一个联动开关变量 $P_{gen}$ 来切换的, 最终产生词所使用的词库是 *extended vocabulary*， 也就是[训练词库 + 当前输入文本的词]。尽量避免了它“无词可用”的尴尬。   



`约定1: 下标i代表encoder输入token序号，下标j代表decoder输入token的序号。`  
`约定2: encoder输入文本长度是已知的，假设其长度为N；但decoder的预测长度未知。`  

### 1 baseline model  
也就是seq2seq2 with attention 模型。  
按次序把每一个输入文本token输进去，都会产生一个对应当前时刻及过去时刻的隐藏状态$h_i$，把最后一个token输入进去以后，产生的代表整个输入文本的隐藏状态$h_{t=N}$可以直接当作encoder对整个输入文本的一个编码表示，也可以经过某些变换，假设变换后把它记为$s_{t=N}$。    
接下来就是attention score的计算。这个score的计算方式可以有很多种，文中提到的是  
$$e_i^j=V*tanh(W_h h_i+W_s s_j+b_{attn})\tag{1}\label{1}$$  
> 这里注意的地方：   
> 0.当decoder输入第0个token时，使用的初始隐藏状态正是$s_{t=N}$。    
> 1.下标i 代表第几个encoder的输入token，范围是输入token个数，也就是输入文本长度。 $h_i$是固定的，因为每放入一个输入token就会产生一个对应位置的 $h_i$。  
> 2. $s_j$并非不变的，它是decoder每接收一个decoder的输入token产生的对应位置的$s_j$, $j$表示第几个decoder的输入token，decoder 没有输出 \<EOS\> 前，我们是不知道$j$最大能有多大的。比如现在给decoder输入第2个token，我们需要decoder预测第三个token，那么就需要带入$s_2$去公式$\ref{1}$算新的向量$e^3$， 记住$h_i$是固定的。   
> 3. 这里的$e_i$是标量，代表decoder输入第$j$个token预测第$j+1$个token时，对第$i$个encoder输入token的注意值。$e^j=(e_0, e_1, ..., e_N)$ 是一个向量，是decoder在做预测第$j+1$个token时对整个encoder输入序列的全局预览。  
> 4. 矩阵V的作用应该是维度映射。  


计算完attention score，就用一个$softmax$将其转成attention weights。      
$$a^j=softmax(e^j)\tag{2}$$  
用attention weights去weight sum$h_i$, 得到上下文向量$h_j^\*$。  
$$h_j^\*=\sum_i a_i^j h_i\tag{3}\label{3}$$  
公式$\ref{3}$算出来的上下文向量可以当作是decoder在当前这一timestep对整个输入序列的回顾，接下来把它和decoder前一时刻的隐藏状态拼接，再经过两个线性层即可得到预测词的概率分布$P_{vocab}$    
$$P_{vocab}=softmax(V_2(V_1(s_j,h_j^\*)+b_1)+b_2)\tag{4}\label{4}$$  
训练的话，用的是NLL(negative log likelihood)。  
$$loss_j = -logP_{vocab}(w_j)\tag{5}\label{5}$$   
$$loss=\frac {1}{T}\sum_{j=0}^{T}loss_j\tag{6}\label{6}$$  
其中T为最终输出序列长度。


### 2 Pointer-Generator Network  
作者介绍文中的pointer-generator network是baseline和[pointer network](https://arxiv.org/abs/1506.03134)的一个融合。在每次预测时先计算一个开关变量$p_{gen}$，入参分别有当前根据公式$\ref{3}$计算的上下文向量$h_j^\*$，前一时刻传过来的decoder隐藏状态$s_j$，当前时刻decoder输入token$x_j$。  
$$p_{gen}=\sigma(w_{h^\*}h_j^\*+w_s s_j+w_x x_j + b_{ptr})\tag{7}\label{7}$$   
于是产生的词概率分布计算公式由$\ref{4}$升级为$\ref{8}$：  
$$P(w)=p_{gen}P_{vocab}(w)+(1-p_{gen})\sum_{i:w_i = w}a_i^j\tag{8}\label{8}$$  
若$w\notin$训练词库，$p_{vocab}$自然为0；  
若$w\notin$输入文本，$\sum_{i:w_i = w}a_i^j$。  
> 笔者觉得：这里的$\sum_{i:w_i = w}a_i^j$是在第j次预测时，相同的encoder输入token($w_i$)的attention weight求和。比如输入文本为`我是东南大学的学生，东南大学是一所教学质量过硬的高校。`，这里假设令$w_i$为`东南大学`时，因为该token出现了两次，所以要把两个位置的attentiono weight相加。  

这样一来，预测下一个词的概率分布$p(w)$的w着眼于上文提到的 *extended vocabulary* ，训练Loss还是公式$\ref{5}$和公式$\ref{6}$，可以通过引导开关变量，进而让模型学会什么时候需要从输入文本复制词，这样可以避免OOV。

### 3 Coverage mechanism   
重复输入相同内容的问题一直是生成式模型绕不开的话题。本文作者的想法是：维护一个coverage vector($c^j$)，它是之前所有timestep的attention weight累计。这个coverage vector把输入文本的内容和该向量的每一个元素一一对应，这个向量的长度切好就是输入文本token的长度。因为从直觉上，某一槽位的累计值越高，说明它已经越有可能被输出了，而那些低值的槽位，则很有可能还没有被输出。可参考下图。  

<img src="https://github.com/errorplayer/AI_snippets/blob/master/pics/autosummarization-pointer-generator-network-pic1.png" width="50%" align="center">   

其计算公式如下：  
$$c^j=\sum_{t=0}^j a^t \tag{9}\label{9}$$  
> 在原文中，$\sum$号的上标是要减一的，这里没有减是因为，本文描述问题。只要知道是记录过去时刻的attention weight就可以了。  

把覆盖向量融入attention score的计算，将公式$\ref{1}$升级为公式$\ref{10}$：  
$$e_i^j=V*tanh(W_h h_i+W_s s_j+W_c c_i^jb_{attn})\tag{10}\label{10}$$   

然后还需要特别设计一个Loss，这个loss的设计可谓匠心独具。先看公式$\ref{11}$：  
$$covloss_j=\sum_i min(a_i^j, c_i^j)\tag{11}\label{11}$$  

首先它是有界的，因为$covloss_j\le \sum_i a_i^j=1$。如何理解这个loss的设计呢？   
1. 如果某个槽位的累计值很高，那么说明decoder之前已经很关注这个槽位了，那它很有可能已经被输出过，这个公式是两个找最小，所以希望选到$a_i^j$而不是$c_i^j$，那么如果当前的attention weight在这个槽位上比较大，说明decoder又来注意这个槽位的内容了，这个时候可能会重复输出，因此loss的值随着$a_i^j$的值增大而增大；反之，如果$a_i^j$很小，说明模型已经不怎么注意它了，这也符合不重复输出的目的，因而loss的值会小。  
2. 如果某个槽位的累计值很小，那么说明decoder之前没有怎么关注这个槽位，我们希望它的值比$a_i^j$越小越好。然后loss来选它($c_i^j$)。$a_i^j$的值在这种情况下应该越大越好，$c_i^j$在这种情况下应该越小越好。  
3. 从最理想的情况下，我们希望每次attention weight($a^j$)只有一个槽位是1，其他都是0，然后这样累加起来是不会有重叠的，而且会有很多槽位是0(因为这个任务是summarization,肯定不能全盘照顾到)。但是这种情况不会出现，所以作者设计的这个loss其实很精妙。看似表面每次只是简单的选取最小的那个作为loss就完事儿，但是仔细结合公式$\ref{9}$，就会发现，因为$c_i^j$是会累加$a_i^j$的，这样会暗中帮助模型在学习的时候注意让$a_i^j$,  $c_i^j$这两个量一个非常小一个尽量大。  
4. 此外作者还公式$\ref{11}$对比了机器翻译里的loss。作者提到，在Machine Translation里，提倡1:1的翻译比例，放在loss层面讲，也就是最终的coverage vector每个槽位必须刚好为1，不然就是要penalized的。(为什么是这样子，可以结合我前面讲的第3点最理想情况去想)。作者认为公式$\ref{11}$的loss计算方式是非常灵活的。因为针对摘要任务，我们不需要全盘涉及，可以允许内容丢失，所以公式$\ref{11}$的loss只惩罚历次attention weight的严重overlap，对于其他槽位是否也达到一定的值并不做要求，符合摘要的本质思想(不求全)，同时也有效抑制了重复输出的问题。   

最后只需要把公式$\ref{11}$的loss整合进公式$\ref{5}$就可以了:   
$$loss_j = -logP_{vocab}(w_j)+\lambda \sum_i min(a_i^j, c_i^j)\tag{12}\label{12}$$    

#### 谢谢查看！ 如有错漏，欢迎提issue。 


