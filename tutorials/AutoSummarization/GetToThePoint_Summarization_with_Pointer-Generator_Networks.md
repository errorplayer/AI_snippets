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
2. Generator -- 生成词：从它的extended vocabulary里选一个词。extended vocabulary 就是[训练语料的词 + 所有输入文本的词]。尽量避免了它“无词可用”的尴尬。   
3. Coverage machanism -- 记录 $t$ 时刻前的attention累和，用作Loss计算依据，控制模型生成时对输入文本的内容覆盖。  
4. Pointer和Generator在实现过程中是借助一个联动开关变量 $P_{gen}$ 来切换的。


### 1 baseline model  
也就是seq2seq2 with attention 模型。  
按次序把每一个输入文本token输进去，都会产生一个对应当前时刻及过去时刻的隐藏状态$h_i$，把最后一个token输入进去以后，产生的代表整个输入文本的隐藏状态$h_t$可以直接当作encoder对整个输入文本的一个编码表示，也可以经过某些变换，假设变换后把它记为$s_t$。    
接下来就是attention score的计算。这个score的计算方式可以有很多种，文中提到的是  
$$e_i^t=v*tanh(W_h h_i+W_s s_t+b_{attn})\tag{1}\label{1}$$  
> 这里注意的地方：  
> 1.下标i 代表第几个encoder的输入token，范围是输入token个数，也就是输入文本长度。 $h_i$是固定的，因为每放入一个输入token就会产生一个对应位置的 $h_i$。  
> 2. $s_t$并非不变的，它是decoder每接收一个decoder的输入token产生的对应位置的$s_t$, 其实写成$s_j$更好解释，$j$表示第几个decoder的输入token，decoder 没有输出 \<EOS\> 前，我们是不知道$j$最大能有多大的。比如现在给decoder输入第2个token，我们需要decoder预测第三个token，那么就需要带入$s_2$去公式$\ref{1}$算新的向量$e_t$， 记住$h_i$是固定的。   
> 3. 这里的$e_i$是标量，代表decoder输入第$j$个token预测第$j+1$个token时，对第$i$个encoder输入token的注意值。$e^t=(e_0, e_1, ..., e_N)$ 是一个向量，是decoder在做预测第$j+1$个token时对整个encoder输入序列的全局预览。  

计算完attention score，就用一个$softmax$将其转成attention weights。      
$$a^t=softmax(e^t)\tag{2}$$  
用attention weights去weight sum$h_i$, 得到上下文向量$h_t^\*$。  
$$h_t^\*=\sum_i a_i^t h_i\tag{3}$$  
公式$\ref{3}$算出来的上下文向量可以当作是decoder在当前这一timestep对整个输入序列的回顾，接下来把它和decoder前一时刻的隐藏状态拼接，再经过两个线性层即可得到预测词的概率分布$P_{vocab}$    
$$P_{vocab}=softmax(V_2(V_1(s_j,h_t^\*)+b_1)+b_2)\tag{4}\label{4}$$  



