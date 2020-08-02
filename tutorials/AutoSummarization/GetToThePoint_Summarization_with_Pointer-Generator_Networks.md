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
