### Intro  
#### Now it is a simple English2Chinese translation model based on [Transformer](https://arxiv.org/abs/1706.03762).  For instructional purpose.

### Features  
+ supports chinese corpus training (data preprocess methods prepared)
+ supports raw natural chinese sentence test immediately (rough pretrained model & preprocessed corpus dicts pickle files available for a quick taste)
> ![sentence immediate test](https://github.com/errorplayer/AI_snippets/blob/master/Transformer_greedy_decoder/pic/display_sentence_test.JPG)  
+ supports importation of  your new parallel corpus data (easily adjust some parameters to preprocess your own corpus data)  
> we prefer the corpus data with each line following the format below in a *.txt* file:  
> (no space)english sentence **\t** chinese sentence **\n**   
+ readable code comments  
+ this code has been transformed from [graykode/nlp-tutorial/5-1.Transformer](https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer) with some major/tiny additions/modifications for english2chinese training (e.g. new corpus auto-preprocess with configurable parameters, data generator, batch training mode, position embedding auto-added, single sentence immediate test...)



### Reference
#### [Github](https://github.com/graykode/nlp-tutorial/tree/master/5-1.Transformer) graykode/nlp-tutorial   
#### [paper](https://arxiv.org/abs/1706.03762) Attention Is All You Need
#### [Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html) The Annotated Transformer
