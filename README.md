### 文本分类项目
***
**本项目为基于CNN，RNN 和NLP中预训练模型构建的多个常见的文本分类模型。**

#### requirements
* python==3.5.6
* tensorflow==1.4.0

#### 1. 数据集
&ensp;&ensp;数据集为IMDB电影评论的情感分析数据集，总共有三个部分：
* 带标签的训练集：labeledTrainData.tsv
* 不带标签的训练集：unlabeledTrainData.tsv
* 测试集：testData.tsv

&ensp;&ensp;字段的含义：
* id  电影评论的id
* review  电影评论的内容
* sentiment  情感分类的标签（只有labeledTrainData.tsv数据集中有）

#### 2. 数据预处理 
&ensp;&ensp;数据预处理方法/dataHelper/processData.ipynb

&ensp;&ensp;将原始数据处理成干净的数据，处理后的数据存储在/data/preProcess下，数据预处理包括：
* 去除各种标点符号
* 生成训练word2vec模型的输入数据 /data/preProcess/wordEmbedding.txt

#### 3. 训练word2vec词向量
&ensp;&ensp;预训练word2vec词向量/word2vec/genWord2Vec.ipynb
* 预训练的词向量保存为bin格式 /word2vec/word2Vec.bin

#### 4. textCNN 文本分类
&ensp;&ensp;textCNN模型来源于论文[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)

&ensp;&ensp;textCNN可以看作是一个由三个单层的卷积网络的输出结果进行拼接的融合模型，作者提出了三种大小的卷积核[3, 4, 5]，卷积核的滑动使得其
类似于NLP中的n-grams，因此当你需要更多尺度的n-grams时，你可以选择增加不同大小的卷积核，比如大小为2的卷积核可以代表
2-grams.

&ensp;&ensp;textCNN代码在/textCNN/textCNN.ipynb。实现包括四个部分：
* 参数配置类 Config （包括训练参数，模型参数和其他参数）
* 数据预处理类 Dataset （包括生成词汇空间，获得预训练词向量，分割训练集和验证集）
* textCNN模型类 TextCNN
* 模型训练

#### 5. charCNN 文本分类
&ensp;&ensp;textCNN模型来源于论文[Character-level Convolutional Networks for Text
Classification](https://arxiv.org/abs/1509.01626)

&ensp;&ensp;char-CNN是一种基于字符级的文本分类器，将所有的文本都用字符表示，
*注意这里的数据预处理时不可以去掉标点符号或者其他的各种符号，最好是保存论文中提出的69种字符，我一开始使用去掉特殊符号的字符后的文本输入到模型中会无法收敛*。
此外由于训练数据集比较少，即使论文中最小的网络也无法收敛，此时可以减小模型的复杂度，包括去掉一些卷积层等。

&ensp;&ensp;charCNN代码在/charCNN/charCNN.ipynb。实现也包括四个部分，也textCNN一致，但是在这里的数据预处理有很大不一样，剩下
的就是模型结构不同，此外模型中可以引入BN层来对每一层的输出做归一化处理。

#### 6. Bi-LSTM 文本分类
&ensp;&ensp;Bi-LSTM可以参考我的博客[深度学习之从RNN到LSTM](https://www.cnblogs.com/jiangxinyang/p/9362922.html)

&ensp;&ensp;Bi-LSTM是双向LSTM，LSTM是RNN的一种，是一种时序模型，Bi-LSTM是双向LSTM，旨在同时捕获文本中上下文的信息，
在情感分析类的问题中有良好的表现。

&ensp;&ensp;Bi-LSTM的代码在/Bi-LSTM/Bi-LSTM.ipynb中。除了模型类的代码有改动，其余代码几乎和textCNN一样。

#### 7. Bi-LSTM + Attention 文本分类
&ensp;&ensp;Bi-LSTM + Attention模型来源于论文[Attention-Based Bidirectional Long Short-Term Memory Networks for
Relation Classification](http://aclweb.org/anthology/Y/Y15/Y15-1009.pdf)

&ensp;&ensp;Bi-LSTM + Attention 就是在Bi-LSTM的模型上加入Attention层，在Bi-LSTM中我们会用最后一个时序的输出向量
作为特征向量，然后进行softmax分类。Attention是先计算每个时序的权重，然后将所有时序
的向量进行加权和作为特征向量，然后进行softmax分类。在实验中，加上Attention确实对结果有所提升。

&ensp;&ensp;Bi-LSTM + Attention的代码在/Bi-LSTM+Attention/Bi-LSTMAttention.ipynb中，除了模型类中
加入Attention层，其余代码和Bi-LSTM一致。

#### 8. RCNN 文本分类
&ensp;&ensp;RCNN模型来源于论文[Recurrent Convolutional Neural Networks for Text Classification](https://arxiv.org/abs/1609.04243)

&ensp;&ensp;RCNN 整体的模型构建流程如下：
* 利用Bi-LSTM获得上下文的信息，类似于语言模型
* 将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput, wordEmbedding, bwOutput]
* 将拼接后的向量非线性映射到低维
* 向量中的每一个位置的值都取所有时序上的最大值，得到最终的特征向量，该过程类似于max-pool
* softmax分类

&ensp;&ensp;RCNN的代码在/RCNN/RCNN.ipynb中。

#### 9. adversarialLSTM 文本分类
&ensp;&ensp;Adversarial LSTM模型来源于论文[Adversarial Training Methods
For Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)

&ensp;&ensp;adversarialLSTM的核心思想是通过对word Embedding上添加噪音生成对抗样本，将对抗样本以和原始样本
同样的形式喂给模型，得到一个Adversarial Loss，通过和原始样本的loss相加得到新的损失，通过优化该新
的损失来训练模型，作者认为这种方法能对word embedding加上正则化，避免过拟合。

&ensp;&ensp;adversarialLSTM的代码在/adversarialLSTM/adversarialLSTM.ipynb中。

#### 10. Transformer 文本分类
&ensp;&ensp;Transformer模型来源于论文[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

&ensp;&ensp;Transformer模型有两个结构：Encoder和Decoder，在进行文本分类时只需要用到
Encoder结构，Decoder结构是生成式模型，用于自然语言生成的。Transformer的核心结构是
self-Attention机制，具体的介绍见[Transformer模型（Atention is all you need）](https://www.cnblogs.com/jiangxinyang/p/10069330.html)。

&ensp;&ensp;Transformer模型的代码在/Transformer/transformer.ipynb中。
#### 11. ELMo预训练模型 文本分类
&ensp;&ensp;敬请期待
#### 12. Bert预训练模型 文本分类
&ensp;&ensp;敬请期待


