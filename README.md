# Network Alignment Competition

## 赛题名称
噪音网络中的对齐算法
## 赛题背景
随着互联网的普及和信息化时代的来临，社交网络已经进入到人类生活中的方方面面，人们在新浪微博上分享日常并与好友保持联络，在LinkedIn上进行求职招聘，在Google scholar上进行学术交流等。仅针对单一网络的研究限制了可利用的数据的范围，而融合不同社交网络的数据可以从多个角度对人物建立更为立体、全面、综合的画像，挖掘其本质特征，进行更精准的分析并服务于进一步的应用任务。然而，网络中的缺失信息和噪音给对齐任务带来了巨大的挑战。识别不同社交平台中属于同一自然人的用户账号，一些相关研究将此问题称作社交网络对齐、用户身份链接、锚链路预测、用户匹配。

## 赛题任务
依据Arenas的Email网络数据中的拓扑信息，利用机器学习、深度学习等相关技术，建立一个在噪音存在情况下能准确对齐网络实体的模型，从而分析并挖掘网络中用户的对齐关系。注意，在给出的两个网络中，存在结构上的差异（异构），在提供的锚点对信息中，存在噪音信息。

## 赛题数据
数据来自Regal提供的Arenas邮件网络，全部数据已经脱敏。采样网络中共1135个节点，5451条边。
+ 网络拓扑数据：`data/graph/`
+ 锚点对信息：`data/anchor/`
+ 目标：生成所有节点的对应关系。

## Baseline
赛题baseline代码于`src/`文件夹中。输入如下代码开始运行：

```shell
python train.py
```

## 参考文献
+ Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment (IJCAI17)
+ Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks (EMNLP18)
+ Regal: Representation Learning-based Graph Alignment (CIKM18)
+ Bootstrapping Entity Alignment with Knowledge Graph Embedding (IJCAI18)
+ Multi-Channel Graph Neural Network for Entity Alignment (ACL19)
+ Two-stage entity alignment: Combining hybrid knowledge graph embedding with similarity-based relation alignment (PRICAI19)
+ REA: Robust Cross-lingual Entity Alignment between Knowledge Graphs (SIGKDD20)
+ Knowledge Graph Alignment Network with Gated Multi-Hop Neighborhood Aggregation (AAAI20)
+ Adversarial Attack against Cross-lingual Knowledge Graph Alignment (EMNLP21)
+ Make It Easy-An Effective End-to-End Entity Alignment Framework (SIGIR21)
+ Graph Alignment with Noisy Supervision (WWW22)
+ Multilingual Knowledge Graph Completion with Self-Supervised Adaptive Graph Alignment (ACL22)
+ A comprehensive survey of entity alignment for knowledge graphs (AI Open 2021)