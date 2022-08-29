[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=Xiefeng69.Network-Alignment-Competition
[repo-url]: https://github.com/Xiefeng69/Network-Alignment-Competition

# 2022 CCF大数据与计算智能大赛——带噪音的社交网络对齐

[![visitors][visitors-img]][repo-url]

## 赛题名称
带噪音的社交网络对齐，[CCF BDCI2022官网链接](https://wap.datafountain.cn/competitions/598)

关键词：`网络对齐` `智能算法`

👏 欢迎大家参赛～

## 赛题背景
各种类型的社交网络和应用已经进入到人类生活中的方方面面。人们在新浪微博上分享日常并与好友保持联络，在LinkedIn上进行求职招聘，在Arxiv.org上获取和分享学术成果等。

社交网络（节点）对齐，是在多个社交网络之间找到对应的用户，这些对应用户属于真实世界中的同一个自然人。融合不同社交网络的数据可以从多个角度对用户建立更为立体、全面、综合的画像，挖掘其本质特征，进行更精准的分析和服务。近年来在这一领域已有很多研究成果及论文（参考“近年来相关研究论文”一节）。一些相关研究将此问题称作社交网络用户对齐、用户身份链接、锚链路预测、用户匹配等。

值得注意的是，在真实应用中，网络中的结构差异信息和已知锚点对中的噪音会给对齐任务带来巨大的挑战。假设待对齐的两个社交网络分别为G1和G2，nodes(G1)和nodes(G2)分别表示G1和G2的所有节点的集合。为简化任务且不失一般性，这里可假设nodes(G1)=nodes(G2)。对齐任务就是识别G1和G2中属于同一自然人的“用户节点对”。真实应用中的网络差异信息和噪音可以简化为以下两种情况：

+ 网络结构差异：由于信息获取限制和不同社交网络的功能差异，网络G1和G2在结构上会存在不同，记为G1和G2网络结构的差异率alpha。
+ 锚点监督信息的噪音：对于G1和G2的“用户节点对”，在真实应用中通常会有少量“锚点监督信息”（可能是人工标注的）。这些监督信息可能会给对齐算法带来帮助。但真实情况中这些对齐信息本身可能带有“噪音”（错误率），记为beta。对于对齐任务而言，beta度量的是G1和G2锚点监督信息的错误率。

上述两种信息差异或噪音情况（alpha和beta）均会给社交网络对齐任务带来困难。设计出能够应对带有噪音的社交网络对齐算法，能够让算法更加贴近和适合真实应用条件，具有重要的现实意义。

## 赛题任务
依据Arenas的Email网络数据，利用机器学习、深度学习等相关技术，建立一个在噪音存在情况下能准确对齐网络实体的模型，从而分析并挖掘网络中用户的对齐关系。注意，在给出的两个网络中，存在结构上的差异（即alpha），在提供的锚点对信息中，存在监督噪音信息（即beta）。

## 赛题数据
本次任务所使用的数据来自Regal提供的Arenas邮件网络，采样网络中共包含1135个节点，5451条边。全部数据已经脱敏。为了验证参赛者设计的模型在不同噪音场景的鲁棒性，我们设计网络结构差异alpha={0.1,0.2}，已知对齐监督信息噪音比率beta={0.2,0.4}。

+ 网络拓扑数据：`data/graph/`
    + data_G1.npy：网络1的拓扑邻接矩阵（源网络）
    + data_G2.npy：网络2的拓扑邻接矩阵，相比于G1存在0.1的结构差异
    + data_G3.npy：网络3的拓扑邻接矩阵，相比于G1存在0.2的结构差异
+ 锚点对信息：`data/anchor/`
    + anchor_0.2.txt：噪音量为20% (beta=0.2)的已知对齐节点
    + anchor_0.4.txt：噪音量为40% (beta=0.4)的已知对齐节点
+ 目标：生成所有节点的对应关系，不失一般性，我们假设源网络的每个节点等价于目标网络中的唯一实体，
    + 参赛者以.txt文件格式提交，提交模型结果到大数据竞赛平台，平台在线评分，实时排名。txt文件中包含**四组实验**的结果，每组结果应该包含网络1中所有节点向网络2中节点的对齐结果，**即共1035×4条对齐**。实验结果的顺序为：
        + alpha=0.1, beta=0.2
        + alpha=0.1, beta=0.4
        + alpha=0.2, beta=0.2
        + alpha=0.2, beta=0.4
    + 文件的第一列永远为网络1中节点的id，第二列为网络2中节点的id，参考`sample.txt`。
    + 注意请严格按照提交顺序进行提交。
+ 评价标准：本赛题使用Hit@1值进行评价。

## Baseline
赛题baseline代码于`src/`文件夹中。输入如下代码开始运行：

```shell
python train.py --graph_s data_G1 --graph_d data_G2 --anoise 0.2
```

解释：此命令行运行的代码使用G1和G2（存在0.1的结构差异）来进行节点对齐，并利用有0.2噪音的pre-aligned节点对作为监督信息。

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
