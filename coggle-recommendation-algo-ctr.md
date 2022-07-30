## 广告-信息流跨域CTR预估
### 赛题介绍
广告推荐主要基于用户对广告的历史曝光、点击等行为进行建模，如果只是使用广告域数据，用户行为数据稀疏，行为类型相对单一。而引入同一媒体的跨域数据，可以获得同一广告用户在其他域的行为数据，深度挖掘用户兴趣，丰富用户行为特征。引入其他媒体的广告用户行为数据，也能丰富用户和广告特征。

本赛题希望选手基于广告日志数据，用户基本信息和跨域数据优化广告ctr预估准确率。目标域为广告域，源域为信息流推荐域，通过获取用户在信息流域中曝光、点击信息流等行为数据，进行用户兴趣建模，帮助广告域CTR的精准预估。

比赛地址：https://developer.huawei.com/consumer/cn/activity/starAI2022/algo/competition.html#/preliminary/info/006/rank

### 数据说明
提供的数据包括目标域用户行为数据，源域用户行为数据, 以下按照这2个维度分别 说明。


### 评估方式
评估方式：统计广告域的样本CTR预估值，计算GAUC和AUC

评测指标： 本次比赛使用GAUC和AUC的加权求和作为评估指标，具体公式如下：xAUC=α∗GAUC+β∗AUC
xAUC越高，代表结果越优，排名越靠前。其中，AUC为全体样本的AUC统计，GAUC为分组AUC的加权求和，以用户为维度分组，分组权值为分组内曝光量/总曝光)
初赛：α 为 0.7，𝛽为 0.3
