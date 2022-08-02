## In-feed Ads Cross-Domain CTR Prediction: 广告-信息流跨域CTR预估
### Introduction to the competition: 赛题介绍
Ad recommendation is mainly modeled based on users' historical exposure to ads, clicks and other behaviors, and if only ad domain data is used, user behavior data is sparse and behavior types are relatively single. And introducing cross-domain data of the same media can obtain the behavior data of the same ad user in other domains, deeply mining user interests and enriching user behavior characteristics. Introducing the behavior data of ad users in other media can also enrich the user and ad features.

This contest expects players to optimize ad ctr prediction accuracy based on ad log data, basic user information and cross-domain data. The target domain is the ad domain and the source domain is the infomercial recommendation domain, and the user interest modeling is performed to help the accurate prediction of the ad domain CTR by obtaining user behavior data such as exposure and click on infomercial in the infomercial domain.

广告推荐主要基于用户对广告的历史曝光、点击等行为进行建模，如果只是使用广告域数据，用户行为数据稀疏，行为类型相对单一。而引入同一媒体的跨域数据，可以获得同一广告用户在其他域的行为数据，深度挖掘用户兴趣，丰富用户行为特征。引入其他媒体的广告用户行为数据，也能丰富用户和广告特征。

本赛题希望选手基于广告日志数据，用户基本信息和跨域数据优化广告ctr预估准确率。目标域为广告域，源域为信息流推荐域，通过获取用户在信息流域中曝光、点击信息流等行为数据，进行用户兴趣建模，帮助广告域CTR的精准预估。

[Contest Link: 比赛地址](https://developer.huawei.com/consumer/cn/activity/starAI2022/algo/competition.html#/preliminary/info/006/rank)

### Description of data: 数据说明
The data provided includes **target domain user behavior data** and **source domain user behavior data**.

提供的数据包括目标域用户行为数据，源域用户行为数据。


### Assessment methods: 评估方式
Evaluation method: the sample CTR estimates of the advertising domain are counted, and GAUC and AUC are calculated

Evaluation index: this competition uses the weighted sum of GAUC and AUC as the evaluation index with the following formula: xAUC = α∗GAUC + β∗AUC

The higher xAUC means the better result and the higher ranking, where AUC is the AUC statistic of the whole sample, GAUC is the weighted sum of grouped AUCs, grouped by user as dimension, and the group weight is the exposure within the group/total exposure

Preliminary round: α is 0.7, 𝛽 is 0.3

评估方式：统计广告域的样本CTR预估值，计算GAUC和AUC

评测指标： 本次比赛使用GAUC和AUC的加权求和作为评估指标，具体公式如下：xAUC=α∗GAUC+β∗AUC
xAUC越高，代表结果越优，排名越靠前。其中，AUC为全体样本的AUC统计，GAUC为分组AUC的加权求和，以用户为维度分组，分组权值为分组内曝光量/总曝光)

初赛：α 为 0.7，𝛽为 0.3

### Task 1: Simple Trial with Logistic Regression
- Sign up for the contest
- Download Contest Data
- Read Data
- Simple Logistic Regression Model
```python
cols_train = ['log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'device_name',
              'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd']
cols_test = ['log_id', 'user_id', 'age', 'gender', 'residence', 'device_name',
              'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd']

# Data of user behaviour in the target domain 目标域用户行为数据
train_ads = pd.read_csv('../input/ctr-prediction/train_data_ads.csv', usecols = cols_train)
test_ads = pd.read_csv('../input/ctr-prediction/test_data_ads.csv', usecols = cols_test)

# Data Sampling 数据采样
train_ads = pd.concat([train_ads[train_ads['label'] == 0].sample(70000),
                       train_ads[train_ads['label'] == 1].sample(10000)])

# Load Logistic Regression Model for Training 加载逻辑回归模型，训练
clf = LogisticRegression()
clf.fit(train_ads.drop(['log_id', 'label', 'user_id'], axis=1),train_ads['label'])

# Model Prediction 模型预测
test_ads['pctr'] = clf.predict_proba(test_ads.drop(['log_id', 'user_id'], axis=1),)[:, 1]

# Output to csv file 写入文件
test_ads[['log_id', 'pctr']].to_csv('submission.csv',index=None)
```
- Submission Result 提交结果

### Task 2: Exploratory Data Analysis 比赛数据分析
- Analysis of user behavior for the target domain (对目标域用户行为进行分析)
1. What is the percentage of user overlap for the training and test sets? (对于训练集 和 测试集，用户重合的比例是多少)
2. How many numeric and how many non-numeric fields are in the statistical fields? (统计字段中有多少数值字段，多少非数值字段)
3. Count which user attributes (age, gender, mobile device, etc.) correlate most strongly with the label? (统计哪些用户属性, 如年龄、性别、手机设备等， 与标签相关性最强)

- Analyze the source domain user behavior (对源域用户行为进行分析)
1. What is the percentage of overlap between the source domain user behavior and the target domain user behavior for the training and test set users, respectively? (源域用户行为 与 目标域用户行为 训练集和测试集用户重合的比例分别是多少)
2. How many numeric fields and how many non-numeric fields are in the statistics field? (统计字段中有多少数值字段，多少非数值字段)

- Understand the logic of the data fields and try to group the data fields (理解数据字段的逻辑，并尝试对数据字段进行分组)























