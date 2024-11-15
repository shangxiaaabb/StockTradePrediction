<!--
 * @Author: huangjie huangjie20011001@163.com
 * @Date: 2024-10-21 10:02:27
-->
# Trading-Volume Prediction

**高优先级**：尽量在规定时间内完成
**低优先级**：在完成高优先级任务之后可以进行
**长期**：这部分内容需要长期进行没有时间限制不断进行

**Notice**:
1.每次使用代码时候,一定先: ```git pull```
2.代码都放到自己的分支里面(Google如何创建分支)不要上传到`main`分支上

> **HuangJie**

- [ ] （**长期**）继续调研时间序列预测方法论文(GNN、TransFormer在Time Series中应用)

**Paper:**
1.Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting
2.Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks
3.Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting
4.iTransformer: Inverted Transformers Are Effective for Time Series Forecasting
.................................................

## 11.15--11.21

- [ ] （**高优先级**）1.搞明白下面两篇论文中思路，以及傅里叶变换方法

**Paper:**
1.FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective
> https://github.com/aikunyi/FourierGNN

2.FAN: Fourier Analysis Networks

> **DongYuqing**

- [ ] （**长期**）可以去调研一下金融方面论文，了解一下还有什么可以构建的特征用到模型中
> 比如说可以整理为Excel文件（**然后做成共享链接**）: 1.特征；2.计算方法；3.论文

## 11.15--11.21

- [ ] （**高优先级**）1.先搞明白Feature-Engine中的代码思路,模型用到的特征是如何构建的
> 可以先搞明白[代码](https://github.com/shangxiaaabb/StockTradePrediction/blob/build-new-model/data_features/features_engine.ipynb)里面的方法，然后自己重新写一下

- [ ] （**低优先级**）2.时间充裕可以去了解一下[模型](https://github.com/shangxiaaabb/StockTradePrediction/blob/build-new-model/gcn/new_model/model.py)是如何构建的


**Notice1:**
原始代码里面使用的[数据](https://pan.baidu.com/s/1X4WIrhqQDQ_DqaSq3qySpA?pwd=hij7).SSH连接(以Vscode连接为例)服务器:
```
HOST Sun
  HostName 111.178.1.124
  Port 8083
  User rzhu
```
密码:zry113398
> 1.其他连接方式自己可以Google(Jupyter NoteBook, Spider等方式连接)
> 2.代码本地修改,**不要上传到Github上**(或者自己重新创建一个新的分支),最新的代码直接用```git pull```从Github上拉取代码即可
> 3.服务器上尽量只运行代码，**尽量不要去下载上传大文件**

**Notice2:**
数据也可以用[这部分的数据](https://github.com/shangxiaaabb/StockTradePrediction/tree/build-new-model/gcn/data/raw_data)不过这部分数据可能需要自己在去对代码修改

**Code**:
1.**Feature-Engine**:https://github.com/shangxiaaabb/StockTradePrediction/blob/build-new-model/data_features/features_engine.ipynb
2.**Model-Code**: https://github.com/shangxiaaabb/StockTradePrediction/blob/build-new-model/gcn/new_model/model.py
3.**FanLayer**: https://github.com/shangxiaaabb/StockTradePrediction/blob/build-new-model/gcn/new_model/FanModel.py