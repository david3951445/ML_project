# Final Profect Proposal
題目：台灣牧場乳量預測
組員：洪廷維、林宏宇、趙柏鈞

## 進度表
- 5/20~5/24 
    - 資料前處理
- 5/25~5/31
    - 資料前處理
- 6/1~6/7
    - 機器學習模型
- 6/8~6/14
    - 測試、調參
- 6/15~6/21
    - 測試、調參
- 6/22~6/28(DEMO)
     - 測試、調參
- 6/29~7/5(report)
    - 打report

## 問題分析
- 適合的模型?
    - 監督式學習、回歸模型
        - BLR(Bayesian linear regression) 、MLR(Maximum Likelihood regression)、
        - Ridge Regression、CART(Classification and Regression Trees)、
        - GBM(Gradient Boosting Machine)、NN(Neural Network)
## 軟體架構
Python, Tensorflow, Keras
### 資料前處理
- 缺失值處理
- 類別資料處理
- 資料特徵縮放
- 其他
    - 刪除不重要欄位
    - 新增欄位

### 機器學習模型

> [name=david3951445]選哪種?
> 目前暫定以神經網路NN(Neural Network)為我們的訓練模型的基礎，以均方誤差作為損失函數與評估標準的依據。
參數及優化方式會根據測試結果作調整，若有效果更好的模型會再嘗試。

## 軟體流程
利用Keras的Sequential模型建立神經網路
