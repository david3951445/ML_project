import numpy as np
import pandas as pd
import math
from pandas.core.frame import DataFrame
import os 
import csv

from datetime import datetime
from pandas.core.reshape.concat import concat

# ignore warning : This TensorFlow binary is optimized with oneAPI ...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    ''' load data '''

    data_report = pd.read_csv('data/report.csv')
    data_submission = pd.read_csv('data/submission.csv')
    data_birth = pd.read_csv('data/birth.csv')
    # data_submission = pd.read_csv('data/submission.csv')
    # data_submission = pd.read_csv('data/submission.csv')
    
    '''
    # Data pre-processing #
    important data : season of calv-ing,  氣候,  泌乳高峰第幾天 , stocking  rate
    一開始的六個星期中奶量不斷提高，一直到每日25至60升，然後不斷下降
    
    brith.csv
    COL 2, 3 :
        COL 3 - COL 2 = 乾乳期), 胎次
        repeat, drop()
    COL 10 : 分娩難易度
    COL 11, 12: 犢牛體型, 犢牛性 
        insufficient, drop()
    COL 13 : 酪農場代號
        repeat, drop()

    
    bread.csv


    report.csv
    COL 2 : 年
        drop
    COL 3 : 月
    x_train.replace([3, 4, 5], 'spring')
    x_train.replace([6, 7, 8], 'summer')
    x_train.replace([9, 10, 11], 'autumn') 
    x_train.replace([12, 1, 2], 'winter')
    COL 4 : 農場代號
    COL 5 : 乳牛編號
    COL 6, 7 : 父、母
        drop()
    COL 8 : 出生日期
        drop()
    COL 9 : 胎次
        反比
    COL 10 : 泌乳天數 (COL 15 - COL 12)
    COL 11 : 乳量
    COL 12 : 最近分娩
        if 19 has value
            分娩間隔 = COL 12 - COL 19
        else
            分娩間隔 = COL 12 - COL 8 # 第一次分娩 - 出生日期
    COL 13 : 採樣日期 (COL 15 - (1day ~ 3day))
        drop()
    COL 14 : 月齡
        反比
    COL 15 : 檢測日期 (年/月 : COL 2 / COL 3)
        drop()
    COL 16 : 最後配種日期 (=受精)
    COL 17 : 最後配種精液
    COL 18 : 配種次數
        反比
    COL 19 : 前次分娩日期
        drop()
    COL 20 : 第一次配種日期
    COL 21 : 第一次配種精液

    spec.csv (health)
 
    '''

    # # construct train data
    x_train = pd.DataFrame(data_report.iloc[:, 0])# ID

    temp = data_report.iloc[:, 2]
    temp = temp.replace([[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]], ['spring', 'summer', 'autumn', 'winter'])
    x_train = pd.concat([x_train, temp], axis=1) # axis=1 means colume

    x_train = pd.concat([x_train, data_report.iloc[:, 3]], axis=1)
    x_train = pd.concat([x_train, data_report.iloc[:, 8]], axis=1)
    x_train = pd.concat([x_train, data_report.iloc[:, 9]], axis=1)
    y_train = pd.concat([x_train, data_report.iloc[:, 10]], axis=1)
    
    temp1 = data_report.iloc[:, 11]
    temp2 = data_report.iloc[:, 18]
    i1 = np.where(temp1.isna())[0]
    i2 = np.where(temp2.isna())[0]
    temp1.values[i1] = temp1.values[i1 + 1] # 補缺項, temporary method
    temp2.values[i2] = data_report.values[i2, 7] # 補缺項, 第一次分娩 - 出生日期
    date1 = [datetime.strptime(i, "%Y/%m/%d %H:%M") for i in temp1]
    date2 = [datetime.strptime(i, "%Y/%m/%d %H:%M") for i in temp2]
    birth_interval = [(i - j).days for i, j in zip(date1, date2)]
    x_train = pd.concat([x_train, pd.DataFrame(birth_interval, columns=['birth_interval'])], axis=1)

    x_train = pd.concat([x_train, data_report.iloc[:, 13]], axis=1)
    x_train = pd.concat([x_train, data_report.iloc[:, 17]], axis=1)

    temp = data_birth.iloc[:, 10]
    i = np.where(temp.notna())[0]
    temp1 = data_birth.iloc[:, 0]
    ii = temp1.drop_duplicates()
    print(i.shape)
    print(ii.shape)

    # # 補缺項

    # # one hot
    
    # # split x_test from x_train
    output_id = data_submission['ID']
    # x_test = x_train.iloc[output_id] # test data (x)
    # x_train = x_train.drop(output_id) # train data (x, y)
    # x_train = 
    # y_train = 
    
    ''' ML model '''

    ''' output '''

    # # input x_test, output y_predict
    y_predict = np.ones(len(output_id))
    # y_predict = model.predict(x_test)
    data_submission['1'] = y_predict
    data_submission.to_csv('out.csv', index=False)

    # print('MSE of BLR = {e1}, MSE of MLR= {e2}.' .format(e1=, e2=, )

if __name__ == '__main__':
    main()