import numpy as np
from numpy.core.numeric import NaN
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
        COL 2 - COL 3(前一胎次) = 乾乳期
    COL 4, 5 : 犢牛1, 犢牛2
        insufficient, drop()
    COL 6 : 母牛體重
    COL 7, 9 : 登錄日期, 胎次 
        repeat, drop()
    COL 8 : 計算胎次
        meaningless, drop()
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
    x_train = pd.DataFrame()

    # COL 3
    temp = data_report.iloc[:, 2]
    temp = temp.replace([[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]], ['spring', 'summer', 'autumn', 'winter'])
    x_train = pd.concat([x_train, temp], axis=1) # axis=1 means colume


    x_train = pd.concat([x_train, data_report['4']], axis=1)
    x_train = pd.concat([x_train, data_report['9']], axis=1)
    x_train = pd.concat([x_train, data_report['10']], axis=1)
    x_train = pd.concat([x_train, data_report['14']], axis=1)
    x_train = pd.concat([x_train, data_report['18']], axis=1)
    y_train = pd.concat([x_train, data_report['11']], axis=1)


    # # birth_interval
    temp1 = data_report['12'].copy()
    temp2 = data_report['19'].copy()
    i1 = np.where(temp1.isna())[0]
    i2 = np.where(temp2.isna())[0]

    # 補缺項
    temp1.iloc[i1] = temp1.iloc[i1[0] + 1] , # temporary method
    temp2.iloc[i2] = data_report.iloc[i2, 7] # 第一次分娩 - 出生日期

    birth_interval = day_interval(temp1, temp2, 'birth_interval')
    x_train = pd.concat([x_train, birth_interval], axis=1)


    # # # dry_interval
    # # 先透過 data_birth 計算乾乳期
    data_birth_copy = data_birth.copy() # copy
    data_birth_copy = data_birth_copy.sort_values(by=['1', '9']) # sort 牛編號, 胎次
    i_cow_b = ~data_birth_copy.duplicated(subset=['1']) # find all cow

    temp = data_birth_copy.iloc[:, 2].shift() # 原資料的乾乳時間是下個胎次的
    temp.loc[i_cow_b] = NaN # teporary method, 第一胎次乾乳期 = NaN
    dry_interval = day_interval(data_birth_copy.iloc[:, 1], temp, 'dry_interval')
 
    # 補缺項
    data_birth_copy = pd.concat([data_birth_copy, dry_interval, pd.DataFrame(i_cow_b, columns=['i_cow_b'])], axis=1)  
    index = [a or b for a, b in zip(data_birth_copy['dry_interval'] < 0, data_birth_copy['dry_interval'] > 5*30)]
    data_birth_copy = data_birth_copy.drop(data_birth_copy.loc[index].index) # 不合理的值直接排除 (保留 0~150天)
    mean = np.mean(data_birth_copy['dry_interval'])
    temp_cow_dry = data_birth_copy.fillna(mean) # teporary method, NaN(第一胎次乾乳期) = 平均值

    # # 把 birth 的資料融入 report
    # 索引操作
    data_report_copy = data_report.copy()
    data_report_copy = data_report_copy.sort_values(by=['5', '9']) # sort 牛編號, 胎次
    i_cd_b = temp_cow_dry.set_index(keys = ['1', '9']) # 將牛編號, 胎次轉為 index
    i_cd_b = i_cd_b.drop(i_cd_b.columns.drop(['dry_interval']), axis=1) # 保留 index, dry_interval
    i_cd_r = data_report_copy.set_index(keys = ['5', '9']) # 將牛編號, 胎次轉為 index
    i_cd_r = i_cd_r.drop(i_cd_r.columns, axis=1) # 保留 index 就好
    
    # 補缺項
    for a, b in i_cd_b.index :
        try :
            i_cd_r.loc[(a, b), 'dry_interval'] = i_cd_b.loc[(a, b), 'dry_interval'].iloc[0]
        except :
            continue # 如果birth有report沒有的牛，跳過 (經測試，只有一隻)
    i_cd_r['dry_interval'] = i_cd_r['dry_interval'].fillna(mean) # teporary method, 如果report有birth沒有的牛 乾乳期 = 平均值

    # 塞進 x_train
    array = i_cd_r.to_numpy()
    temp = pd.DataFrame(array, columns=['dry_interval'])
    temp.loc[data_report_copy.index.values, ['dry_interval']] = array
    x_train = pd.concat([x_train, temp], axis=1)

    print(x_train)
    
    #test = pd.DataFrame(a)
    x_train.to_csv('test.csv')

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

# day intervel of two Series with string type
def day_interval(temp1, temp2, name) :
    date1 = pd.to_datetime(temp1)
    date2 = pd.to_datetime(temp2)
    #date1 = [datetime.strptime(i, "%Y/%m/%d %H:%M") for i in temp1]
    #date2 = [datetime.strptime(i, "%Y/%m/%d %H:%M") for i in temp2] 
    return pd.DataFrame([(a - b).days for a, b in zip(date1, date2)], columns=[name], index=temp1.index) # preserver temp1.index

if __name__ == '__main__':
    main()

    # df2 = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
    # print(df2)
    # f = pd.Series([0, 2])
    # df2.iloc[f, [0, 2]] = [[10, 10], [10, 10]]
    # print(df2)
