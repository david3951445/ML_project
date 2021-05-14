import numpy as np
import pandas as pd
import math
from pandas.core.frame import DataFrame
import os 
import csv

# ignore warning : This TensorFlow binary is optimized with oneAPI ...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    ''' load data '''

    data_report = pd.read_csv('data/report.csv')
    data_submission = pd.read_csv('data/submission.csv')
    data_birth = pd.read_csv('data/birth.csv')
    # data_submission = pd.read_csv('data/submission.csv')
    # data_submission = pd.read_csv('data/submission.csv')

    ''' Data pre-processing '''

    '''
    brith.csv
    drop col : 8,   
    '''
    print(data_birth.shape)
    data_birth = data_birth.drop_duplicates(['1'])
    print(data_birth.shape)
    

    '''
    bread.csv
    '''

    '''
    report.csv
    '''
    # data_report = data_report.drop(['19'], axis=1) # drop colume19
    # data = data_report.dropna()
    # print(data_report.shape)
    # print(data.shape)
    
    '''
    spec.csv (health)
    '''

    ''' split x_test from report.csv '''

    output_id = data_submission['ID']
    # x_test = data_report.loc[output_id] # test data (x)
    # data_report = data_report.drop(output_id) # train data (x, y)
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