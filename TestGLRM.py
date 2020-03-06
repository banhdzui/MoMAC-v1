'''
Created on 05 Nov 2019

@author: danhbuithi
'''

import sys 
import pandas as pd 

from common.CommandArgs import CommandArgs
from common.DataSet import DataSet
from TestRuleBasedMethods import evaluateByF1
from aix360.algorithms.rbm.GLRM import GLRMExplainer
from aix360.algorithms.rbm.features import FeatureBinarizer
from aix360.algorithms.rbm.logistic_regression import LogisticRuleRegression


if __name__ == '__main__':
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'test'   : ('', 'Path of testing data file'),
                          'class'   : (None, 'Class index')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
    
    class_index = int(config.get_value('class'))    
    for i in range(5):
        train_data = DataSet()
        train_data.load(config.get_value('train')+'.'+str(i), class_index)
        
        test_data = DataSet()
        test_data.load(config.get_value('test')+'.'+str(i), class_index)
        print(train_data.size())
        
        feature_model = FeatureBinarizer()
        
        '''
        Convert data into binary
        '''
        rel_train_X = train_data.get_X_in_binary()
        rel_train_Y = train_data.get_Y_in_numeric()
        train_X = rel_train_X.relation_matrix
        
        origtrain_X = pd.DataFrame({'a'+str(i): train_X[:,i] for i in range(train_X.shape[1])})
        feature_model.fit(origtrain_X)
        train_X = feature_model.transform(origtrain_X)

        train_Y = rel_train_Y.values
        train_Y[train_Y < 1] = -1
        train_Y[train_Y >= 1] = 1
        
        
        test_X = test_data.get_X_in_binary_with(rel_train_X.item_dict)
        test_X = pd.DataFrame({'a'+str(i):test_X[:,i] for i in range(test_X.shape[1])})
        test_X = feature_model.transform(test_X)
                                         
        test_Y = test_data.get_Y_in_numeric_with(rel_train_Y.item_dict)
        test_Y[test_Y < 1] = -1
        test_Y[test_Y >= 1] = 1
        
        
        #model = LinearRuleRegression()
        model = LogisticRuleRegression(lambda0=0.001, lambda1=0.001)
        rf = GLRMExplainer(model)
        
        rf.fit(train_X, train_Y)
        
        print(rf.explain(200).to_string())
        
        print('Predict for training data')
        y_pred=rf.predict(train_X)
        #print(y_pred)
        print(evaluateByF1(y_pred, train_Y))
              
        print('Predict for testing data')
        y_pred=rf.predict(test_X)
        print(evaluateByF1(y_pred, test_Y))
