'''
Created on 05 Nov 2019

@author: danhbuithi
'''

import sys 
import numpy as np

from common.CommandArgs import CommandArgs
from common.DataSet import DataSet
from rulefit.rulefit import RuleFit
from sklearn.metrics.classification import f1_score
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
import time


def evaluateByF1(y_pred, y_true):
    a = f1_score(y_true, y_pred, average='micro')
    b = f1_score(y_true, y_pred, average='macro')
    return (a, b)

if __name__ == '__main__':
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'test'   : ('', 'Path of testing data file'),
                          'class'   : (None, 'Class index'),
                          'n'   : (250, 'Maximum number of rules')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    
    class_index = int(config.get_value('class'))
    nrules = int(config.get_value('n'))

    
    for i in range(5):
        train_data = DataSet()
        train_data.load(config.get_value('train')+'.'+str(i), class_index)
        
        test_data = DataSet()
        test_data.load(config.get_value('test')+'.'+str(i), class_index)
        print(train_data.size())
        
        '''
        Convert data into binary
        '''
        rel_train_X = train_data.get_X_in_binary()
        rel_train_Y = train_data.get_Y_in_numeric()
        train_X = rel_train_X.relation_matrix
        train_Y = rel_train_Y.values
        train_Y[train_Y < 1] = -1
        train_Y[train_Y >= 1] = 1
        print(np.unique(train_Y, return_counts=True))
            
        test_X = test_data.get_X_in_binary_with(rel_train_X.item_dict)
        test_Y = test_data.get_Y_in_numeric_with(rel_train_Y.item_dict)
        test_Y[test_Y < 1] = -1
        test_Y[test_Y >= 1] = 1
        print(np.unique(test_Y, return_counts=True))
        
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=10, learning_rate=0.01,random_state=1)
        rf = RuleFit(tree_generator=gb,rfmode='classify',max_rules=nrules,random_state=1)
        N=train_X.shape[0]
        start = time.time()
        rf.fit(train_X, train_Y)
        rules = rf.get_rules()
        print('# rules', len(rules))
        print('executing time', time.time()-start)
        
        print('Predict for training data')
        y_pred=rf.predict(train_X)
        print(evaluateByF1(y_pred, train_Y))
              
        print('Predict for testing data')
        y_pred=rf.predict(test_X)
        print(evaluateByF1(y_pred, test_Y))
