'''
Created on 19 Mar 2018

@author: danhbuithi
'''
import sys
from common.CommandArgs import CommandArgs
from common.DataSet import DataSet
from corels.corels import CorelsClassifier
from TestRuleBasedMethods import evaluateByF1
    
             
if __name__ == '__main__':
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'class'   : (None, 'Class index'),
                          'test'  : ('', 'Path of testing data file'),
                          'nloop'   : (10000, 'Number of loops'),
                          'c': (0.01, 'Length penalty'),
                          'n':  (5, 'Number of subsets'),
                          'minsup'  : (0.1, 'Minimum support'),
                          'card'    : (2, 'Maximum card')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
    
    class_index = int(config.get_value('class')) 
    nsubsets = int(config.get_value('n'))   
    min_sup = float(config.get_value('minsup'))
    max_card = int(config.get_value('card'))
    
    C = float(config.get_value('c'))
    nloops = int(config.get_value('nloop'))
        
    for i in range(nsubsets):
        print('Test for case ...' + str(i))
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
        train_Y[train_Y < 1] = 0
        train_Y[train_Y >= 1] = 1
        
        test_X = test_data.get_X_in_binary_with(rel_train_X.item_dict)
        test_Y = test_data.get_Y_in_numeric_with(rel_train_Y.item_dict)
        test_Y[test_Y < 1] = 0
        test_Y[test_Y >= 1] = 1
    
        '''
        CORELS
        '''        
        print('Run CORELS....')
        corels_model = CorelsClassifier(C, n_iter=nloops, max_card=max_card, min_support=min_sup)
        corels_model.fit(train_X, train_Y)
        
        
        rf_pred_train = corels_model.predict(train_X)
        c1 = evaluateByF1(rf_pred_train, train_Y)
        
        rf_pred_test = corels_model.predict(test_X)
        c2 =  evaluateByF1(rf_pred_test, test_Y)
        print(corels_model.rl_.__str__())
        print('length:', len(corels_model.rl_.rules))
        print('train', c1)
        print('test', c2)
    
