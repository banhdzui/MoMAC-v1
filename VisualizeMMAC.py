'''
Created on 19 Mar 2018

@author: danhbuithi
'''
import sys
import numpy as np

from sklearn.metrics import f1_score


from common.CommandArgs import CommandArgs
from common.DataSet import DataSet

from rules_mining.RuleMiner import RuleMiner
from rules_mining.AssociationRule import AssociationRule
from rule_based_classifiers.NetMMAC import NetMMAC
from rule_based_classifiers.MMAC import MMAC


def evaluateByF1(y_pred, y_true):
    a = f1_score(y_true, y_pred, average='micro')
    b = f1_score(y_true, y_pred, average='macro')
    return (a, b)    
            
def printList(x):
    i = 0
    for c in x:
        if i % 4 == 0: print(' ')
        print(c)
        i += 1
        
    print('-----------------')
    
def separateRulesAndFeatures(rule_feature_dict):
    rules = []
    X = []
    for key, value in rule_feature_dict.items():
        rules.append(key)
        X.append(value)
    return rules, np.array(X)
    

def preprocessRuleFeatureDict(rule_feature_dict):
    '''
    Normalize feature using min-max scaler
    '''
    rules, features = separateRulesAndFeatures(rule_feature_dict)
    rule_full_list = [AssociationRule.string_2_rule(x) for x in rules]
    
   
    return rule_full_list, features, features[:, 0]
    
if __name__ == '__main__':
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'test'   : ('', 'Path of testing data file'),
                          'class'   : (None, 'Class index'),
                          'minsup'  : (0.1, 'Minimum support'),
                          'minconf' : (0.0, 'Minimum confidence'),
                          'format'  : ('spect', 'valid format of rules/item-sets'),
                          'sol'     : ('', 'Path of output file'),
                          'option'  : ('net', 'Selected algorithm')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    class_index = int(config.get_value('class'))
    train_data = DataSet()
    train_data.load(config.get_value('train'), class_index)
    
    test_data = DataSet()
    test_data.load(config.get_value('test'), class_index)
    
    min_sup = float(config.get_value('minsup'))
    min_conf = float(config.get_value('minconf'))
    rule_format = config.get_value('format')
    
    labels = sorted(train_data.count_classes().keys())
      
    '''
    Generate association rules
    '''
    rule_miner = RuleMiner(rule_format, train_data.create_dataset_without_class())
    rule_miner.generate_itemsets_and_rules(min_sup, min_conf)
    rule_feature_dict = rule_miner.load_rules_features_as_dictionary()
    print('#rules ', len(rule_feature_dict))
      
    freq_itemsets_dict = rule_miner.load_freq_itemset_dictionary()
    rule_list, rule_features, rule_supports = preprocessRuleFeatureDict(rule_feature_dict)
    print('#filtered rules ', len(rule_list))
    train_args = {'data': train_data, 'rule': rule_list, 'label': labels, 'feature': rule_features, 'sup': rule_supports}
    
    m = rule_features.shape[1]
    interesting_learner = None 
    if config.get_value('option') == 'net':
        interesting_learner = NetMMAC(train_args, m, coverage=3)
    else:
        interesting_learner = MMAC(train_args, m, coverage=3)
  
    variables, objectives, constraints = interesting_learner.load_solutions(config.get_value('sol'))
    
    
    netmaces_classifier = interesting_learner.visualize_solutions(variables, objectives, constraints)    

    print('Use learned measure (for training): ')
    
    bnn_pred_train = netmaces_classifier.predict(train_data)
    print(evaluateByF1(bnn_pred_train, train_data.data_labels))
    
      
    print('Use learned measure (for testing): ')
    bnn_pred_test = netmaces_classifier.predict(test_data)
    print(evaluateByF1(bnn_pred_test, test_data.data_labels))
