'''
Created on 19 Mar 2018

@author: danhbuithi
'''
import sys

from common.CommandArgs import CommandArgs
from common.DataSet import DataSet

from rules_mining.RuleMiner import RuleMiner
from rule_based_classifiers.NetMMAC import NetMMAC
from rule_based_classifiers.MMAC import MMAC

from TestRuleBasedMethods import preprocessRuleFeatureDict

    
if __name__ == '__main__':
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'test'   : ('', 'Path of testing data file'),
                          'class'   : (0, 'Class index'),
                          'minsup'  : (0.1, 'Minimum support'),
                          'minconf' : (0.0, 'Minimum confidence'),
                          'format'  : ('spect', 'valid format of rules/item-sets'),
                          'out'     : ('', 'Path of output file'),
                          'nloop'   : (10000, 'Number of loops'),
                          'label'   : ('', 'Positive label'),
                          'option'  : ('net', 'Name of algorithm')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    class_index = int(config.get_value('class'))
    nloop = int(config.get_value('nloop'))
    
    min_sup = float(config.get_value('minsup'))
    min_conf = float(config.get_value('minconf'))
    rule_format = config.get_value('format')
    
      
    train_data = DataSet()
    train_data.load(config.get_value('train'), class_index)
    
    test_data = DataSet()
    test_data.load(config.get_value('test'), class_index)
    
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
    max_nrules = 2000
    print('max nrules ', max_nrules)
    if config.get_value('option') == 'net':
        interesting_learner = NetMMAC(train_args, m, coverage=3, max_rules=max_nrules)
    else:
        interesting_learner = MMAC(train_args, m, coverage=3, max_rules=max_nrules)
    
    solutions = interesting_learner.fit(max_iters=nloop)
    interesting_learner.save_solutions(config.get_value('out'), solutions)

