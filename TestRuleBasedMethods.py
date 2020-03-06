'''
Created on 19 Mar 2018

@author: danhbuithi
'''
import sys
import numpy as np

from sklearn.metrics import f1_score
from sklearn.ensemble.forest import RandomForestClassifier


from common.CommandArgs import CommandArgs
from common.DataSet import DataSet

from rule_based_classifiers.CBA import CBA
from rule_based_classifiers.CMAR import CMAR

from rules_mining.RuleMiner import RuleMiner
from rules_mining.AssociationRule import AssociationRule

from rule_based_classifiers.MACES import eCMAR
from objective_measures.Interestingness import ObjectiveMeasure as om
from sklearn.tree.tree import DecisionTreeClassifier
    

def evaluateByF1(y_pred, y_true):
    a = f1_score(y_true, y_pred, average='micro')
    return a
             
def examineTraditionalMeasure(rule_list, freq_itemsets_dict, train_data, test_data, label_list):
        
    print('For traditional measure...')
    
    measures = [om.confidence, om.coverage, om.prevalence, om.recall, om.specificity, 
                    om.classificationError, om.lift, om.leverage, om.change_of_support, 
                    om.jaccard, om.certainty_factor, 
                    om.klosgen, om.weighting_dependency, 
                    om.jmeasure, 
                    om.one_way_support, om.two_ways_support, 
                    om.piatetsky_shapiro,
                    om.information_gain, om.least_contradiction, 
                    om.counter_example_rate, om.zhang]

    rule_and_measures = []
    ntransactions = freq_itemsets_dict.ntransactions
    
    for rule in rule_list:
        interestingness = []
        lhs_frequency, rhs_frequency, both_frequency = freq_itemsets_dict.get_frequency_tuple(rule)
                
        for index in range(len(measures)):
            value = measures[index](lhs_frequency/ntransactions, 
                                    rhs_frequency/ntransactions, 
                                    both_frequency/ntransactions)
            interestingness.append(value)
        rule_and_measures.append((rule, interestingness, both_frequency/ntransactions))
        
    test_result = []
    train_result = []
    for i in range(len(measures)):
        rule_and_score_list = [{'r': r, 'ins': v[i], 'sup': s} for r, v, s in rule_and_measures]
        
        
        cba_classifier = CBA().fit(train_data, rule_and_score_list, label_list)
        cba_pred_test = cba_classifier.predict(test_data)
        test_result.append(('CBA', evaluateByF1(cba_pred_test, test_data.data_labels)))
        
        print('CBA : ', cba_classifier.size(), 'rules')
        cba_pred_train = cba_classifier.predict(train_data)
        train_result.append(('CBA', evaluateByF1(cba_pred_train, train_data.data_labels)))
        
            
        cmar_classifier = CMAR().fit(train_data, rule_and_score_list, label_list, freq_itemsets_dict, coverage_thresh=3, diff_thresh=0.01)
        #cmar_classifier.myPrint()
        cmar_pred_test = cmar_classifier.predict(test_data)
        test_result.append(('CMAR', evaluateByF1(cmar_pred_test, test_data.data_labels)))
        
        print('CMAR : ', cmar_classifier.size(), 'rules')
        cmar_pred_train = cmar_classifier.predict(train_data)
        train_result.append(('CMAR', evaluateByF1(cmar_pred_train, train_data.data_labels)))
            
        print('----------------------')
    return train_result, test_result
    
            
def printList(x):
    i = 0
    for c in x:
        if i % 2 == 0: print(' ')
        print(c)
        i += 1
        
    print('-----------------')
    
def save_matrix_in_csv(file_name, A):
    with open(file_name, 'w') as file_writer:
        for row in A:
            file_writer.write(','.join([str(x) for x in row]))
            file_writer.write('\n')
    
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
                          'test'   : ('', 'Path of training data file'),
                          'class'   : (None, 'Class index'),
                          'minsup'  : (0.1, 'Minimum support'),
                          'minconf' : (0.0, 'Minimum confidence'),
                          'format'  : ('spect', 'valid format of rules/item-sets'),
                          'n'       : (5, 'Number of sub-learning sets'),
                          'nloop'   : (100, 'Number of loops')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    class_index = int(config.get_value('class'))
    min_sup = float(config.get_value('minsup'))
    min_conf = float(config.get_value('minconf'))
    rule_format = config.get_value('format')
    
    nsubsets = int(config.get_value('n'))
    nloop = int(config.get_value('nloop'))    
    
    for i in range(nsubsets):
        print('Test for case ...' + str(i))
        train_data = DataSet()
        train_data.load(config.get_value('train')+'.'+str(i), class_index)
        print('#transactions', train_data.size())
        test_data = DataSet()
        test_data.load(config.get_value('test')+'.'+str(i), class_index)
    
        labels = sorted(train_data.count_classes().keys())
        
        '''
        Convert data into binary
        '''
        rel_train_X = train_data.get_X_in_binary()
        rel_train_Y = train_data.get_Y_in_numeric()
        
        test_X = test_data.get_X_in_binary_with(rel_train_X.item_dict)
        test_Y = test_data.get_Y_in_numeric_with(rel_train_Y.item_dict)
        
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
        
        '''
        Running CBA and CMAR
        '''
        train_result = []
        test_result = []
        train_result, test_result = examineTraditionalMeasure(rule_list, freq_itemsets_dict, train_data, test_data, labels)
        train_args = {'data': train_data, 'rule': rule_list, 'label': labels, 'feature': rule_features, 'sup': rule_supports}
        
        '''
        Random forest
        '''
                
        print('Run random forest....')
        rr_model = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=1)
        rr_model.fit(rel_train_X.relation_matrix, rel_train_Y.values)
        
        rf_pred_train = rr_model.predict(rel_train_X.relation_matrix)
        train_result.append(('RF', evaluateByF1(rf_pred_train, rel_train_Y.values)))
        
        rf_pred_test = rr_model.predict(test_X)
        test_result.append(('RF', evaluateByF1(rf_pred_test, test_Y)))
        
        
        print('Run decision tree....')
        id3_model = DecisionTreeClassifier(max_depth=10, random_state=1)
        id3_model.fit(rel_train_X.relation_matrix, rel_train_Y.values)
        
        id3_pred_train = id3_model.predict(rel_train_X.relation_matrix)
        train_result.append(('ID3', evaluateByF1(id3_pred_train, rel_train_Y.values)))
        
        id3_pred_test = id3_model.predict(test_X)
        test_result.append(('ID3', evaluateByF1(id3_pred_test, test_Y)))
        
        print('Performance of CBA and CMAR with different measures:')
        printList(train_result)
        printList(test_result)
        
    
    