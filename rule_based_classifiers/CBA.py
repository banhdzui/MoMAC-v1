'''
Created on 07 Nov 2018

@author: danhbuithi
'''
import heapq
import numpy as np
from rule_based_classifiers.RuleListClassifier import RuleListClassifier,\
    SingleRuleBasedClassifier


class CBA(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    '''
    Create a heap to keep rules which are candidates for classifier.
    Each element is a tuple consisting of confidence, support and rule.
    '''
    def _createQueue(self, rule_list):
        Q = []
        for x in rule_list:
            heapq.heappush(Q, (-x['ins'], -x['sup'], len(x['r'].left_items), x['r']))         
        return Q
            
    '''
    Build classifier (a list of rules) by pruning rules
    '''
    def fit(self, train_data, rule_list, labels):
        selected_rules = []
        Q = self._createQueue(rule_list)
        remove_markers = np.zeros(train_data.size())
        
        error_by_rules = 0.0
        indices = np.where(remove_markers == 0)[0]
        while Q and len(indices) > 0:
            
            rule_tuple = heapq.heappop(Q)
            rule = rule_tuple[-1]
            
            
            is_marked = False
            local_error_by_rule = 0.0
            
            '''
            Find transactions satisfy the rule
            '''
            satisified_indices = []
            for i in indices:
                d = train_data.get_transaction(i)
                if rule.is_satisfied(d):
                    satisified_indices.append(i)
                    if train_data.data_labels[i] in rule.right_items:
                        is_marked = True
                    else:
                        local_error_by_rule += 1
            '''
            If there's at least a transaction be classified correctly, then choose the rule
            '''
            if is_marked == False: continue
            
            error_by_rules += local_error_by_rule
            for i in satisified_indices:
                remove_markers[i] = 1 
            
            '''
            Update un-classified sample list
            '''
            indices = np.where(remove_markers == 0)[0]
            
            
            '''
            Select default class in the case this rule is the last rule in selection.
            '''
            default_class_count = 0
            default_class_name = None 
            if len(indices) > 0:
                default_class_name, default_class_count = RuleListClassifier.localDefaultClass(train_data, indices)
            
                
            '''
            Add rule and its probability into the selected list
            '''
            adding_rule = ({'r':rule, 'ins':-rule_tuple[0], 'sup':-rule_tuple[1]}, 
                           {'r': default_class_name, 'ins': 1.0}, 
                           default_class_count + error_by_rules)
            selected_rules.append(adding_rule)
     
        '''
        Finalize the selected rules --> create model.
        '''
        p = selected_rules.index(min(selected_rules, key = lambda x: x[2]))
        model = [x[0] for x in selected_rules[:p+1]]
        default_class = selected_rules[p][1]
                
        '''
        If the selected rules cover whole data set, then choose the majority class as default class.
        '''    
        if (default_class['r'] is None):
            temp_default_class, _ = RuleListClassifier.globalDefaultClass(train_data)
            default_class = {'r':temp_default_class, 'ins': 1.0}
        
        return SingleRuleBasedClassifier(model, labels, default_class)
        
            