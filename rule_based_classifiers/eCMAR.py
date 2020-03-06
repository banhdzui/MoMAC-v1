'''
Created on 07 Nov 2018

@author: danhbuithi
'''
import numpy as np 
import heapq

from rule_based_classifiers.RuleListClassifier import RuleListClassifier, MultiRuleBasedClassifier

class eCMAR(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        
    '''
    Create a heap to keep rules which are candidates for classifier.
    Each element is a tuple consisting of interestingness and rule.
    '''
    def _createQueue(self, rule_list):
        Q = []
        for x in rule_list:
            heapq.heappush(Q, (-x['ins'], -x['sup'], len(x['r'].left_items), x['r']))
        return Q
    
   
        
    def _getPrediction(self, predict_cache, default_class):
        y_pred = []
        for x in predict_cache:
            key = max(x, key=x.get)
            if x[key] == 0: y_pred.append(default_class)
            else: y_pred.append(key) 
        return y_pred
    
    
            
    '''
    Select sufficient rules by prunning based on database coverage
    '''
    def fit(self, train_data, rule_list, label_list,coverage_thresh = 3):
        ntransactions = train_data.size()
        cover_counter = np.zeros(ntransactions)
        Q = self._createQueue(rule_list)
        
        selected_rules = []
        
        while Q:
            rule_tuple = heapq.heappop(Q)
            f = -rule_tuple[0] 
            r = rule_tuple[-1]
        
             
            indices = np.where(cover_counter < coverage_thresh)[0]
            if len(indices) == 0: break
            
            satisfied_indices = []
            for i in indices:
                if r.is_satisfied(train_data.get_transaction(i)): 
                    satisfied_indices.append(i)

            
            
            if len(satisfied_indices) > 0:
                for j in satisfied_indices:
                    cover_counter[j] += 1
                    
                default_class = RuleListClassifier.getDefaultClass(cover_counter, train_data)
                selected_rules.append(({'r': r, 'ins': f, 'sup':-rule_tuple[1]}, default_class))
                
        model = [x[0] for x in selected_rules]
        default_class = selected_rules[-1][1]
        
        
        return MultiRuleBasedClassifier(model, label_list, {'r':default_class, 'ins':1.0}, coverage_thresh)
    
