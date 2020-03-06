'''
Created on 07 Nov 2018

@author: danhbuithi
'''
import numpy as np 
import heapq

from rule_based_classifiers.RuleListClassifier import RuleListClassifier, MultiRuleBasedClassifier

class CCMAR(object):
    '''
    classdocs
    '''


    def __init__(self, max_nrules = 150):
        '''
        Constructor
        '''
        self.max_nrules = max_nrules
        
    '''
    Create a heap to keep rules which are candidates for classifier.
    Each element is a tuple consisting of interestingness and rule.
    '''
    def _createQueue(self, rule_list):
        Q = []
        for x in rule_list:
            heapq.heappush(Q, (-x['ins'], -x['sup'], len(x['r'].left_items), x['r']))
        return Q
    
    def _update_prediction_cache(self, data, cache, r, f):
        c = r.right_items[0]
        for i in range(data.size()):
            x = data.get_transaction((i))
            if r.is_satisfied(x):
                cache[i][c] += f 
                n = len(cache[i])
                for key in cache[i].keys():
                    if key != c: cache[i][key] += (1.0-f)/(n-1.0)
                
    
    def _cost(self, data, predict_cache, default_class):
        y_pred = []
        ntransactions = data.size()
        for i in range(data.size()):
            label_dict = predict_cache[i]
            b = np.sum(np.array([t for t in label_dict.values()]))
            if b == 0: 
                label_dict[default_class] = 1.0
                b = 1.0
            
            a = label_dict[data.data_labels[i]]
            c = a/b
            if c == 0: c = 1e-8
        
            y_pred.append(c)
        return np.sum(-np.log(np.array(y_pred)))/ntransactions
        
    '''
    Select sufficient rules by prunning based on database coverage
    '''
    def fit(self, train_data, rule_list, label_list):
        ntransactions = train_data.size()
        Q = self._createQueue(rule_list)
        default_class, _ = RuleListClassifier.globalDefaultClass(train_data)
        
        
        predict_cache = [{key: 0 for key in label_list} for _ in range(ntransactions)]
        selected_rules = []
        
        while Q and len(selected_rules) < self.max_nrules:
            rule_tuple = heapq.heappop(Q)
            f = -rule_tuple[0] 
            r = rule_tuple[-1]
            self._update_prediction_cache(train_data, predict_cache, r, f)
            cost = self._cost(train_data, predict_cache, default_class)
            
            #print(r.serialize(), cost)
            selected_rules.append(({'r': r, 'ins': f, 'sup':-rule_tuple[1]}, cost))
            
        p = selected_rules.index(min(selected_rules, key = lambda x: x[1]))
        #print('select position ', p)
        model = [x[0] for x in selected_rules[:p+1]]
        
        
        return MultiRuleBasedClassifier(model, label_list, {'r':default_class, 'ins':1.0}, -1)
    
