'''
Created on 19 Nov 2018

The class represents an object which is a list of rule playing as a classifier. 
The rules have to be sorted; the rule going first in doing classification. 
This classifier supports two modes for prediction: using only the first rule that the sample satisfies its LHS or using multiple rules.

@author: danhbuithi
'''

import numpy as np 
from collections import Counter
from sklearn.metrics.classification import accuracy_score
import heapq

class RuleListClassifier(object):
    '''
    classdocs
    '''

    def __init__(self, rules, labels, default_class = None):
        '''
        Constructor
        '''
        self.rules = rules
        self.default_class = default_class
        self.labels = labels
        self.rule_index_dict = self._index_rule_list()
        
    def _index_rule_list(self):
        rule_index_dict = {}
        for i in range(len(self.rules)):
            r = self.rules[i]['r']
            for item in r.left_items:
                if item not in rule_index_dict:
                    rule_index_dict[item] = []
                rule_index_dict[item].append(i)
        return rule_index_dict
        
    def size(self):
        return len(self.rules) + 1       
    
    
    def myPrint(self):
        for rule_info in self.rules:
            print((rule_info['r'].serialize(), rule_info['ins'], rule_info['sup']))
        if self.default_class is not None:
            print(self.default_class)
        print('--------------------------')
        
    
    def _is_satisfied(self, x, k):
        #print(x)
        #self.myPrint()
        
        Q = []
        for item in x:
            if item not in self.rule_index_dict: continue
            for rule_id in self.rule_index_dict[item]:
                heapq.heappush(Q, rule_id)
                #print(rule_id)
        if len(Q)==0: return []
        
        selected_rules = []
        current_rule = heapq.heappop(Q)
        counter = len(self.rules[current_rule]['r'].left_items)-1
        
        while Q and len(selected_rules) < k:
            top_rule = heapq.heappop(Q)
            if top_rule == current_rule: counter -= 1
            else:
                if counter == 0: selected_rules.append(current_rule)
                current_rule = top_rule 
                counter = len(self.rules[current_rule]['r'].left_items)-1
        if counter == 0 and len(selected_rules) < k:
            selected_rules.append(current_rule)        
        #print(selected_rules)
        return selected_rules
            
        
    def _predictOne(self, x):
        selected_rules = self._is_satisfied(x, 1)
        if len(selected_rules) > 0:
            return self.rules[selected_rules[0]]['r'].right_items[0]
            
        if self.default_class is None: return 'unknown'
        return self.default_class['r']
     
    def predict(self, data_set):
        Y = []
        for i in range(data_set.size()):
            label = self._predictOne(data_set.get_transaction(i))
            Y.append(label)
        return Y
    
    def meanLength(self):
        total = 0
        n = self.size()
        for rule_info in self.rules:
            total += len(rule_info['r'].left_items)/n
        return total
    
    def meanSupport(self):
        total = 0
        for rule_info in self.rules:
            if (rule_info['sup'] < 1e-6):
                total += 1
        return total
    
    def cost(self, data):
        y_pred = self.predict(data)
        return 1 - accuracy_score(data.data_labels, y_pred)
    
    
    @staticmethod
    def globalDefaultClass(data):
        un_classified_classes = data.count_classes()
        default_class = max(un_classified_classes, key=un_classified_classes.get)
        return default_class, un_classified_classes[default_class]
        
    @staticmethod
    def localDefaultClass(data, indices):
        un_classified_classes = Counter([data.data_labels[i] for i in indices])
        default_class = max(un_classified_classes, key=un_classified_classes.get)
        return default_class, un_classified_classes[default_class]
    
    @staticmethod
    def getDefaultClass(cover_counter, data):
        default_class = None
        indices = np.where(cover_counter ==0)[0]
        if len(indices) > 0:
            default_class, _ = RuleListClassifier.localDefaultClass(data, indices)
        else:
            default_class, _ = RuleListClassifier.globalDefaultClass(data)
            
        return default_class
            
class SingleRuleBasedClassifier(RuleListClassifier):
    
    def __init__(self, rules, labels, default_class = None):
        '''
        Constructor
        '''
        RuleListClassifier.__init__(self, rules, labels, default_class)

class MMultiRuleBasedClassifier(RuleListClassifier):
    
    def __init__(self, rules, labels, default_class = None, coverage_thresh = 3):
        '''
        Constructor
        '''
        RuleListClassifier.__init__(self, rules, labels, default_class)
        self.coverage_threshold = coverage_thresh
    
        
    def _predictOne(self, x):
        selected_rules = self._is_satisfied(x, self.coverage_threshold)
        if len(selected_rules) == 0:
            if self.default_class is None: return 'unknown'
            return self.default_class['r']
        
        
        label_dict = {key: 0 for key in self.labels}
        for i in selected_rules:
            rule_info = self.rules[i]
            r = rule_info['r']
            f = rule_info['ins']
            label_dict[r.right_items[0]] = max(f,label_dict[r.right_items[0]])
            
        p = [(value, key) for key, value in label_dict.items()]
        return max(p)[1] 
    
        
class MultiRuleBasedClassifier(RuleListClassifier):
    
    def __init__(self, rules, labels, default_class = None, coverage_thresh = 3):
        '''
        Constructor
        '''
        RuleListClassifier.__init__(self, rules, labels, default_class)
        self.coverage_threshold = coverage_thresh
        
    def _compute_weights_per_class(self, x):
        selected_rules = self._is_satisfied(x, self.coverage_threshold)
        label_dict = {key: 0 for key in self.labels}
        
        if len(selected_rules) == 0:
            if self.default_class is None: return None
            label_dict[self.default_class['r']] = 1.0
        
        nlabels = len(self.labels)
        
        for i in selected_rules:
            rule_info = self.rules[i]
            r = rule_info['r']
            f = rule_info['ins']
            label_dict[r.right_items[0]] += f
            for k in self.labels: 
                if k != r.right_items[0]: label_dict[k] += (1-f)/(nlabels-1) 
            
        return label_dict
    
    def _predictOne(self, x):
        label_dict = self._compute_weights_per_class(x)
        if label_dict is None: return 'unknown'
        c = [(value, key) for key, value in label_dict.items()]    
        return max(c)[1]

    
    def cost(self, data, weights):
        y_pred = []
        for i in range(data.size()):
            x = data.get_transaction(i)
            label_dict = self._compute_weights_per_class(x)
            
            c = 1e-8
            if label_dict is not None:
                a = label_dict[data.data_labels[i]]
                b = np.array([t for t in label_dict.values()])
            
                c = a/np.sum(b)
                if c == 0: c = 1e-8
            
            y_pred.append(c)
            
        w = np.array([weights[y] for y in data.data_labels])
        return np.sum(-np.log(np.array(y_pred)) * w)/np.sum(w)
    
        
    def cost2(self, data):
        
        y_pred = []
        prediction_per_class = {c : [] for c in self.labels}
        
        for i in range(data.size()):
            x = data.get_transaction(i)
            label_dict = self._compute_weights_per_class(x)
            c = 1e-8
            if label_dict is not None :
                a = label_dict[data.data_labels[i]]
                b = np.array([t for t in label_dict.values()])
                
                c = a/np.sum(b)
                if c == 0: c = 1e-8
            y_pred.append(c)
            prediction_per_class[data.data_labels[i]].append(c)
            
        costs = [np.average(-np.log(np.array(x))) for x in prediction_per_class.values()]
        total_cost =  np.average(-np.log(np.array(y_pred)))

        return costs, total_cost