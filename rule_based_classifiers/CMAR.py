'''
Created on 06 Nov 2018

@author: danhbuithi
'''
import numpy as np
import heapq

from scipy.stats import chi2_contingency
from rule_based_classifiers.RuleListClassifier import MMultiRuleBasedClassifier


class CMAR(object):
    '''
    classdocs
    '''

    def __init__(self, ):
        '''
        Constructor
        '''
        
    '''
    Check whether rule r1 is more general than rule r2
    r1 is more general if its LHS is subset of r2's LHS 
    '''
    def _isLessGeneral(self, r1, r2):
        return set(r1.left_items) > set(r2.left_items)
    
    
    def isLessGeneral(self, r, rule_list):
        for rule_info in rule_list:
            if self._isLessGeneral(r, rule_info['r']):
                return True 
        return False
    
    
    @staticmethod
    def chi_squared(nleft, nright, nboth, ntransactions):
        nAB = nboth 
        n_notA_notB = ntransactions - (nleft + nright) + nboth
        n_A_notB = nleft - nboth 
        n_notA_B = nright - nboth 
        g, p, _, _ = chi2_contingency(np.array([[nAB, n_A_notB], [n_notA_B, n_notA_notB]]))
        return (g, p)
    
    @staticmethod 
    def max_chi_squared(nleft, nright, ntransactions):
        n_notleft = ntransactions-nleft 
        n_notright = ntransactions-nright
        
        e = 1.0/(nleft * nright) + 1.0/(nleft * n_notright) + 1.0/(n_notleft*nright) + 1.0/(n_notleft * n_notright)
        return ((min(nleft, nright) - nleft * nright/ntransactions)**2) * ntransactions * e 
        
    
    '''
    Select sufficient rules by prunning. There're 3 prunning steps 
    - Remove specific and lower ranking rules
    - Remove non-positively correlated rules
    - Remove based on database coverage
    '''
    def fit(self, train_data, rule_list, label_list, freq_itemset_dict,coverage_thresh = 4, diff_thresh = 0.01):
        cover_counter = np.zeros(train_data.size())
        ntransactions = freq_itemset_dict.ntransactions
         
        Q = []
        for x in rule_list:
            heapq.heappush(Q, (-x['ins'], -x['sup'], len(x['r'].left_items), x['r']))
    
        selected_rules = []
        counter = 1
        
        indices = np.where(cover_counter < coverage_thresh)[0]
        while Q and len(indices) > 0:
            rule_tuple = heapq.heappop(Q)
            r = rule_tuple[-1]
            
            counter += 1
            if counter % 5000 == 0: print(counter)
            
            if self.isLessGeneral(r, selected_rules): continue
            
            nleft, nright, nboth = freq_itemset_dict.get_frequency_tuple(r)
            g, p = CMAR.chi_squared(nleft, nright, nboth, ntransactions)
            if p > 0.05 or g < diff_thresh : continue
            
            maxg = CMAR.max_chi_squared(nleft, nright, ntransactions)
            
            satisfied_indices = []
            checker = False
            for i in indices:
                if r.is_satisfied(train_data.get_transaction(i)): 
                    satisfied_indices.append(i)
                    if train_data.data_labels[i] in r.right_items:
                        checker = True
                        
            if checker == True > 0:
                selected_rules.append({'r': r, 'chi': g, 'mchi': maxg, 'sup': -rule_tuple[1]})
                for j in satisfied_indices:
                    cover_counter[j] += 1
                    
            indices = np.where(cover_counter < coverage_thresh)[0]
        return MMultiRuleBasedClassifier([{'r': x['r'], 'ins': x['chi']**2/x['mchi'], 'sup':x['sup']} for x in selected_rules], 
                                           label_list, None, coverage_thresh)
        