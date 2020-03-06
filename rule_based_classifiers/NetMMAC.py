'''
Created on 06 Apr 2018

@author: danhbuithi
'''

import numpy as np
from common.ActivateFunctions import sigmoid
from rule_based_classifiers.MMAC import MMAC


class NetMMAC(MMAC):
    '''
    classdocs
    '''
    
    def __init__(self, train_args, ninputs, hidden_layers=[8, 4],
                 coverage = 3, 
                 class_weight = False,
                 max_rules = 250):
        
        MMAC.__init__(self, train_args, ninputs, coverage, class_weight, max_rules)
        self.hidden_layers = hidden_layers
        
    
    def number_of_params(self):
        n = 0 
        nprevious = self.ninputs
        for k in self.hidden_layers:
            n += (nprevious * k + 1)
            nprevious = k 
        n += (self.hidden_layers[-1] + 1)
        return n 
    
    def _collect_net_weights(self, w):
        nprevious = self.ninputs
        start = 0
        
        Ws = []
        for k in self.hidden_layers:
            nparams = nprevious * k
            end = start + nparams
            #print(start, end)
            W0 = np.reshape(w[start: end], (nprevious, k))
            b0 = w[end]
            Ws.append((W0, b0))
            start = end + 1
            nprevious = k
        '''
        Do for output layers
        '''
        W0 = np.reshape(w[start:-1], (-1,1))
        b0 = w[-1]
        Ws.append((W0, b0))
        return Ws 
            
    def computeInterestingness(self, w):
        X = self.train_args['feature']
        Ws = self._collect_net_weights(w)
        
        scores = X 
        for W0, b0 in Ws:
            scores = sigmoid(np.dot(scores, W0) + b0)
        
        return self._get_interestingness_list(scores)
        