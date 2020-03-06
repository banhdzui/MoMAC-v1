'''
Created on 06 Apr 2018

@author: danhbuithi
'''

import numpy as np

from cma import CMAEvolutionStrategy    
from common.ActivateFunctions import sigmoid
from rule_based_classifiers.eCMAR import eCMAR
from collections import Counter


class MACES(object):
    '''
    classdocs
    '''
    
    def __init__(self, train_args, class_weight = False, my_lambda=0.01, k = 1):
        self.my_lambda = my_lambda
        self.train_args = train_args
        
        self.weights = self.getClassWeights(class_weight)
        print(self.weights)
        self.K = k
        self.C = 1
        
        
    def getClassWeights(self, use_weight):
        weights = {label: 1 for label in self.train_args['label']}
        if use_weight == True:
            nlabels = len(self.train_args['label'])
            nsamples = self.train_args['data'].size()
            
            countings = Counter(self.train_args['data'].data_labels)
            for k, v in countings.items():
                weights[k] = nsamples/(nlabels * v)
                
        return weights 
                
    def computeInterestingness(self, w):
        b = w[0]
        #wt = np.reshape(w[1:], (-1, 1))
        wt = np.reshape(w[1:], (-1, self.K))
        
        scores = np.max(sigmoid(np.dot(self.train_args['feature'], wt) + b), axis = 1)
        
        rule_list = self.train_args['rule']
        rule_supports = self.train_args['sup']
    
        return [{'r': rule_list[i], 
                 'ins': scores[i], 
                 'sup': rule_supports[i]
                 } 
                 for i in range(len(rule_list))]
       
    def createClassifier(self, w, coverage):
            
        rule_list = self.computeInterestingness(w)
        return eCMAR().fit(self.train_args['data'], rule_list, self.train_args['label'], coverage_thresh=coverage)
    
    def createOriginClassifier(self, coverage):
        rule_list = self.train_args['rule']
        rule_supports = self.train_args['sup']
        rule_scores = self.train_args['feature']
        new_rule_list = [{'r': rule_list[i], 'ins':rule_scores[i][2], 'sup': rule_supports[i]} for i in range(len(rule_list))]
        return eCMAR().fit(self.train_args['data'], new_rule_list, self.train_args['label'], coverage_thresh=coverage)
    
    @staticmethod
    def cost(w, *args):
        ranker, my_beta, coverage, is_debug = args
        return ranker.nmcost(w, my_beta, coverage, is_debug)

    
    def nmcost(self, w, my_beta, coverage, is_debug):
        source_model = self.createClassifier( w, coverage)
        score1 = source_model.cost(self.train_args['data'], self.weights)
        
        b = self.my_lambda * np.linalg.norm(w)**2
        c = my_beta * source_model.size()#/self.C    
    
        if is_debug == True:
            print(score1, source_model.size())
        
        return score1 + b + c

   
    def randomw0(self, nfeatures):
        
        w0 = np.random.randn(nfeatures * self.K + 1) * 0.01
        w0[0] = 0
        return w0 
            
    def fit(self, my_beta, coverage=3, max_iters=50):
       
        #while(True):
        print('lambda = ' ,self.my_lambda)
        print('beta = ' , my_beta)
        
        origin_cmar = self.createOriginClassifier(coverage)
        self.C = origin_cmar.size()
        print('original cmar size', self.C)
        
        my_args = (self, my_beta, coverage, False)
        
        nfeatures = self.train_args['feature'].shape[1]
        w0 = self.randomw0(nfeatures)
        
        sigma_value = 0.2
        print('sigma0 ', sigma_value)
        es = CMAEvolutionStrategy(w0, sigma0 = sigma_value, inopts ={'maxiter': max_iters, 'popsize': 40})
 
        while not es.stop():
            solutions = es.ask()
            fitnesses = [MACES.cost(x, *my_args) for x in solutions]
            es.tell(solutions, fitnesses)
            es.disp()
            
            self.nmcost(es.result[0], my_beta, coverage, is_debug=True)
                            
        final_model = self.createClassifier(es.result[0], coverage)
        return final_model
            