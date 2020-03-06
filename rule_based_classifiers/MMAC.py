'''
Created on 06 Apr 2018

@author: danhbuithi
'''

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from platypus.algorithms import NSGAII
from platypus.core import nondominated

from common.ActivateFunctions import sigmoid
from rule_based_classifiers.eCMAR import eCMAR
from rule_based_classifiers.InterestingnessProblem import InterestingnessProblem



class MMAC(object):
    '''
    classdocs
    '''
    
    def __init__(self, train_args, ninputs, coverage = 3, class_weight = False, max_rules = 250):
        self.train_args = train_args    
        self.weights = self.getClassWeights(class_weight)
    
        self.ninputs = ninputs
        print('#features ', ninputs)
        
        self.coverage = coverage
        self.max_rules = max_rules
        
    def getClassWeights(self, use_weight):
        weights = {label: 1 for label in self.train_args['label']}
        if use_weight == True:
            nlabels = len(self.train_args['label'])
            nsamples = self.train_args['data'].size()
            
            countings = Counter(self.train_args['data'].data_labels)
            for k, v in countings.items():
                weights[k] = nsamples/(nlabels * v)
                
        return weights 
    
    def number_of_params(self):
        return self.ninputs+1
        
    
    def _get_interestingness_list(self, scores):
        rule_list = self.train_args['rule']
        rule_supports = self.train_args['sup']
    
        return [{'r': rule_list[i], 
                 'ins': scores[i,0], 
                 'sup': rule_supports[i]
                 } 
                 for i in range(len(rule_list))]
                
    def computeInterestingness(self, w):
        X = self.train_args['feature']
        W0 = np.reshape(w[0:-1], (-1, 1))
        b0 = w[-1]
        
        scores = sigmoid(np.dot(X, W0) + b0) 
        return self._get_interestingness_list(scores)
        
       
    def createClassifier(self, w):
            
        rule_list = self.computeInterestingness(w)
        return eCMAR().fit(self.train_args['data'], 
                           rule_list, 
                           self.train_args['label'], 
                           coverage_thresh=self.coverage)
    
    
    def compute_cost(self, w):
        source_model = self.createClassifier( w)
        classifier_errors, total_error = source_model.cost2(self.train_args['data'])
        model_size = source_model.size()
        
        constraints = [x - 1.0 for x in classifier_errors]
        constraints.append(model_size-self.max_rules)
        return [total_error, model_size], constraints
            
            
            
    def fit(self, max_iters=10000):
       
        #print(len(self.train_args['rule']))
        start = time.time()
        
        algorithm = NSGAII(InterestingnessProblem(self))
        algorithm.run(max_iters)
    
        
        end = time.time()
        print('execution time for interestingness learning: ' , end - start)
        non_dominated_solutions = nondominated(algorithm.result)
        feasible_solutions = [s for s in non_dominated_solutions if (s.feasible)]
        print('# feasible solutions ', len(feasible_solutions))
        return feasible_solutions
        
    def save_solutions(self, file_name, solutions):
        variables = []
        constraints = []
        objectives = []
        for x in solutions:
            variables.append(x.variables[:])
            constraints.append(x.constraints[:])
            objectives.append(x.objectives[:])
                        
        
        with open(file_name, 'w') as file_writer:
            s = json.dumps(variables)
            file_writer.write(s)
            file_writer.write('\n')
            s = json.dumps(constraints)
            file_writer.write(s)
            file_writer.write('\n')
            s = json.dumps(objectives)
            file_writer.write(s)
            file_writer.write('\n')
    
    def load_solutions(self, file_name):
        with open(file_name, 'r') as file_reader:
            s = next(file_reader)
            variables = json.loads(s)
            
            s = next(file_reader)
            constraints = json.loads(s)
            
            s = next(file_reader)
            objectives = json.loads(s)
            return variables, objectives, constraints
            
    def visualize_solutions(self, variables, objectives, constraints):
        anchor_values = []
        nsolutions = len(objectives)
        print('#sol', nsolutions)
        for i in range(nsolutions): 
            anchor_values.append((objectives[i][0], objectives[i][-1]))
            
        sorted_indices = np.argsort(np.array(anchor_values, dtype=[('x', 'f4'), ('y', 'f4')]), order=('x','y'))   
        x = []
        y = []
        Z = []
        for i in sorted_indices:
            x.append(objectives[i][0])
            y.append(objectives[i][-1])
            Z.append([a for a in constraints[i]])

        x = np.array(x)
        y = np.array(y)
        Z = np.array(Z)
        
        _, ax1 = plt.subplots()
        ax1.plot(x, y, '-o')
        ax1.set_xlabel('error')
        ax1.set_ylabel('#rules')
        
        i = 0
        
        for a,b in zip(x, y): 
            ax1.text(a, b, str(sorted_indices[i]))
            i += 1
            
        ax2 = ax1.twinx()
        ax2.set_ylabel('class error')
        label_list = self.train_args['label']
        class_counter = Counter(self.train_args['data'].data_labels)
        for i in range(len(label_list)):
            color = np.random.rand(3,)
            c = label_list[i]
            ax2.plot(x, Z[:,i], color = color, label=label_list[i]+'-'+str(class_counter[c]))
            j = 0
            for a,b in zip(x, Z[:,i]): 
                ax2.text(a, b, str(sorted_indices[j]))
                j += 1
            
        plt.legend(loc='upper right')
        
        plt.show()
        
        j = int(input('Please choose the solution'))
        w = np.array(variables[j])
        final_model = self.createClassifier(w)
        print(objectives[j][0], final_model.size(), final_model.meanSupport())
        
        
        return final_model
    
    '''
    def visualize_solutions(self, variables, objectives, constraints):
        anchor_values = []
        nsolutions = len(objectives)
        print('#sol', nsolutions)
        for i in range(nsolutions): 
            anchor_values.append(objectives[i][0])
            
        sorted_indices = np.argsort(anchor_values)    
        X = []
        Y = []
        for i in sorted_indices:
            X.append([x for x in objectives[i]])
            Y.append([y for y in constraints[i]])
        X = np.array(X)
        Y = np.array(Y)
        print(Y.shape)
        print(X.shape)
        
        t = np.arange(len(sorted_indices))
        _, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('index')
        ax1.set_ylabel('#rules', color = color)
        plt.xticks(t, sorted_indices)
        ax1.plot(t, X[:, -1])
        
        ax2 = ax1.twinx()
        ax2.plot(t, X[:, 0], color = 'tab:red')
            
        
    
        ax3 = ax1.twinx()
        label_list = self.train_args['label']
        class_counter = Counter(self.train_args['data'].data_labels)
        for i in range(len(label_list)):
            #ax2 = ax1.twinx()
            color = np.random.rand(3,)
            c = label_list[i]
            ax3.plot(t, Y[:,i], color = color, label=label_list[i]+'-'+str(class_counter[c]))
        
        plt.legend(loc='upper right')
        plt.show()
        
        j = int(input('Please choose the solution'))
        w = np.array(variables[j])
        final_model = self.createClassifier(w)
        print(objectives[j][0], final_model.size(), final_model.meanSupport())
        
        return final_model
    '''