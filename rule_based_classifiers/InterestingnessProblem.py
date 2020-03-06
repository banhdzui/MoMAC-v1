'''
Created on 23 Oct 2019

@author: danhbuithi
'''
import numpy as np
from platypus import Problem, Real


class InterestingnessProblem(Problem):
    
    def __init__(self, host):
        
        self.host = host 
        n = host.number_of_params()
        nconstraints = len(host.train_args['label']) + 1
        print('#number of constraints ', nconstraints)
        super(InterestingnessProblem, self).__init__(n, 2, nconstraints)
        
        self.types[:] = [Real(-10, 10) for _ in range(n)]
        self.constraints[:] = "<=0"

        
    def evaluate(self, solution):
        w = np.array(solution.variables[:])
        o, c = self.host.compute_cost(w)
        solution.objectives[:] = o
        solution.constraints[:] = c 
