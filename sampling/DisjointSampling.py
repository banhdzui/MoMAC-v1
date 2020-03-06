'''
Created on Feb 28, 2017

@author: BanhDzui
'''

import numpy as np
import heapq

class DisjointSampling:
    
    '''
    This class is used to generate association rules according to DISJOINT algorithm. 
    This algorithm selects rules which are conflicted in interesting measures by themselves
    '''

    @staticmethod
    def execute(association_rules_dict, kBest, rank_matrix):
        selected_patterns = []
        ''' Compute mean and standard deviation of rankings'''
        std_for_patterns = np.std(rank_matrix, axis = 1)
        
        m = np.argmax(std_for_patterns)
        selected_patterns.append(association_rules_dict[m])
        association_rules_dict.pop(m)
        
        n = len(association_rules_dict)
        dij = np.zeros((n, n))
        
        for i in range(n):
            if i % 100 == 0:
                print(str(i))
            for j in range (0, i):
                delta = rank_matrix[i] - rank_matrix[j]
                mean_delta = np.mean(delta)
                std_delta = np.std(delta)
                
                dij[i][j] = mean_delta + std_delta
                dij[j][i] = -mean_delta + std_delta
                
        sum_dij = np.sum(dij, axis=0)
        largest_indexes = heapq.nlargest(kBest - 1, range(n), sum_dij.take) 
        for index in largest_indexes:
            selected_patterns.append(association_rules_dict[index])       
                            
        return selected_patterns         