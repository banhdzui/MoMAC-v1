'''
Created on May 3, 2017

@author: BanhDzui
'''
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

class RuleClusteringEngine(object):
    
    def __init__(self, eps, minpts, nThreads, sample_features, samples):
        self.eps = eps
        self.minpts = minpts
        self.nthreads = nThreads
        self.sample_features = sample_features
        self.samples = samples
        
        m, _ = self.sample_features.shape
        self.labels = [-1 for _ in range(m)]
        self.visisted = [False for _ in range(m)]
        
    def region_query(self, p):
        temp = self.sample_features[p,:]
        temp = np.reshape(temp, (1, -1))
        distance = cosine_distances(temp, self.sample_features)
        
        neighbors = np.where(distance <= self.eps)
        return neighbors[1].tolist()
    
    def expand_cluster(self, p, C, neighbors):
        self.labels[p] = C
        while len(neighbors) > 0:
            other_p = neighbors.pop(0)
            if self.visisted[other_p] == False:
                self.visisted[other_p] = True
                other_neighbors = self.region_query(other_p)
                
                if (len(other_neighbors) >= self.minpts):
                    neighbors.extend(other_neighbors)
                    tmp = list(set(neighbors))
                    neighbors.clear()
                    neighbors.extend(tmp)
                    
            if self.labels[other_p] == -1:
                self.labels[other_p] = C
                
         
    
    def run(self):
        C = -1
        m, _ = self.sample_features.shape
        for p in range(m):
            #if p % 50 == 0: print (p)
            print(str(p) + '###' + str(C))
            if self.visisted[p] == True: continue
            
            self.visisted[p] = True
            neighbors = self.region_query(p)
            if len(neighbors) >= self.minpts:
                C += 1
                self.expand_cluster(p, C, neighbors)
        
                
class RulesClustering(object):
    
    def __init__(self, features_matrix, samples_length):
        self.sample_features = features_matrix 
        self.samples_length = samples_length

        
    def run_dbscan(self, my_eps, my_min_samples, number_of_threads):
        print ('Doing clustering ....')
        db = RuleClusteringEngine(my_eps, 
                                  my_min_samples, 
                                  number_of_threads, 
                                  self.sample_features, 
                                  self.samples_length)
        db.run()
        
        n_clusters = len(set(db.labels))- (1 if -1 in db.labels else 0)
        n_noises = db.labels.count(-1)
        
        print('Number of clusters' + str(n_clusters))
        print('Number of noises' + str(n_noises))
    
        return n_clusters, db.labels
                
        