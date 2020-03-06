from scipy import stats
import numpy as np
from sklearn.metrics.classification import matthews_corrcoef
 
def compute_pearman(rank_matrix):
    return stats.spearmanr(rank_matrix)

def compute_pairwise_pearson(rank_matrix):
    _, m = rank_matrix.shape
    correlations = np.zeros((m, m))
    p_values = np.zeros((m, m))
    
    for i in range(m):
        correlations[i,i] = 1
        p_values[i,i] = 0
        for j in range(i + 1, m):
            r = stats.pearsonr(rank_matrix[:,i], rank_matrix[:,j])
            correlations[i,j] = r[0]
            correlations[j,i] = r[0]
            p_values[i,j] = r[1]
            p_values[j,i] = r[1]
    return correlations, p_values

def compute_pearson(y, rank_matrix):
    correlations = []
    _, m = rank_matrix.shape
    for i in range(m):
        correlations.append(stats.pearsonr(y, rank_matrix[:,i]))
    return correlations


def compute_matthews(y, rank_matrix):
    correlations = []
    _, m = rank_matrix.shape
    for i in range(m):
        f = matthews_corrcoef(y, rank_matrix[:, i])
        print (y)
        print(rank_matrix[:,i])
        print('-------------')
        correlations.append(f)
    return np.array(correlations)