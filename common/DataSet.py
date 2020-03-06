
# Transaction databases, each transaction is a set of items
import copy
import numpy as np

from scipy import stats

from collections import Counter

from common.RelationArray import RelationArray2D
from common.RelationArray import RelationArray1D

class DataSet:
    def __init__(self):
        self.current = 0
        self.train_data = []
        self.data_labels = []
        
    
    #def __iter__(self):
    #    return iter(self.train_data)
                
    '''
    Get number of transactions.
    '''
    def size(self):
        return len(self.train_data)
    
    '''
    Get the i-th transaction.
    '''
    def get_transaction(self, index):
        return self.train_data[index]
    
    '''
    Remove all transactions in data-set
    '''
    def clear(self):
        self.train_data.clear()
    
    '''
    Add new transaction into the end of data-set.
    '''    
    def add_transaction(self, t):
        return self.train_data.append(t) 
        
    
    '''
    Create a copy of data-set.
    '''
    def copy(self):
        data_set = DataSet()
        data_set.train_data = copy.deepcopy(self.train_data)
        data_set.data_labels = copy.deepcopy(self.data_labels)
        
        return data_set
    
    '''
    Remove a list of transactions based on their indices.
    '''
    def pop(self, indices):
        sorted_indices = sorted(indices, reverse = True)
        for i in sorted_indices:
            self.train_data.pop(i)
            self.data_labels.pop(i)
        
    '''
    Get number of distinguish classes in data-set
    '''
    def count_classes(self):
        return Counter(self.data_labels)
        
    '''
    Load data set from a file. The input file must be formated in CSV (comma separated)
    class_index is used in the case of data-set with labels. 
    '''
    def load(self, file_path, class_index = None, has_header = False):
        self.train_data = []
        if class_index != -1: self.data_labels = []
        
        with open(file_path, "r") as text_in_file:
            if has_header == True:
                text_in_file.readline()
                
            for line in text_in_file:
                transaction = [x.strip() for x in line.split(',')]
                transaction = list(filter(None, transaction))
                
                if (class_index is not None):
                    self.data_labels.append(transaction[class_index])
                    del transaction[class_index]
                    
                self.train_data.append(list(set(transaction)))
    
    '''
    Save transactions and their labels to a file.
    '''
    def save(self, file_path):
        with open(file_path, 'w') as file_writer:
            for i in range(self.size()):
                s = self.data_labels[i] + ','
                s += (','.join(self.train_data[i]))
                
                file_writer.write(s)
                file_writer.write('\n')

    '''
    Gets number of classes in data (if have).
    '''            
    def number_of_classes(self):
        if self.data_labels == None: return 0
        return len(set(self.data_labels))
    
    
    def get_X_in_binary_with(self, items_dict):
        n_items = len(items_dict)
        X = np.zeros((self.size(), n_items))
        
        k = 0
        for transaction in self.train_data:
            for item in transaction:
                if item not in items_dict:
                    continue
                i = items_dict[item]
                X[k, i] = 1.0
            k += 1
        return X
         
    def get_Y_in_numeric_with(self, classes_dict):
        Y = []
        for label in self.data_labels:
            if label not in classes_dict:
                print('not in classes')
                Y.append(-1)
            else:
                Y.append(classes_dict[label])
        return np.array(Y)
    
    '''
    Get unique items in the data-set.
    '''
    def get_items_dict_(self):
        attr_dict = {}
        #check existing data
        for transaction in self.train_data:
            for index in range (len(transaction)):
                item_name = transaction[index]
                if item_name not in attr_dict:
                    attr_dict[item_name] = True
        return attr_dict

    def get_X_in_binary(self):
        attr_dict = self.get_items_dict_()
         
        # Sort items in alphabet order.
        items_list = sorted(attr_dict.keys())
        attr_dict = {items_list[i] : i for i in range(len(items_list))}
        
        #Generate binary matrix (X_train) and array of labels(Y_train)
        X = self.get_X_in_binary_with(attr_dict)
        return  RelationArray2D(attr_dict, X)
    
    def get_Y_in_numeric(self):
         
        # Sort items and classes in alphabet order.
        classes_list = sorted(set(self.data_labels))
        classes_dict = {classes_list[i] : i for i in range(len(classes_list))}
        
        #Generate binary matrix (X_train) and array of labels(Y_train)
        Y = self.get_Y_in_numeric_with(classes_dict)        
        return RelationArray1D(classes_dict, np.array(Y))
        
            
    @staticmethod
    def write_relation_matrix_(matrix):
        with open('item_relation.csv', 'w') as file_writer:
            item_names = sorted(matrix.item_dict.keys())
            file_writer.write('o0o,')
            file_writer.write(','.join(item_names))
            file_writer.write('\n')
            for i in range(len(item_names)):
                file_writer.write(item_names[i] + ',')
                file_writer.write(','.join(str(x) for x in matrix.relation_matrix[i].tolist()))
                file_writer.write('\n')
        
    '''
    This method estimates relationship among items. There're two kinds of relationship
    - Correlation:including negative correlation (<= -0.3) and positive correlation (>= 0.3)
    - Cover: threshold 1.0, including cover (2) and covered (-2) 
    '''
    def items_relationship(self):
        
        print ('Computing item relation matrix...')
        
        X_train, _ = self.convert_2_binary_format()
        correlation_matrix, p_values = stats.spearmanr(X_train.relation_matrix.todense(), axis = 0)
        zeros_mask = (p_values <= 0.05).astype(int)
        negative_correlation = (correlation_matrix < -0.1).astype(int)
        negative_correlation = (correlation_matrix * negative_correlation * zeros_mask)
        
        positive_correlation = (correlation_matrix > 0.1).astype(int)
        positive_correlation = (correlation_matrix * positive_correlation * zeros_mask)
        
        relation_matrix = negative_correlation + positive_correlation
        
        a = RelationArray2D(X_train.item_dict, relation_matrix)
        DataSet.write_relation_matrix_(a)
        return a
    
    '''
    Split data into smaller data-sets. The first one takes rate(%) of data.
    '''
    def split_in_two(self, rate=0.85):
        n = len(self.train_data)
        x = np.arange(n)
        np.random.shuffle(x)
        
        m = int(rate * n)
        data_set_1 = DataSet()
        data_set_2 = DataSet()
        
        for i in range(n):
            j = x[i]
            if i < m: 
                data_set_1.train_data.append(self.train_data[j])
                data_set_1.data_labels.append(self.data_labels[j])
            else:
                data_set_2.train_data.append(self.train_data[j])
                data_set_2.data_labels.append(self.data_labels[j])
        return data_set_1, data_set_2
    
    '''
    Sampling data with replacement
    '''
    def samplize_with_replacement(self, rate = 0.5):
        n = self.size()
        m = int(rate * n)
    
        indices = np.random.choice(n, m)
        new_data_set = DataSet()
        for i in indices:
            new_data_set.train_data.append(self.train_data[i])
            new_data_set.data_labels.append(self.data_labels[i])
        return new_data_set
    
    '''
    Split data into k smaller data-sets.
    '''
    def split_in_k(self, k):
        parts = []
        n = len(self.train_data)
        x = np.arange(n)
        np.random.shuffle(x)
        
        for i in range(k):
            parts.append(DataSet())
        
        for i in range(n):
            j = i % k
            parts[j].train_data.append(self.train_data[x[i]])
            parts[j].data_labels.append(self.data_labels[x[i]])
            
        return parts
    def split_random_in_k(self, k):
        parts = []
        n = len(self.train_data)
        x = np.arange(n)
        
        for i in range(k):
            parts.append(DataSet())
        
        for i in range(n):
            j = np.random.randint(0, k)
            parts[j].train_data.append(self.train_data[x[i]])
            parts[j].data_labels.append(self.data_labels[x[i]])
            
        return parts
    
    '''
    Merge a list of datasets into one.
    '''
    @staticmethod
    def merge_datasets(datasets):
        a = DataSet()
        for p in datasets:
            a.train_data.extend(p.train_data)
            a.data_labels.extend(p.data_labels)
        return a
    
    '''
    Create a data-set that merges classes into transactions
    '''
    def create_dataset_without_class(self):
        data_set = DataSet()
        data_set.train_data = copy.deepcopy(self.train_data)
        for i in range(len(self.data_labels)):
            data_set.train_data[i].append(self.data_labels[i])
        return data_set
    
    
    '''
    Create two data-sets based on cross validation. 
    '''
    @staticmethod
    def create_datasets_by_crossvalidation(sub_datasets, i):
        a = sub_datasets[i]
        b = []
        for j in range(len(sub_datasets)):
            if i == j: continue
            b.append(sub_datasets[j])
        c = DataSet.merge_datasets(b)
        return a, c
    
    
    @staticmethod 
    def create_binary_transaction_dataset(X, items_dict, data_labels):
        data_set = DataSet()
        data_set.data_labels = copy.deepcopy(data_labels)
        
        nitems = len(items_dict)
        items_list = ['' for i in range(nitems)]
        for c, i in items_dict.items():
            items_list[i] = c 
        
        for t in X:
            nt = [items_list[i]+'@'+str(t[i]) for i in range(nitems)]
            data_set.add_transaction(nt)
            
        return data_set
                
        