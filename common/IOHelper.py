import json
import numpy as np
from numpy import double

class IOHelper:

    @staticmethod
    def write_file_in_lines(file_name, data, header = None):
        with open(file_name, "w") as text_file:
            if header is not None:
                text_file.write(header)
                text_file.write('\n')
            for transaction in data:
                text_file.write(transaction)
                text_file.write('\n')
    
    @staticmethod        
    def read_file_in_lines(inputfile, has_header = False):
        data = []
        with open(inputfile, "r") as text_file:
            file_iter = iter(text_file)
            if has_header == True:
                next(file_iter)
            
            for line in file_iter:
                data.append(line.strip())
        return data
    
    @staticmethod
    def read_ranking_file(input_file):
        patterns = []
        ranking = []
        k = 0
        with open(input_file, "r") as text_file:
            for line in text_file:
                subStrings = line.split(';')
                rule_key = subStrings[0].strip()
                patterns.append(rule_key)
                ranking.append([])
                for v in subStrings[1:]:
                    r = int(v)
                    ranking[k].append(r)
                
                k += 1
                #if k % 1000 == 0: print(str(k))
        return patterns, np.array(ranking)
    
    @staticmethod 
    def write_in_json(file_name, o):
        with open (file_name, 'w') as text_file:
            json.dump(o, text_file)
            
    @staticmethod        
    def load_json_object(file_name):
        with open(file_name, 'r') as text_file:
            o = json.load(text_file)
            return o
    
    @staticmethod    
    def write_matrix(file_name, matrix):
        with open(file_name, "w") as text_file:
            for line in matrix:
                text_file.write(','.join(str(x) for x in line.tolist()))
                text_file.write('\n')
       
    @staticmethod    
    def write_matrix_and_labels(file_name, matrix, labels):
        with open(file_name, "w") as text_file:
            text_file.write('o0o,')
            text_file.write(','.join(labels))
            text_file.write('\n')
            i = 0
            for line in matrix:
                text_file.write(labels[i] + ',')
                text_file.write(','.join(str(x) for x in line.tolist()))
                text_file.write('\n')
                i += 1
                
    @staticmethod
    def load_binary_data(input_file):
        with open(input_file, 'r') as text_file:
            temp = []
            header_line = next(text_file)
            feature_names = [x.strip() for x in header_line.split(',')]
            feature_names.pop(0)
            
            for line in text_file:
                if line == '': continue
                temp.append([float(x) for x in line.split(',')])
            data = np.array(temp)
            
            X = data[:,1:]
            Y = data[:,0]
            return feature_names, X, Y
        
        return None
    @staticmethod
    def write_tuple_list(file_name, tuples_list):
        with open(file_name, 'w') as writer:
            for item in tuples_list:
                writer.write(str(item))
                writer.write('\n')
              
    @staticmethod  
    def load_data(file_name):
        X = []
        Y = []
        with open(file_name, 'r') as file_reader:
            for line in file_reader: 
                _, c, features = json.loads(line)
                X.append(features)
                
                t = 0
                if c == 'yes': t = 1
                Y.append(t)
                
        return np.array(X).astype(double), np.array(Y)