'''
Created on Feb 28, 2017

@author: BanhDzui
'''

import random

class RandomSplitter(object):
            
    @staticmethod
    def split(input_file, sampling_rate, has_header = False):
        selected_lines = []
        others = []
        
        header = None
        with open(input_file, "r") as text_file:
            if has_header == True:
                header = next(text_file)
                
            for line in text_file:
                r = random.uniform(0, 1)
                if r <= sampling_rate: 
                    selected_lines.append(line.strip())
                else:
                    others.append(line.strip())
        return selected_lines, others, header.strip()
    
    @staticmethod
    def splitIndexes(input_file, sampling_rate, has_header = False):
        selected_lines = []
        with open(input_file, "r") as text_file:
            if has_header == True:
                next(text_file)
            k = 0
            for line in text_file:
                if line == '': continue
                r = random.uniform(0, 1)
                if r <= sampling_rate: 
                    selected_lines.append(k)
                k+= 1
        return selected_lines
    
    @staticmethod
    def splitFile(input_file, selected_indexes, has_header = False):
        selected_lines = []
        others = []
        
        header = None
        with open(input_file, "r") as text_file:
            if has_header == True:
                header = next(text_file)
                
            k = 0
            i = 0
            for line in text_file:
                if i < len(selected_indexes) and k == selected_indexes[i]:
                    selected_lines.append(line.strip())
                    i+= 1
                else:
                    others.append(line.strip())
                k += 1
               
        return selected_lines, others, header.strip()