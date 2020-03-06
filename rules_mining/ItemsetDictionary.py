'''
Created on Apr 28, 2017

@author: BanhDzui
'''
from rules_mining.Helper import string_2_itemset

class ItemsetDictionary(object):
    

    def __init__(self, ntransactions = 0):
        self.itemsets = {}
        self.ntransactions = ntransactions
        self.length_of_max_itemset = 1
            
    def size(self):
        return len(self.itemsets)
    
    def exists(self, itemset_key):
        return itemset_key in self.itemsets
    
    def add_itemset(self, itemset_key, amount):
        self.itemsets[itemset_key] = amount
        
    def clear(self):
        self.itemsets.clear()
            
    def get_names(self):
        return self.itemsets.keys()
        
    def get_frequency(self, itemset_key):
        if self.exists(itemset_key):
            return self.itemsets[itemset_key]
        return 0
        
    def get_confidence(self, rule):
        left_frequency = self.get_frequency(rule.lhs_string())
        both_frequency = self.get_frequency(rule.itemset_string())
        if left_frequency == 0: return 0
        return both_frequency/left_frequency
    
    def get_frequency_tuple(self, rule):
        lhs_frequency = self.get_frequency(rule.lhs_string())
        rhs_frequency =self.get_frequency(rule.rhs_string())
        both_frequency = self.get_frequency(rule.itemset_string())
        
        return lhs_frequency, rhs_frequency, both_frequency
    
    def get_support(self, itemset_key):     
        return self.get_frequency(itemset_key)/self.ntransactions
       
    def split(self, nChunk):
        itemsets_names = self.itemsets.keys()
        nItemsets = len(itemsets_names)
        
        #print ('Number of frequent item-sets: ' + str(nItemsets))
        itemset_chunks = [[] for _ in range(nChunk)]
        size_of_chunk = (int)(nItemsets/nChunk) + 1
                    
        index = 0
        counter = 0
        
        for itemset_key in itemsets_names:
            if counter < size_of_chunk:
                itemset_chunks[index].append(string_2_itemset(itemset_key))
                counter += 1
            elif counter == size_of_chunk:
                index += 1
                itemset_chunks[index].append(string_2_itemset(itemset_key))
                counter = 1  
                  
        return itemset_chunks
    
    def save_file(self, file_name, write_mode = 'a', write_support = False):
        with open(file_name, write_mode) as text_file:
            for key, value in self.itemsets.items():
                t = value
                if write_support == True:
                    t = value/self.ntransactions
                text_file.write(key + ':' + str(t))
                text_file.write('\n')
            
    def load_file(self,file_name):
        self.itemsets.clear()
        
        with open(file_name, "r") as text_file:
            self.ntransactions = int(text_file.readline())
            for line in text_file:
                
                subStrings = line.split(':')
                itemset_key = subStrings[0].strip()
                frequency = int(subStrings[1].strip())
                
                self.itemsets[itemset_key] = frequency
                m = len(string_2_itemset(itemset_key))
                if m > self.length_of_max_itemset:
                    self.length_of_max_itemset = m
        