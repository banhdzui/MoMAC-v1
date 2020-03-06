'''
Created on Mar 15, 2017

@author: BanhDzui
'''

class ItemsetFormatter(object):
   
    @staticmethod
    def mydefault(itemset):
        return True
    
    @staticmethod
    def mass(itemset):
        for item in itemset:
            if item.isdigit() == False:
                return True
        return False
    
    @staticmethod
    def tcr(itemset):
        for item in itemset:
            if item == 'CD4' or item == 'CD8':
                return True
        return False
    
    @staticmethod
    def rna(itemset):
        for item in itemset:
            if 'rna_' in item:
                return True
        return False
        
    @staticmethod
    def ank3(itemset):
        for item in itemset:
            if item == 'CASE' or item == 'HEALTHY':
                return True
        return False
    
    @staticmethod
    def spect(itemset):
        for item in itemset:
            if 'class@' in item:
                return True
        return False
    
    @staticmethod
    def kdd(itemset):
        for item in itemset:
            if 'c_' in item:
                return True
        return False
    
    @staticmethod
    def tcrm(itemset):
        a_count = 0
        b_count = 0
        for item in itemset:
            if 'b_' in item:
                b_count += 1
            if 'a_' in item:
                a_count += 1
        return (a_count > 0 and b_count > 0)

    @staticmethod
    def ppi(itemset):
        a_count = 0
        b_count = 0
        for item in itemset:
            if 'h@' in item:
                b_count += 1
            if 'v@' in item:
                a_count += 1
        return (a_count > 0 and b_count > 0)

    @staticmethod
    def splice(itemset):
        for item in itemset:
            if item == 'EI' or item == 'IE' or item == 'N@':
                return True
        return False
    
    @staticmethod
    def peptide(itemset):
        has_class = False
        has_peptide = False
        for item in itemset:
            if 'c@' in item: has_class = True 
            if 'p@' in item: has_peptide = True
            
        return has_class and has_peptide 
