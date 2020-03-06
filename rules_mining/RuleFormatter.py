'''
Created on Mar 15, 2017

@author: BanhDzui
'''

class RuleFormatter(object):
    
    @staticmethod
    def mydefaultLeft(item):
        return True
    
    @staticmethod
    def mydefaultRight(item):
        return True
    
    @staticmethod
    def mydefault(rule):
        return True
    
    
    @staticmethod
    def massLeft(item):
        return item.isdigit()
    
    @staticmethod
    def massRight(item):
        return not item.isdigit()
    
    @staticmethod
    def mass(rule):
        return rule.lhs_string().isdigit() and (not rule.rhs_string().isdigit())
    
    @staticmethod
    def rna(rule):
        condition = (len(rule.right_items) == 1)
        condition &= ('rna_' in rule.rhs_string())
        condition &=  ('rna_' not in rule.lhs_string())
        return condition
    
    @staticmethod
    def tcrLeft(item):
        return item != 'CD4' and item != 'CD8'
    
    @staticmethod
    def tcrRight(item):
        return item == 'CD4' or item == 'CD8'    
    
    @staticmethod
    def tcr(rule):
        left_key = rule.lhs_string()
        right_key = rule.rhs_string()
        return ('CD4' not in left_key) and ('CD8' not in left_key) and (right_key == 'CD4' or right_key == 'CD8')
    
    @staticmethod
    def ank3Left(item):
        return item != 'CASE' and item != 'HEALTHY'
    
    @staticmethod
    def ank3Right(item):
        return item == 'CASE' or item == 'HEALTHY'
        
    @staticmethod
    def ank3(rule):
        left_key = rule.lhs_string()
        right_key = rule.rhs_string()
        return ('CASE' not in left_key) and ('HEALTHY' not in left_key) and (right_key == 'CASE' or right_key == 'HEALTHY')
    
    @staticmethod
    def spectLeft(item):
        return 'class@' not in item
    
    @staticmethod
    def spectRight(item):
        return 'class@' in item
        
    @staticmethod
    def spect(rule):
        flag = True
        for item in rule.right_items:
            if 'class@' not in item:
                flag = False
                break
        left_key = rule.lhs_string()
        return ('class@' not in left_key) and flag == True
    
    @staticmethod
    def kddLeft(item):
        return ('c_' in item) == False
    
    @staticmethod
    def kddRight(item):
        return 'c_' in item
        
    @staticmethod
    def kdd(rule):
        left_key = rule.lhs_string()
        right_key = rule.rhs_string()
        return ('c_' not in left_key) and (len(rule.right_items) == 1 and 'c_' in right_key)
    
    @staticmethod
    def tcrmLeft(item):
        return True
    
    @staticmethod
    def tcrmRight(item):
        return True
        
    @staticmethod
    def tcrm(rule):
        a_count1 = 0
        b_count1 = 0
        for item in rule.left_items:
            if 'b_' in item:
                b_count1 += 1
            if 'a_' in item:
                a_count1 += 1
        if a_count1 > 0 and b_count1 > 0: return False
        
        a_count2 = 0
        b_count2 = 0
        for item in rule.right_items:
            if 'b_' in item:
                b_count2 += 1
            if 'a_' in item:
                a_count2 += 1
        if a_count2 > 0 and b_count2 > 0: return False
        
        return (a_count1 > 0 and b_count2 > 0) or (b_count1 > 0 and a_count2 > 0)
    
    @staticmethod
    def ppiLeft(item):
        return True
    
    @staticmethod
    def ppiRight(item):
        return True
        
    @staticmethod
    def ppi(rule):
        a_count1 = 0
        b_count1 = 0
        for item in rule.left_items:
            if 'h@' in item:
                b_count1 += 1
            if 'v@' in item:
                a_count1 += 1
        if a_count1 > 0 and b_count1 > 0: return False
        
        a_count2 = 0
        b_count2 = 0
        for item in rule.right_items:
            if 'h@' in item:
                b_count2 += 1
            if 'v@' in item:
                a_count2 += 1
        if a_count2 > 0 and b_count2 > 0: return False
        
        return (a_count1 > 0 and b_count2 > 0) or (b_count1 > 0 and a_count2 > 0)
    
    
    @staticmethod
    def spliceLeft(item):
        return item != 'EI' and item != 'IE' and item != 'N@'
    
    @staticmethod
    def spliceRight(item):
        return item == 'D_0' or item == 'D_1' or item == 'N@'
        
    @staticmethod
    def splice(rule):
        left_key = rule.lhs_string()
        right_key = rule.rhs_string()
        return ('EI' not in left_key) and \
            ('IE' not in left_key) and \
            ('N@' not in left_key) and \
            (right_key == 'EI' or right_key == 'IE' or right_key == 'N@')
    
    @staticmethod
    def peptideLeft(item):
        return 'c@' not in item
    
    @staticmethod
    def peptideRight(item):
        return 'c@' in item
        
    @staticmethod
    def peptide(rule):
        left_key = rule.lhs_string()
        right_key = rule.rhs_string()
        return ('m@' in left_key or 'd@' in left_key) and \
            ('p@' in left_key) and ('c@' not in left_key) and \
            (right_key == 'c@0' or right_key == 'c@1') 
        