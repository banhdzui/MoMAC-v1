import math
import sys


def divide(both, condition):
    if both == 0:
        return 0
    if condition == 0:
        return sys.float_info.max#float('inf')
    return both/condition

# These objective_measures for rule: left -> right

class ObjectiveMeasure:
    
    @staticmethod
    def support(pleft, pright, pboth):
        return pboth

    @staticmethod
    def confidence(pleft, pright, pboth, k = 1, m = 1):
        return divide(pboth, pleft)
    
    @staticmethod
    def coverage(pleft, pright, pboth, k = 1, m = 1):
        return pleft
    
    @staticmethod
    def prevalence(pleft, pright, pboth, k = 1, m = 1):
        return pright
    
    @staticmethod
    def recall(pleft, pright, pboth, k = 1, m = 1):
        return divide(pboth, pright)
    
    @staticmethod
    def _notLeftnotRight(pleft, pright, pboth):
        return max(0, 1 - (pleft + pright - pboth))
    
    @staticmethod
    def _leftNotRight(pleft,  pright, pboth):
        return pleft - pboth
    
    @staticmethod
    def _rightNotLeft(pleft, pright, pboth):
        return pright - pboth
    
    @staticmethod
    def specificity(pleft, pright, pboth, k = 1, m = 1):
        not_both = ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)
        not_left = 1 - pleft
        return divide(not_both, not_left)
    
    @staticmethod
    def classificationError(pleft, pright, pboth, k = 1, m = 1):
        return pboth + ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)
    
    @staticmethod
    def lift(pleft, pright, pboth, k = 1, m = 1):
        return ObjectiveMeasure.confidence(pleft, pright, pboth, k, m) * pright

    @staticmethod
    def leverage(pleft, pright, pboth, k = 1, m = 1):
        return ObjectiveMeasure.confidence(pleft, pright, pboth, k, m) - pleft * pright
    
    @staticmethod
    def change_of_support(pleft, pright, pboth, k = 1, m = 1):
        return ObjectiveMeasure.confidence(pleft, pright, pboth, k, m) - pright
    
    @staticmethod
    def relative_risk(pleft, pright, pboth, k = 1, m = 1):
        x = ObjectiveMeasure.confidence(pleft, pright, pboth, k, m)
        y = divide(ObjectiveMeasure._rightNotLeft(pleft, pright, pboth), 1 - pleft)
        
        if x == 0: return 0
        if y == 0: return sys.float_info.max
        return x/y 
    
    @staticmethod
    def jaccard(pleft, pright, pboth, k = 1, m = 1):
        return pboth/(pleft + pright - pboth)
    
    @staticmethod
    def certainty_factor(pleft, pright, pboth, k = 1, m = 1):
        x = (ObjectiveMeasure.confidence(pleft, pright, pboth, k, m) - pright)
        y = (1 - pright)
        if x == 0: return 0
        if y == 0: return sys.float_info.max
        return x/y
    
    @staticmethod
    def odd_ratio(pleft, pright, pboth, k = 1, m = 1):
        x = pboth * ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)
        y = ObjectiveMeasure._leftNotRight(pleft, pright, pboth) * ObjectiveMeasure._rightNotLeft(pleft, pright, pboth)
        
        if (x == 0): return 0
        if y == 0: return sys.float_info.max
        return x / y
    
    @staticmethod
    def yuleQ(pleft, pright, pboth, k = 1, m = 1):
        a = pboth * ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)
        b = ObjectiveMeasure._leftNotRight(pleft, pright, pboth) * ObjectiveMeasure._rightNotLeft(pleft, pright, pboth)
        
        x = (a - b)
        y = (a + b)
        
        if x == 0: return 0
        if y == 0: return sys.float_info.max
        return x/y
    
    @staticmethod
    def yuleY(pleft, pright, pboth, k = 1, m = 1):
        a = math.sqrt(pboth * ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth))
        b = math.sqrt(ObjectiveMeasure._leftNotRight(pleft, pright, pboth) * ObjectiveMeasure._rightNotLeft(pleft, pright, pboth))
        
        x = (a - b)
        y = (a + b)
        
        if x == 0: return 0
        if y == 0: return sys.float_info.max
        return x/y
    
    @staticmethod    
    def klosgen(pleft, pright, pboth, k = 1, m = 1):
        a = ObjectiveMeasure.confidence(pleft, pright, pboth, k, m) - pright
        b = ObjectiveMeasure.confidence(pright, pleft, pboth, k, m) - pleft
        return math.sqrt(pboth) * max(a, b)
        
    @staticmethod
    def conviction(pleft, pright, pboth, k = 1, m = 1):
        x = pleft * (1 - pright)
        y = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        
        if x == 0: return 0
        if y == 0: return sys.float_info.max
        return x / y
    
    @staticmethod    
    def weighting_dependency(pleft, pright, pboth, k = 1, m = 1):
        return (math.pow(pboth/(pleft * pright), k) - 1) * math.pow(pboth, m)
    
    @staticmethod
    def collective_strength(pleft, pright, pboth, k = 1, m = 1):
        a = pboth + ObjectiveMeasure.specificity(pleft, pright, pboth, k, m)
        b = pleft * pright + (1 - pleft) * (1 - pright)
        
        if a == 0 or (1 - b) == 0: return 0
        if (1 - a) == 0 or b == 0: return sys.float_info.max
        
        return (a * (1 - b))/ (b * (1 - a))
    
    @staticmethod
    def gini_index(pleft, pright, pboth, k = 1, m = 1):
        t = 0
        a = ObjectiveMeasure.confidence(pleft, pright, pboth, k, m)
        b = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)/pleft
        t += (pleft * (a ** 2 + b ** 2))
        
        a = ObjectiveMeasure._rightNotLeft(pleft, pright, pboth)/(1 - pleft)
        b = ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)/(1 - pleft)
        t += ((1- pleft) * (a ** 2 + b ** 2))
        
        t -= (pright ** 2)
        t -= ((1-pright) ** 2)
        
        return t
    
    @staticmethod
    def jmeasure(pleft, pright, pboth, k = 1, m = 1):
        a = pboth * math.log(ObjectiveMeasure.confidence(pleft, pright, pboth, k, m)/pright)
        b = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        if b != 0:
            a += b * math.log(b/(pleft * (1 - pright)))
        return a
      
    @staticmethod
    def one_way_support(pleft, pright, pboth, k = 1, m = 1):
        a = ObjectiveMeasure.confidence(pleft, pright, pboth, k, m)
        if a == 0: return 0
        return a * math.log2(a/pright)
     
    @staticmethod    
    def two_ways_support(pleft, pright, pboth, k = 1, m = 1): 
        if pboth == 0: return 0
        return pboth * math.log2(pboth / (pleft * pright))  
        
    @staticmethod
    def two_ways_support_variation(pleft, pright, pboth, k = 1, m = 1):
        a = ObjectiveMeasure.two_ways_support(pleft, pright, pboth, k, m)
        
        b = 0
        t = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        if t != 0: 
            b = t * math.log2(t / (pleft * (1- pright)))
            
        c = 0
        t = ObjectiveMeasure._rightNotLeft(pleft, pright, pboth)
        if t != 0: 
            c = t * math.log2(t / ((1 - pleft) * pright))
        
        d = 0
        t = ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)
        if t != 0:
            d = t * math.log2(t / ((1 - pleft) * (1-pright)))
        
        return a + b + c + d
    
    @staticmethod
    def linear_coefficient(pleft, pright, pboth, k = 1, m = 1):
        x = ObjectiveMeasure.piatetsky_shapiro(pleft, pright, pboth, k, m)
        y = pleft * pright * (1 - pleft) * (1 - pright)
         
        if pleft == 1 or pright == 1: return sys.float_info.max
        return x/math.sqrt(y)
    
    @staticmethod
    def piatetsky_shapiro(pleft, pright, pboth, k = 1, m = 1):
        return pboth - pleft * pright
    
    @staticmethod
    def cosine(pleft, pright, pboth, k = 1, m = 1):
        return pboth/math.sqrt(pleft * pright)
    
    @staticmethod
    def loevinger(pleft, pright, pboth,  k = 1, m = 1):
        y = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        if y == 0: return sys.float_info.max
        
        return 1 - (pleft * (1 - pright))/y
    
    @staticmethod
    def information_gain(pleft, pright, pboth, k = 1, m = 1):
        return math.log(ObjectiveMeasure.confidence(pleft, pright, pboth, k, m)/pright)
    
    @staticmethod
    def sebag_schoenauner(pleft, pright, pboth,  k = 1, m = 1):
        y = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        if y == 0: return sys.float_info.max
        return pboth/y
        
    @staticmethod
    def least_contradiction(pleft, pright, pboth, k = 1, m = 1):
        return (pboth - ObjectiveMeasure._leftNotRight(pleft, pright, pboth)/pright)
        
    @staticmethod
    def odd_multiplier(pleft, pright, pboth, k = 1, m = 1):
        y = ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        if y == 0: return sys.float_info.max
        return (pboth * (1 - pright))/(pright * y)
    
    @staticmethod
    def counter_example_rate(pleft, pright, pboth, k = 1, m = 1):
        return 1 - (ObjectiveMeasure._leftNotRight(pleft, pright, pboth)/pboth)
    
    @staticmethod
    def zhang(pleft, pright, pboth, k = 1, m = 1):
        a = pboth * (1 - pright)
        b = pright * ObjectiveMeasure._leftNotRight(pleft, pright, pboth)
        
        x = pboth - pleft * pright
        y = max(a, b)
        if x == 0: return 0
        if y == 0: return sys.float_info.max
        return x / y
    
    
        
    @staticmethod 
    def myInfogain(pleft, pright, pboth, k=1, m=1):
        D = 0
        if pright != 0:
            D += pright * math.log2(pright)
        if 1 - pright != 0:
            D += (1-pright) * math.log2(1 - pright)
        D = -D 
        
        I = 0
        if pboth != 0:
            I += pboth * math.log2(pboth/pleft)
            
        p = pleft - pboth
        if p != 0:
            I += p * math.log2(p/pleft)
        
        
        p_not_left = 1 - pleft
        p = pright - pboth 
        if p != 0:
            I += p * math.log2(p/p_not_left)
        
        p = ObjectiveMeasure._notLeftnotRight(pleft, pright, pboth)
        if p != 0:
            I += p * math.log2(p/p_not_left)
        
        #I = -I
        return I 
        