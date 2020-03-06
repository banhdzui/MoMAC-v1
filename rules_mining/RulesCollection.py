'''
Created on Apr 28, 2017

@author: BanhDzui
'''
from rules_mining.AssociationRule import AssociationRule

class RulesCollection(object):

    def __init__(self):
        self.rules = []
        
        
    def size(self):
        return len(self.rules)
        
    def add(self, r):
        self.rules.append(r)
        
    def clear(self):
        self.rules.clear()
        
    def save(self, file_name, is_append):
        mode = 'w'
        if is_append == True:
            mode = 'a'
        with open(file_name, mode) as text_file:
            for rule in self.rules:
                text_file.write(rule.serialize())
                text_file.write('\n')
                
    def load_file(self, file_name):    
        with open(file_name, "r") as text_file:
            for line in text_file:
                rule = AssociationRule.string_2_rule(line)
                self.rules.append(rule)
        
    def remove_redundancy(self, observations_dict):
        new_rules = []
        for r in self.rules:
            if r.is_redundant_rule(observations_dict):
                continue
            new_rules.append(r)
        self.rules = new_rules 
                
class RulesDictionary():
    
    def __init__(self):
        self.rules = {}
                    
    def load_file(self, file_name):
        with open(file_name, "r") as text_file:
            for line in text_file:
                rule = AssociationRule.string_2_rule(line)
                self.rules[line.strip()] = rule
    
    def get_rules(self):
        return list(self.rules.values())
    
    def get_rule_string(self):
        return list(self.rules.keys())
    
    def clear(self):
        self.rules.clear()
    