'''
Created on Apr 6, 2017

@author: BanhDzui
'''
from rules_mining.Apriori import Apriori
from rules_mining.Generator import Generator
from rules_mining.ItemsetDictionary import ItemsetDictionary

from rules_mining.ItemsetFormatter import ItemsetFormatter
from rules_mining.RuleFormatter import RuleFormatter
from rules_mining.AssociationRule import AssociationRule

from rules_mining.HashTable import HashTable
from rules_mining.Helper import merge_itemsets, itemset_2_string
from rules_mining.RulesCollection import RulesDictionary

from objective_measures.Interestingness import ObjectiveMeasure as om

import numpy as np
import json

class NegativeRuleMiner(object):    
    '''
    This class is used to generate and store a Naive Belief System 
    by using the most confident association rules
    '''

    def __init__(self, filter_name, data_set):
        
        self.temp_folder = 'tmp/'
        
        self.itemset_tmp_file = self.temp_folder + 'miner.tmp.itemsets'
        self.rules_tmp_file = self.temp_folder + 'miner.tmp.rules'
        self.best_rules_file = self.temp_folder + 'miner.best_tmp.rules'
        
        self.interestingness_tmp_file = self.temp_folder +'miner.tmp.interestingness'
        self.probabilities_tmp_file = self.temp_folder +'miner.tmp.probabilities'
        
        self.filter_name = filter_name
        self.data_set = data_set
        
        self.nthreads = 4
        
        
        self.feature_tmp_file = self.temp_folder +'miner.tmp.features'
        self.non_redundant_rule_tmp_file = self.temp_folder +'miner.tmp.non_redundant_rules'
        self.non_redundant_rule_feature_tmp_file = self.temp_folder + 'miner.tmp.non_redundant_rules.features'
        
        
    '''
    Find all pairs of items such that one of them includes the other
    This function returns a dictionary. If (A includes B or B includes A) then they should not appear in same item-sets. 
    '''
    def findInclusiveItemPairs(self, minsup, exclusive_item_filter):
        inclusive_items_dict = {}
        apriori_model = Apriori()
        
        L1 = apriori_model.generate_L1(self.data_set, minsup)
        freq_one_item_itemset_dict = L1.get_itemset_dictionary()
        freq_one_item_itemset_dict.ntransactions = self.data_set.size()
         
        L2 = HashTable()
        apriori_model.generate_Lk(minsup, L1, L2, k=2)
        freq_two_item_itemset_dict = L2.get_itemset_dictionary()
        freq_two_item_itemset_dict.ntransactions = self.data_set.size()
        
        nitems = len(freq_one_item_itemset_dict.itemsets)
        all_items = list(freq_one_item_itemset_dict.itemsets.keys())
        for i in range(nitems-1):
            first_item = all_items[i]
            nfirst = freq_one_item_itemset_dict.get_frequency(first_item[0])
            if exclusive_item_filter(first_item): continue
            
            for j in range(i+1, nitems):
                second_item = all_items[j]
                if exclusive_item_filter(second_item): continue
                
                merge_key = itemset_2_string(merge_itemsets(first_item, second_item))
                nboth = freq_two_item_itemset_dict.get_frequency(merge_key)
                if nboth == 0: continue
                
                nsecond = freq_one_item_itemset_dict(second_item[0])
                if nboth/nfirst >= 0.999 or nboth/nsecond >= 0.999:
                    inclusive_items_dict[merge_key] = True
        return inclusive_items_dict
                    
        
    '''
    Load generated frequent itemsets from file. 
    This method must be called after generate_freq_itemsets is called
    '''
    def load_freq_itemset_dictionary(self):
        observations_dict = ItemsetDictionary(0)
        observations_dict.load_file(self.itemset_tmp_file)
        return observations_dict
    
    '''
    Load generated association rules from file. 
    This method must be called after generate_association_rules is called
    '''
    def load_rule_dictionary(self):
        rules_dict = RulesDictionary()
        
        for i in range(self.nthreads):
            file_name = self.rules_tmp_file + '.' + str(i)
            rules_dict.load_file(file_name)
                
        return rules_dict

        
    '''
    Generate frequent itemsets from data-set
    '''
    def generate_freq_itemsets(self, min_sup_src, itemset_max_size):
        
        print ('generating frequent item-sets...')
        inclusive_items_filter = getattr(RuleFormatter, self.filter_name + 'Right')
        inclusive_items_dict = self.findInclusiveItemPairs(min_sup_src, inclusive_items_filter)
        
        apriori_model = Apriori(self.temp_folder)
        apriori_model.generate_freq_itemsets_w(self.data_set, 
                                               min_sup_src*self.data_set.size(), 
                                               self.nthreads, 
                                               itemset_max_size, 
                                               inclusive_items_dict, 
                                               self.itemset_tmp_file)
        
    '''
    Generate association rules from data-set. This method must be called after generate_freq_itemsets(...) is called
    '''
    def generate_association_rules(self, min_conf):
        freq_itemsets_dict = self.load_freq_itemset_dictionary()
        
        print ('generating rules ....')
        itemset_formatter = getattr(ItemsetFormatter, self.filter_name)
        rule_formatter = getattr(RuleFormatter, self.filter_name)
        rule_generator = Generator(freq_itemsets_dict, min_conf, itemset_formatter, rule_formatter, self.nthreads)
        rule_generator.execute(self.rules_tmp_file)
        
    '''
    Generate association rules and select K patterns with highest confidence.
    '''    
    def generate_itemsets_and_rules(self, min_sup_src, min_conf, itemset_max_size = -1):
        self.generate_freq_itemsets(min_sup_src, itemset_max_size)
        self.generate_association_rules(min_conf)
        self._extract_feature_vectors()
         
    '''
    Compute confidence for all association rules generated from data-set
    '''
    def compute_confidence(self, association_rules_list):
        observations_dict = self.load_freq_itemset_dictionary()
        
        rule_confidence_dict = {}
        for rule in association_rules_list:
            left, _, both = observations_dict.get_frequency_tuple(rule)
            rule_confidence_dict[rule.serialize()] = (both/left, both)
        return rule_confidence_dict

    '''
    Compute values of 31 interestingness measures for all association rules generated from data-set
    '''
    def compute_interestingnesses(self, output_file):
        print ('computing correlation among interestingness measures...')
        #measures = [om.confidence, om.lift]
        
        measures = [om.confidence, om.coverage, om.prevalence, om.recall, om.specificity, 
                    om.classificationError, om.lift, om.leverage, om.change_of_support, om.relative_risk, 
                    om.jaccard, om.certainty_factor, om.odd_ratio, om.yuleQ, om.yuleY, 
                    om.klosgen, om.conviction, om.weighting_dependency, 
                    om.collective_strength, om.jmeasure, 
                    om.one_way_support, om.two_ways_support, om.two_ways_support_variation, 
                    om.linear_coefficient, om.piatetsky_shapiro, om.loevinger,
                    om.information_gain, om.sebag_schoenauner, om.least_contradiction, 
                    om.odd_multiplier, om.counter_example_rate, om.zhang]
        
        print('loading frequent item-sets....')
        freq_itemsets_dict =  self.load_freq_itemset_dictionary()
        association_rules = self.load_association_rules()
        ntransactions = freq_itemsets_dict.ntransactions
        print ('computing interestingness for all rules ....')
        
        with open(output_file, 'w') as write_file:
            for rule in association_rules:
                interestingness = []
                lhs_frequency, rhs_frequency, both_frequency = freq_itemsets_dict.get_frequency_tuple(rule)
                
                for index in range(len(measures)):
                    value = measures[index](lhs_frequency/ntransactions, 
                                            rhs_frequency/ntransactions, 
                                            both_frequency/ntransactions)
                    interestingness.append(value)
                write_file.write(rule.serialize() + ';')            
                write_file.write(';'.join([str(x) for x in interestingness]))
                write_file.write('\n')   
    
    '''
    Load features of non-redundant rules which are saved in temp files.
    This method returns a sparse feature matrix.
    '''
    def load_feature_vectors(self):
        
        data = []
        with open(self.non_redundant_rule_tmp_file, 'r') as feature_reader:
            for line in feature_reader:
                _, f_vector = json.loads(line.strip())
                data.append(f_vector)
                
        return np.array(data)
        
    '''
    Load non-redundant rules which are saved in temp files.
    This method returns a list of rules.
    '''
    def load_association_rules(self):
        association_rules_list = []
        with open(self.non_redundant_rule_tmp_file, 'r') as rules_reader:
            for line in rules_reader:
                rule_text, _ = json.loads(line.strip())
                association_rules_list.append(AssociationRule.string_2_rule(rule_text.strip()))
        return association_rules_list
        
    '''
    Load features of non-redundant rules which are saved in temp files
    This method return features under dictionary form in which key is rule and value is its features.
    '''
    def load_rules_features_as_dictionary(self):
        features_dict = {}
        with open(self.non_redundant_rule_tmp_file, 'r') as feature_reader:
            for line in feature_reader:
                rule_text, f_vector = json.loads(line.strip())
                features_dict[rule_text] = f_vector
                
        return features_dict

      
    '''
    Extract features for rules which are save in temp files.
    Result is saved into another temp file.
    '''
    
    def _extract_feature_vectors(self):
        #print ('Determine feature vector ....')
        
        measures = [om.support, om.confidence, om.coverage, om.prevalence, om.recall, om.specificity, 
                    om.classificationError, om.lift, om.leverage, om.change_of_support, om.relative_risk, 
                    om.jaccard, om.certainty_factor, om.odd_ratio, om.yuleQ, om.yuleY, 
                    om.klosgen, om.conviction, om.weighting_dependency, 
                    om.collective_strength, om.jmeasure, 
                    om.one_way_support, om.two_ways_support, om.two_ways_support_variation, 
                    om.linear_coefficient, om.piatetsky_shapiro, om.loevinger,
                    om.information_gain, om.sebag_schoenauner, om.least_contradiction, 
                    om.odd_multiplier, om.counter_example_rate, om.zhang]
        
        #measures = [om.support, om.confidence, om.zhang]
        observations_dict = self.load_freq_itemset_dictionary()
        ntransactions = observations_dict.ntransactions
        
        features_writer = open(self.non_redundant_rule_tmp_file, 'w')
        for i in range(self.nthreads):
            input_file = self.rules_tmp_file + '.' + str(i)
            
            with open(input_file, 'r') as rules_reader:
                for line in rules_reader:
                    
                    rule = AssociationRule.string_2_rule(line.strip())
                    
                    f_vector = []
                    lhs_frequency, rhs_frequency, both_frequency = observations_dict.get_frequency_tuple(rule)
                
                    for index in range(len(measures)):
                        value = measures[index](lhs_frequency/ntransactions, 
                                            rhs_frequency/ntransactions, 
                                            both_frequency/ntransactions)
                        f_vector.append(value)
                    
                    f_vector.append(len(rule.left_items))
                    #f_vector = rule.compute_probs(observations_dict)
                    features_writer.write(json.dumps((rule.serialize(),f_vector)))
                    features_writer.write('\n')
                
        features_writer.close()
    