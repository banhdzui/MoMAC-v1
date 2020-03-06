from rules_mining.AssociationRule import AssociationRule
from rules_mining.Helper import itemset_2_string

from multiprocessing import Process
import json
from rules_mining.RulesCollection import RulesCollection

class Generator:
    
    def __init__(self, observations_dict, min_conf, itemset_formatter, rule_formatter, nThreads):
        self.itemset_formatter = itemset_formatter
        self.rule_formatter = rule_formatter
        
        self.nthreads = nThreads
        self.freq_itemset_dict = observations_dict
        
        self.min_conf = min_conf
    
    @staticmethod
    def string_2_rule_prob(s):
        sub_strings = s.split('#')
        rule  = Generator.string_2_rule(sub_strings[0].strip())
        v = json.loads(sub_strings[1].strip())
        return rule, v
    
    @staticmethod
    def rule_prob_2_string(rule, p):
        return rule.serialize() + '#' + json.dumps(p)
                
    '''
    Generate association rules for one item-set
    '''
    def enumerate_subsets(self, bit_mask, item_set, position, rule_collection, both_frequency): 
        '''
        Run out of items --> create rule and check format criterion
        '''
        if position >= len(item_set):
            lhs = []
            rhs = []
                    
            for index in range(len(bit_mask)):
                if bit_mask[index] == True:
                    lhs.append(item_set[index])
                else:
                    rhs.append(item_set[index])
                                      
            if (len(lhs) > 0 and len(rhs) > 0):
                rule = AssociationRule(lhs, rhs)
                
                if (self.rule_formatter == None or self.rule_formatter(rule) == True):
                    rule_collection.add(rule)
            
            return 
      
        value_domain = [True, False]
        '''
        Include position-th item into LHS 
        '''
        
        for value in value_domain:
            bit_mask[position] = value
               
            if (value == False):
                lhs_itemset = []
                for index in range(len(bit_mask)):
                    if bit_mask[index] == True:
                        lhs_itemset.append(item_set[index])
                        
                lhs_frequency = self.freq_itemset_dict.get_frequency(itemset_2_string(lhs_itemset))
                confidence = 0
                if lhs_frequency > 0: confidence = both_frequency/lhs_frequency
                
                if confidence < self.min_conf:
                    bit_mask[position] = True
                    continue
                
                self.enumerate_subsets(bit_mask, item_set, position+1, rule_collection, both_frequency)
            else:
                self.enumerate_subsets(bit_mask, item_set, position+1, rule_collection, both_frequency)
                
            bit_mask[position] = True
    '''
    Generate association rules for a set of item-sets and write results to a file
    '''
    def generate_rules(self, freq_itemsets_collection, output_file_name):
        total_rules = 0
        remaining_rules = 0
        k = 0
        rule_collection = RulesCollection()
        
        x = open(output_file_name, 'w')
        x.close()

            
        for itemset in freq_itemsets_collection:
            '''
            Check item-set first if it can generate a rule
            '''
            if len(itemset) == 1:
                continue
         
            if self.itemset_formatter is not None and self.itemset_formatter(itemset) == False:
                continue
            
            '''
            Write generated rule_collection into file
            '''
            k += 1
            if k % 200 == 0:
                #print ('writing some rule_collection to file: ' + str(k))
                total_rules += rule_collection.size()
                rule_collection.remove_redundancy(self.freq_itemset_dict)
                rule_collection.save(output_file_name, True)
                remaining_rules += rule_collection.size()
                rule_collection.clear()
            
            '''
            Generating association rule_collection.
            '''
            both_frequency = self.freq_itemset_dict.get_frequency(itemset_2_string(itemset))
            bit_mask = [True] * len(itemset)
            self.enumerate_subsets(bit_mask, itemset, 0, rule_collection, both_frequency)
                    
        total_rules += rule_collection.size()
        rule_collection.remove_redundancy(self.freq_itemset_dict)
        rule_collection.save(output_file_name, True)
        remaining_rules += rule_collection.size()
        rule_collection.clear()
        
        
    '''
    Generate association rules for a set of item-sets and write results to a file
    '''
    def generate_rules_spect(self, freq_itemsets_collection, output_file_name):
        total_rules = 0
        remaining_rules = 0
        k = 0
        rule_collection = RulesCollection()
        
        x = open(output_file_name, 'w')
        x.close()

            
        for itemset in freq_itemsets_collection:
            '''
            Check item-set first if it can generate a rule
            '''
            if len(itemset) == 1:
                continue
         
            if self.itemset_formatter is not None and self.itemset_formatter(itemset) == False:
                continue
            
            '''
            Write generated rule_collection into file
            '''
            k += 1
            if k % 200 == 0:
                total_rules += rule_collection.size()
                rule_collection.remove_redundancy(self.freq_itemset_dict)
                rule_collection.save(output_file_name, True)
                remaining_rules += rule_collection.size()
                rule_collection.clear()
            
            '''
            Generating association rule_collection.
            '''
            rhs = []
            lhs = []
            
            for item in itemset:
                if 'class@' in item:
                    rhs.append(item)
                else:
                    lhs.append(item)
                                      
            if (len(lhs) > 0 and len(rhs) > 0):
                rule = AssociationRule(lhs, rhs)
                confidence = self.freq_itemset_dict.get_confidence(rule)
                if confidence >= self.min_conf:
                    rule_collection.add(rule)
            
        total_rules += rule_collection.size()
        rule_collection.remove_redundancy(self.freq_itemset_dict)
        rule_collection.save(output_file_name, True)
        remaining_rules += rule_collection.size()
        rule_collection.clear()
        
    '''
    Generate association rules for whole data-set
    '''  
    def execute(self, output_file_name):
        
        itemset_chunks = self.freq_itemset_dict.split(self.nthreads)
        
        processes = []
        for index in range(self.nthreads):
            file_name = output_file_name + '.' + str(index)
            process_i = Process(target=self.generate_rules_spect, args=(itemset_chunks[index], file_name))
            processes.append(process_i)
            
            
        for process_i in processes:
            process_i.start()
            
        # wait for all thread completes
        for process_i in processes:
            process_i.join()
            
            
            