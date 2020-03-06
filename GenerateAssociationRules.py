'''
Created on Feb 6, 2017

@author: BanhDzui
'''

import sys

from common.DataSet import DataSet
from common.CommandArgs import CommandArgs
from rules_mining.RuleMiner import RuleMiner

if __name__ == '__main__':
    config = CommandArgs({'input'   : ('', 'Path of data-set file'),
                          'format'  : ('mydefault', 'Format of input data'),
                          'minsup'  : (0.1, 'Minimum support'),
                          'minconf' : (0.3, 'Minimum confidence'),
                          'maxitems': (-1, 'Maximum number of items in the rules'),
                          'class'   : (-1, 'Class index')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    print('Loading data....')
    train_data_set = DataSet()
    class_index = int(config.get_value('class'))
    train_data_set.load(config.get_value('input'), class_index)
    
    print('Generating rules ....')
    min_sup_src = float(config.get_value('minsup'))
    min_conf = float(config.get_value('minconf'))
    itemset_max_size = int(config.get_value('maxitems'))
    
    miner = RuleMiner(config.get_value('format'), train_data_set)
    miner.generate_itemsets_and_rules(min_sup_src, min_conf, itemset_max_size)
    
    print('Finished!!!')
    