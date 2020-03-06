'''
Created on 12 Nov 2019

@author: danhbuithi
'''
import sys 
from common.CommandArgs import CommandArgs
from common.DataSet import DataSet


if __name__ == '__main__':
    config = CommandArgs({
                          'data'   : ('', 'Path of training data file'),
                          'n'       : (5, 'Number of sub-learning sets'),
                          'class'   :(-1, 'Class index')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    nsubsets = int(config.get_value('n'))
    class_index = int(config.get_value('class'))

    all_data = DataSet()
    all_data.load(config.get_value('data'), class_index)
    
    subsets = all_data.split_random_in_k(nsubsets)
    
    for i in range(nsubsets):
        test_data, train_data = DataSet.create_datasets_by_crossvalidation(subsets, i)
        
        test_data.save(config.get_value('data')+'.test'+'.'+str(i))
        train_data.save(config.get_value('data')+'.train'+'.'+str(i))