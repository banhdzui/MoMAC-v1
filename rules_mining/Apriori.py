from multiprocessing import Process
from multiprocessing.managers import BaseManager

from rules_mining.HashTable import HashTable
from rules_mining.HashItem import HashItem
from rules_mining.HashItemCollection import HashItemCollection
from rules_mining.Helper import string_2_itemset

class Apriori:
    
    '''
    Apriori generator.
    - data_set is the data used to generate frequent itemsets.
    - synonym_items_dict: a dictionary of items that have same meaning. Item-sets are redundant if containing both of them.
    - root_folder: the folder is used to save intermediate result.
    '''
    def __init__(self, root_folder='tmp/'):
        
        self.tmp_folder = root_folder
        self.freq_itemsets_tmp_file = self.tmp_folder + 'freqitemsets.tmp'
        
    '''
    Generate all candidates for frequent item-sets that have one item.
    '''    
    @staticmethod
    def generate_C1(data_set):
        itemset_key = ''
        C_1 = HashTable()
        C_1.insert_key(itemset_key)
    
        ntransactions = data_set.size()
        
        for tid in range(ntransactions):
            transaction = data_set.get_transaction(tid)
            for item in transaction:
                C_1.add_tid(itemset_key, item, tid)
        return C_1
    
    '''
    Generate all frequent item-sets having one item.
    '''
    @staticmethod
    def generate_L1(data_set, min_sup_src):
        C_1 = Apriori.generate_C1(data_set)
        return C_1.generate_freq_itemsets(min_sup_src)

    
    '''
    Generate the frequent item-sets having k items based on k-1 item frequent item-sets
    '''
    @staticmethod
    def generate_Lk(min_sup_src, L_k1, C_k, k):
        for key, hash_item_collection in L_k1.get_items():
            for index in range(hash_item_collection.size() - 1):
                new_key = ''
                index_th_item = hash_item_collection.get_item(index)
                
                if key == '':
                    new_key = index_th_item.last_item
                else:
                    new_key = key +',' + index_th_item.last_item
                new_hash_collection = HashItemCollection()
                
                #check if it is infrequent item-set
                for item in hash_item_collection.get_items_from(index + 1):
                    
                    '''
                    Create new itemset and check its support
                    '''
                    new_item = HashItem(item.last_item)
                    inter_items = set(index_th_item.tids).intersection(item.tids)      
                    if len(inter_items) >= min_sup_src:  
                        new_item.add_tids(list(inter_items))
                        new_hash_collection.add_item(new_item)
                        
                '''
                Add the new itemsets into next level if there's any.
                '''
                if new_hash_collection.size() > 0:        
                    C_k.insert(new_key,  new_hash_collection) 

    '''
    Generate frequent item-sets from data. This is a multiprocess function.
    These item-sets and their supports are written to output_file (if required).
    '''
    def generate_freq_itemsets(self, data_set, min_sup_src, nthreads, 
                               end_index, output_file,
                               write_support = False):
        
        '''
        Step 1: Generate frequent item-sets with 1 item and write to file
        '''
        ntransactions = data_set.size()
        with open(output_file, 'w') as text_file:
            text_file.write(str(ntransactions))
            text_file.write('\n')
        
        L1 = Apriori.generate_L1(min_sup_src)
        freq_itemsets_dict = L1.get_itemset_dictionary()
        freq_itemsets_dict.ntransactions = ntransactions
        freq_itemsets_dict.save_file(output_file, 'a', write_support)
        freq_itemsets_dict.clear()
        
        '''
        Step 2: Generate frequent item-sets with more than 1 item and append to the file
        '''
        k = 2    
        L_k1 = L1
        
        while not L_k1.isEmpty() and (end_index == -1 or k <= end_index):
                     
            '''
            Divide data into many parts and create processes to generate frequent item-sets
            '''
            L_k = HashTable()
            chunks = L_k1.split(nthreads)
            processes = []
            
            C_ks = []
            BaseManager.register("AprioriHash", HashTable)
            manager = BaseManager()
            manager.start()
            C_ks.append(manager.AprioriHash())
            
            index = 0
            for L_k_1_chunk in chunks:
                process_i = Process(target = Apriori.generate_Lk, 
                                    args=(min_sup_src, L_k_1_chunk,C_ks[index], k))
                processes.append(process_i)
                index += 1
            
            # wait for all thread completes
            for process_i in processes:
                process_i.start()
                process_i.join()
             
            '''
            Merge results which returns from processes
            '''
            for new_C_k in C_ks:
                L_k.append(new_C_k)
            L_k1.clear()
            L_k1 = L_k
    
            '''
            Append frequent item-sets with k items to file
            '''
            freq_itemsets_dict = L_k1.get_itemset_dictionary()
            freq_itemsets_dict.ntransactions = ntransactions
            freq_itemsets_dict.save_file(output_file, 'a', write_support)
            freq_itemsets_dict.clear()
            
            k += 1
            
        
    @staticmethod 
    def checkInclusiveItems(itemset, new_item, inclusive_items_dict):
        for item in itemset:
            if item + ',' + new_item in inclusive_items_dict: return True 
        return False 
    
    '''
    Generate the frequent item-sets having k items based on k-1 item frequent item-sets
    Note: the intermediate results are written to a temporary file.
    '''
    @staticmethod
    def generate_Lk_w(min_sup_src, L_k1, C_k_file, k, inclusive_items_dict):
        file_writer = open(C_k_file, 'w') 
        for key, hash_item_collection in L_k1.get_items():
            for index in range(hash_item_collection.size() - 1):
                
                index_th_item = hash_item_collection.get_item(index)
                new_key = ''
                if key == '':
                    new_key = index_th_item.last_item
                else:
                    new_key = key +',' + index_th_item.last_item
                new_hash_collection = HashItemCollection()
                
                #check if it is infrequent item-set
                previous_itemset = string_2_itemset(new_key)
                for item in hash_item_collection.get_items_from(index + 1):
                    
                    '''
                    Check if the itemset contains any inclusive pair of items.
                    '''
                    if Apriori.checkInclusiveItems(previous_itemset, item.last_item, inclusive_items_dict):
                        continue
                    
                    '''
                    Create new itemset and check its support
                    '''
                    new_item = HashItem(item.last_item)
                    inter_items = set(index_th_item.tids).intersection(item.tids)      
                    if len(inter_items) >= min_sup_src:  
                        new_item.add_tids(list(inter_items))
                        new_hash_collection.add_item(new_item)
                
                '''
                Write the new itemsets into file if there's any.
                '''
                if new_hash_collection.size() > 0:  
                    file_writer.write(new_key)
                    file_writer.write('\n')
                    file_writer.write(new_hash_collection.serialize())      
                    file_writer.write('\n')
        file_writer.close()

    '''
    Generate frequent item-sets from data. This is a multiprocess function.
    These item-sets and their supports are written to output_file (if required).
    Note: the intermediate results are written to a temporary file.
    '''
    def generate_freq_itemsets_w(self, data_set, min_sup_src, nthreads, end_index, inclusive_items_dict, output_file):
        
        '''
        Step 1: Generate frequent item-sets with 1 item and write to file
        '''
        ntransactions = data_set.size()
        with open(output_file, 'w') as text_file:
            text_file.write(str(ntransactions))
            text_file.write('\n')
        
        
        L1 = Apriori.generate_L1(data_set, min_sup_src)
        L1.get_itemset_dictionary_w(output_file, 'a')
        
        '''
        Step 2: Generate frequent item-sets with more than 1 item and append to the file
        '''
        k = 2    
        L_k1 = L1
        
        while not L_k1.isEmpty() and (end_index == -1 or k <= end_index):
            
            '''
            Divide data into many parts and create processes to generate frequent item-sets
            '''
            chunks = L_k1.split(nthreads)
            L_k1 = None
            processes = []
            
            index = 0
            for L_k_1_chunk in chunks:
                chunk_output_file = self.freq_itemsets_tmp_file +'.'+ str(index)
                process_i = Process(target = Apriori.generate_Lk_w, 
                                    args=(min_sup_src, L_k_1_chunk,chunk_output_file, k, inclusive_items_dict))
                processes.append(process_i)
                index += 1
            
            # wait for all thread completes
            for process_i in processes:
                process_i.start()
                process_i.join()
             
            '''
            Merge results which returns from processes
            '''
            L_k1 = HashTable()
            for index in range(len(chunks)):
                chunk_input_file = self.freq_itemsets_tmp_file +'.'+ str(index)
                L_k1.deserialize(chunk_input_file, False)
            
            '''
            Append frequent item-sets with k items to file
            '''
            L_k1.get_itemset_dictionary_w(output_file, 'a')
            k += 1
            
        
        
        