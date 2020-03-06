from rules_mining.HashItemCollection import HashItemCollection
from rules_mining.ItemsetDictionary import ItemsetDictionary

class HashTable:
    def __init__(self):
        self.table = {}
        
    def size(self):
        return len(self.table)
    
    def isEmpty(self):
        return len(self.table) == 0;
    
    def is_contain(self, key, last_item):
        return (key in self.table) and (self.table[key].is_contain(last_item))
    
    def get_items(self):
        return self.table.items()
    
    # insert a new key into the table
    def insert_key(self, key):
        self.table[key] = HashItemCollection()
    
    def insert(self, key, value):
        self.table[key] = value
            
    # remove a key from the table
    def remove_item(self, key):
        self.table.pop(key, None)
        
    # insert a new transaction id into a specific item-set
    def add_tid(self, key, item, tid):
        self.table[key].add_tid(item, tid)
        
    # insert a item set and its transaction 
    def add_item(self, key, hash_item):
        self.table[key].add_item(hash_item)
    
    # get all item-set in the hash table
    def get_itemset_dictionary(self):
        collection = ItemsetDictionary()
        for key, hash_item_collection in self.table.items():
            for hash_item in hash_item_collection:
                new_key = ''
                if key == '': 
                    new_key = hash_item.last_item
                else:
                    new_key = key + ',' + hash_item.last_item
                collection.add_itemset(new_key, hash_item.size())
        return collection
    
    def get_itemset_dictionary_w(self, output_file, write_mode):
        count = 0
        file_writer = open(output_file, write_mode)
        for key, hash_item_collection in self.table.items():
            for hash_item in hash_item_collection:
                new_key = ''
                if key == '': 
                    new_key = hash_item.last_item
                else:
                    new_key = key + ',' + hash_item.last_item
                file_writer.write(new_key + ':' + str(hash_item.size()))
                file_writer.write('\n')
                count += 1
                    
        file_writer.close()
        return count
    
    # get number of item-set have same K - 1 first items.
    def count_itemsets(self, key):
        return self.table[key].size()
    
    # get frequent item-set
    def generate_freq_itemsets(self, minsup):
        L = HashTable()
        for key, hash_item_collection in self.table.items():
            L.insert_key(key)
            for hash_item in hash_item_collection:
                if hash_item.size() >= minsup:
                    L.add_item(key, hash_item)
            if L.count_itemsets(key) == 0:
                L.remove_item(key)
        return L
                 
    def sort(self):
        for hash_item_collection in self.table.values():
            hash_item_collection.sort()

    # this function is used for multi-thread
    def append(self, other_hash_table):
   
        for key, hash_item_collection in other_hash_table.get_items():
            self.table[key] = hash_item_collection

    def clear(self):
        self.table.clear()
        
    def split(self, n):
        number_of_keys = self.size()
        if number_of_keys < n:
            return [self]
        
        number_for_each_part = (int)(number_of_keys/n) + 1
        counter = 0
        sub_hash_tables = []
        sub_hash_table = HashTable()
        
        for key, hash_item_collection in self.get_items():
            if counter < number_for_each_part:
                sub_hash_table.insert(key, hash_item_collection)
            elif counter == number_for_each_part:
                sub_hash_tables.append(sub_hash_table)
                sub_hash_table = HashTable()
                sub_hash_table.insert(key, hash_item_collection)
                counter = 0
            counter += 1
        sub_hash_tables.append(sub_hash_table)
        return sub_hash_tables     
    
    def serialize(self, file_name):
        with open(file_name, "w") as text_file:
            #json.dump(self.table, text_file)
            k = 0
            for key, value in self.table.items():
                if k > 0:
                    text_file.write('\n')
                text_file.write(key)
                text_file.write('\n')
                text_file.write(value.serialize())
                k += 1
            
    def deserialize(self, file_name, reset_table = True):
        if reset_table == True:
            self.table = {}
        with open(file_name, "r") as text_file:
            k = 0
            collection_key = None
            for line in text_file:
                if k % 2 == 0:
                    collection_key = line.strip()
                else:
                    collection = HashItemCollection()
                    collection.deserialize(line.strip())
                    self.table[collection_key] = collection
                k = k + 1