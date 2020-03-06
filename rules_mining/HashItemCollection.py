from rules_mining.HashItem import HashItem
import json

class HashItemCollection:
    def __init__(self):
        self.train_data = []
    
    def __iter__(self):
        return iter(self.train_data)
    
    def get_item(self, index):
        return self.train_data[index]
    
    def get_items_from(self, index):
        return self.train_data[index : ]
    
    def size(self):
        return len(self.train_data)
    
    def is_contain(self, item):
        for current_item in self.train_data:
            if current_item.last_item == item : 
                return True
        return False
        
    def sort(self):
        self.train_data.sort(key=lambda x: x.last_item, reverse=False)
    
    def add_item(self, hash_item):
        self.train_data.append(hash_item)
        
    def find_item(self, item):
        left = 0
        right = len(self.train_data) - 1
        while (left <= right):
            pivot = int((left + right)/2)
            if self.train_data[pivot].last_item == item: 
                return pivot
            if self.train_data[pivot].last_item < item:
                left = pivot + 1
            else:
                right = pivot - 1 
        return -1
        
    def add_tid(self, item, tid):
        index = self.find_item(item)
        if index == -1:
            hash_item = HashItem(item)
            hash_item.add_tid(tid)
            
            index = len(self.train_data) - 1
            self.train_data.append(hash_item)
            
            while index >= 0:
                if self.train_data[index].last_item > item:
                    self.train_data[index + 1] = self.train_data[index]
                    index -= 1
                else:
                    break
            self.train_data[index + 1] = hash_item        
        else:
            self.train_data[index].add_tid(tid)
    
    def serialize(self):
        temp = []
        for item in self.train_data:
            temp.append(item.serialize())
        return json.dumps(temp)

    def deserialize(self, json_string):
        self.train_data = []
        
        temp = json.loads(json_string)
        for item_string in temp:
            item = HashItem(None)
            item.deserialize(item_string)
            self.train_data.append(item)
        