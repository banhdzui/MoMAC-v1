import json

class HashItem:
    
    def __init__(self, item):
        self.last_item = item 
        self.tids = []
    
    def add_tid(self, tid):
        self.tids.append(tid)
        
    def add_tids(self, tids):
        self.tids.extend(tids)
    
    def size(self):
        return len(self.tids)
    
    def serialize(self):
        return json.dumps((self.last_item, self.tids))
    
    def deserialize(self, json_string):
        result = json.loads(json_string)
        self.last_item = result[0]
        self.tids = result[1]
        