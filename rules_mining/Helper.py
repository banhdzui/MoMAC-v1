
def string_2_itemset(key):
    if key == '':
        return []
    else: 
        return key.split(',')

def itemset_2_string(itemset):
    return ",".join(itemset)

def merge_itemsets(itemset_1, itemset_2):
    merged_items = []
    merged_items.extend(itemset_1)
    merged_items.extend(itemset_2)
    merged_items = list(set(merged_items))
    merged_items = sorted(merged_items)
    
    return merged_items

def get_full_path(prefix, file_name):
    if prefix == '': return file_name
    return prefix + '//' + file_name


def is_equal_itemset(itemset_1, itemset_2):
    if len(itemset_1) == 0 or len(itemset_2) == 0: return False
    if (len(itemset_1) != len(itemset_2)): return False
    for i in range(len(itemset_1)):
        if itemset_1[i] != itemset_2[i]: return False
    return True
