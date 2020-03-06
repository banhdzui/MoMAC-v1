'''
Created on 16 Feb 2018

@author: danhbuithi
'''

class DictionaryHelper:
    
    @staticmethod
    def revert_key_value(my_dict):
        return {v : k for k, v in my_dict.items()}