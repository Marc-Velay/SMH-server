# -*- coding: utf-8 -*-
import pickle
import json

if __name__ == "__main__":
    with open('save.pkl', 'rb') as f:
        message = pickle.load(f)
    data=json.loads(message.decode('utf-8'))
    title=data["label"]
    
    
    
