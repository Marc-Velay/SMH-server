# -*- coding: utf-8 -*-
import pickle
import json
from pathlib import Path
from os.path import isfile

if __name__ == "__main__":
    with open('save.pkl', 'rb') as f:
        message = pickle.load(f)
    data=json.loads(message.decode('utf-8'))
    
    title=data["label"]
    n=0
    mf=Path('data/'+str(title)+'_'+str(n))
    while (isfile(mf)):
        n=n+1
        mf=Path('data/'+str(title)+'_'+str(n))

    with open('data/'+str(title)+'_'+str(n), 'wb') as f:
        pickle.dump(data, f)
    
    
    
