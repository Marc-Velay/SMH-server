import os
import string
import json
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


fileList = os.listdir("data/RotG/")
fileList.sort(key=natural_keys)
print (fileList)

for file in fileList:
    with open("data/RotG/"+file, 'r') as f:
        jsonData = json.load(f)
        print(jsonData["label"])
    #jsonData["label"] = "RotG"
    #with open("data/RotG/"+file, 'w') as f:
    #    json.dump(jsonData, f)
