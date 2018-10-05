import os
import string
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


fileList = os.listdir("data/")
fileList.sort(key=natural_keys)
#zooms = [fileName for fileName in fileList if 'ZoomIn' in fileName]
nums = [[int(s) for s in file if s.isdigit()] for file in [fileName for fileName in fileList if 'ZoomIn' in fileName]]
nums = [int(''.join(''.join( str(x) for x in numI ))) for numI in nums]
#print([int(''.join(''.join( str(x) for x in numI )) for numI in nums])
print(max(nums))
