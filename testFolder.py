import os
import string

fileList = os.listdir("data/")
fileList.sort()
#zooms = [fileName for fileName in fileList if 'ZoomIn' in fileName]
nums = [[int(s) for s in file if s.isdigit()] for file in [fileName for fileName in fileList if 'ZoomIn' in fileName]]
print(max(nums)[0])
