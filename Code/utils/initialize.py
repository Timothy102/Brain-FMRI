import os
import shutil
import re

def initfolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else: 
        shutil.rmtree(path)
        os.makedirs(path)
