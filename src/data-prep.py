# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 00:40:32 2022

@author: AMIT CHAKRABORTY
"""

import glob
import os
import shutil

os.makedirs('classy/train_data', exist_ok =True)
os.makedirs('classy/val_data', exist_ok =True)
parent_dir1 = 'classy/train_data'
parent_dir2 = 'classy/val_data'
dir_miss = 'miss'
dir_no = 'nomiss'
path1 = os.path.join(parent_dir1, dir_miss)
path2 = os.path.join(parent_dir2, dir_miss)
path3 = os.path.join(parent_dir1, dir_no)
path4 = os.path.join(parent_dir2, dir_no)
#############################################
os.makedirs(path1, exist_ok = True)
os.makedirs(path2, exist_ok =True)
os.makedirs(path3, exist_ok = True)
os.makedirs(path4, exist_ok =True)

##############################################

def data_prep():
    folders = os.listdir('classy/miss')
    file_mark = len(folders)*0.75
    i = 0
    for f in folders:
        if (i < file_mark):
            source = os.path.join('classy/miss', f)
            dest = os.path.join(path1, f)
            shutil.copy(source, dest)
        else:
            source = os.path.join('classy/miss', f)
            dest = os.path.join(path2, f)
            shutil.copy(source, dest)
        i = i + 1
    ################################################
    files1 = os.listdir('classy/nomiss')
    file_mark = len(files1)*0.75
    i = 0
    for f in files1:
        if (i < file_mark):
            source = os.path.join('classy/nomiss', f)
            dest = os.path.join(path3, f)
            shutil.copy(source, dest)
        else:
            source = os.path.join('classy/nomiss', f)
            dest = os.path.join(path4, f)
            shutil.copy(source, dest)
        i = i + 1



if __name__ == '__main__':    
    data_prep()    