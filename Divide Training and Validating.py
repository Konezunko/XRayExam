#!/usr/bin/env python
# coding: utf-8

# 健常データと疾患データをそれぞれ8:2に訓練データと検証データとして分ける。

# In[23]:


import os
import random
import shutil

base_dir = 'XRayExam'
categories = ['Good Health', 'illness']


train_split = 16000
val_split = 4000
total_files = train_split + val_split

for category in categories:
    files = [f for f in os.listdir(os.path.join(base_dir, category)) if f.endswith('.png')]
    # Ensure we have enough files
    if len(files) < total_files:
        raise ValueError(f"Not enough files in {category}. Expected {total_files} but got {len(files)}")
    random.shuffle(files)
    
    train_files = files[:train_split]
    val_files = files[train_split:train_split+val_split]
    
    train_dir = os.path.join(base_dir, 'train', category)
    val_dir = os.path.join(base_dir, 'val', category)

    # Create directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Move files to corresponding folders
    for file in train_files:
        shutil.move(os.path.join(base_dir, category, file), os.path.join(train_dir, file))
    
    for file in val_files:
        shutil.move(os.path.join(base_dir, category, file), os.path.join(val_dir, file))

