#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

# サンプルデータの準備
y_true = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0]  # この場合、クラス0が4つ、クラス1が6つのサンプルあり

# クラスの重みを計算（逆数的重み）
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_true)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# これを使ってクロスエントロピー損失関数を初期化
criterion = nn.CrossEntropyLoss(weight=class_weights)

# 以下に、モデルの定義とトレーニングループが続く

