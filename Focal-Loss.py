#!/usr/bin/env python
# coding: utf-8

# このとき健常クラスに正常データの4倍の重みを割り当てる。
# アルファパラメータをリストとして設定し、各クラスに異なる重みをつける。

# In[ ]:


import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = self.ce(inputs, targets)
        pt = torch.exp(-logpt)
        loss = self.alpha * (1-pt)**self.gamma * logpt
        return loss.mean()
      
class FocalLoss(nn.Module):
    def __init__(self, alpha=[1.0, 4.0], gamma=2.0):  # alpha値をリストとして設定
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = self.ce(inputs, targets)
        pt = torch.exp(-logpt)
        
        # alpha値をターゲットのラベルに基づいて適用
        alpha = self.alpha[targets]
        
        loss = alpha * (1-pt)**self.gamma * logpt
        return loss.mean()

# 使い方の例
outputs = torch.randn(10, 2)  # 10サンプル, 2クラス (健常と疾患) の場合
targets = torch.randint(0, 2, (10,))

# Focal Lossインスタンスを作成
criterion = FocalLoss(alpha=[1.0, 4.0], gamma=2.0)  # 疾患クラスのalpha値を健常クラスの4倍に設定

# 損失を計算
loss = criterion(outputs, targets)
print(loss)

