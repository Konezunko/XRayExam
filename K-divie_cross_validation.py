#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import KFold
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim

# 仮のモデル、データローダー、変換などを定義します
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

# データ変換
transform = transforms.Compose([transforms.ToTensor()])

# データセットの定義
data_dir = 'path/to/your/dataset'  # あなたのデータセットへのパスを指定してください
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# K分割交差検証
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_ids, val_ids) in enumerate(kf.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')

    # データセットのサブセットを作成
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # データローダーの作成
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=train_subsampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, sampler=val_subsampler)

    # モデル、損失関数、オプティマイザの定義
    model = SimpleNN(input_size=3*224*224, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # トレーニングと評価のループ
    for epoch in range(3):  # ここでは3エポックで仮に設定しています
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs, labels = batch
            inputs = inputs.view(-1, 3*224*224)  # 入力サイズに適した形状に変更

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播 + 逆伝播 + 最適化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader)}')

    # 評価フェーズ（省略）
    # ...

    print('--------------------------------')

