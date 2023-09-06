#!/usr/bin/env python
# coding: utf-8

# ライブラリのインポート

# In[ ]:


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms


# データオーギュメンテーションと正規化を行う:

# In[5]:


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# データローダーをセットアップ:

# In[6]:


data_dir = 'XRayExam'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'val']}


# モデル、損失関数、オプティマイザを定義:

# In[10]:


import torchvision.models as models

model = models.resnet50(pretrained=True)

import torch.nn as nn
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Assuming 2 classes: Good Health and illness

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# トレーニングループを定義

# In[ ]:


num_epochs = 10

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = corrects.double() / len(image_datasets[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()


# モデルを保存

# In[ ]:


torch.save(model.state_dict(), 'xray_resnet50_model.pth')


# 機械学習モデルの訓練・バリデーション中の正解率をトラッキングし、それをグラフで表示するためにmatplotlibを使用して以下の手順で進める：
# 
# エポックごとの正解率を保存するリストを初期化
# トレーニングループの中で、エポックごとの正解率を保存する。
# ループの後で、matplotlibを使用して正解率のグラフを描画する。

# In[ ]:


import matplotlib.pyplot as plt

num_epochs = 10

# 正解率を保存するリストを初期化
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = corrects.double() / len(image_datasets[phase])
        
        # 正解率をリストに保存
        if phase == 'train':
            train_acc_history.append(epoch_acc)
        else:
            val_acc_history.append(epoch_acc)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()

# 正解率のグラフを描画
plt.figure(figsize=(10, 6))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# 訓練とバリデーションの損失の変化を観察するために各エポックの損失をリストに保存し、グラフをプロットする。

# In[ ]:


import matplotlib.pyplot as plt

num_epochs = 500  # エポック数を500に設定

# 損失を保存するリストを初期化
train_loss_history = []
val_loss_history = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = corrects.double() / len(image_datasets[phase])
        
        # 損失をリストに保存
        if phase == 'train':
            train_loss_history.append(epoch_loss)
        else:
            val_loss_history.append(epoch_loss)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print()

# 損失のグラフを描画
plt.figure(figsize=(10, 6))
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss over 500 epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# 適合率を表すグラフを作成する。

# In[ ]:


import matplotlib.pyplot as plt

num_epochs = 500

# Precisionを保存するリストを初期化
train_precision_history = []
val_precision_history = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        TP = 0  # True Positives
        FP = 0  # False Positives
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # TPとFPを計算
            for label, prediction in zip(labels, preds):
                if label == 1 and prediction == 1:
                    TP += 1
                if label == 0 and prediction == 1:
                    FP += 1

            corrects += torch.sum(preds == labels.data)

        precision = TP / (TP + FP) if TP + FP != 0 else 0
        epoch_acc = corrects.double() / len(image_datasets[phase])

        # Precisionをリストに保存
        if phase == 'train':
            train_precision_history.append(precision)
        else:
            val_precision_history.append(precision)

        print(f'{phase} Precision: {precision:.4f} Acc: {epoch_acc:.4f}')

    print()

# Precisionのグラフを描画
plt.figure(figsize=(10, 6))
plt.plot(train_precision_history, label='Train Precision')
plt.plot(val_precision_history, label='Validation Precision')
plt.title('Training and Validation Precision over 500 epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()


# 再現率を表すグラフを作成する

# In[ ]:


import matplotlib.pyplot as plt

num_epochs = 500

# Recallを保存するリストを初期化
train_recall_history = []
val_recall_history = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        TP = 0  # True Positives
        FN = 0  # False Negatives
        corrects = 0

        for inputs, labels in dataloaders[phase]:
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # TPとFNを計算
            for label, prediction in zip(labels, preds):
                if label == 1 and prediction == 1:
                    TP += 1
                if label == 1 and prediction == 0:
                    FN += 1

            corrects += torch.sum(preds == labels.data)

        recall = TP / (TP + FN) if TP + FN != 0 else 0
        epoch_acc = corrects.double() / len(image_datasets[phase])

        # Recallをリストに保存
        if phase == 'train':
            train_recall_history.append(recall)
        else:
            val_recall_history.append(recall)

        print(f'{phase} Recall: {recall:.4f} Acc: {epoch_acc:.4f}')

    print()

# Recallのグラフを描画
plt.figure(figsize=(10, 6))
plt.plot(train_recall_history, label='Train Recall')
plt.plot(val_recall_history, label='Validation Recall')
plt.title('Training and Validation Recall over 500 epochs')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

