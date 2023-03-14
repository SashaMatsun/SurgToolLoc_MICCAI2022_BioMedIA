import os
#import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.ops import focal_loss
from torchvision.models import resnet50, ResNet50_Weights
import wandb
import matplotlib.pyplot as plt      
import torch.nn.functional as F
from torchvision import datasets, transforms   
from torch.optim.lr_scheduler import StepLR   
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

data_path = '/l/users/mugariya.farooq/miccai/frames'
train_path = 'miccai/train_data.csv'
test_path = 'miccai/test_data.csv'
batch_size = 128
epochs = 5
lr = 3e-5
gamma = 0.7
#batch_size = 4
device = torch.device('cuda:0')

class cholec(Dataset):
    def __init__(self, data_dir, label_path, transforms= None):
        self.data_dir = data_dir
        self.label_path = label_path
        self.transforms = transforms
        self.df = pd.read_csv(label_path, index_col=0)
    def __len__(self):
        return  self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row['video_name'] + '_' + str(int(int(row['Frame']) / 25) + 1).zfill(6) + '.png'
        img_path = self.data_dir + '/' + row['video_name'] + '/' + img_name
        #img_data = np.array(Image.open(self.img_dir + '/' + img_name).convert('RGB'), dtype='float32')
        img_data = Image.open(img_path)
        if self.transforms:
            img_data = self.transforms(img_data)
            
        label = row[1:7].values
        return img_data, torch.tensor(label.astype(float),dtype=torch.float)

transies = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transies_t = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_ds= cholec( data_dir= data_path, label_path=train_path, transforms= transies)
test_ds= cholec( data_dir= data_path, label_path=test_path, transforms= transies_t)

dl_t = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
# dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

checkpoint_path = 'checkpoints_cholec'
load_weights = False
weights_pth = 'path/to/checkpoint.pth' # !!!!! WEIGHTS MUST BE WRAPPED IN DATAPARALLEL, OTHERWISE CRASHES !!!!!
wandb_project_name = 'stl_' + 'cholec'


# WANDB
wandb.init(project=wandb_project_name)
wandb.config = {
        'learning_rate': 0.0001,
        'epochs': 5,
        'batch_size': 128
}

# MODEL
print('=' * 10, 'preparing the model', '=' * 10)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
n_in_f = model.fc.in_features
model.fc = nn.Linear(n_in_f, 6)
model = DataParallel(module=model)
model.to(device)

criterion = focal_loss.sigmoid_focal_loss
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))


# TRAIN + VALID CYCLE
for epoch in range(5):
    print('epoch', epoch, 'started')
    
    # TRAIN
    model.train()
    for i, (data_, target_) in enumerate(dl_t):
        data_, target_ = data_, target_.to(device)
        optimizer.zero_grad()
        outputs = model(data_)
        loss = criterion(outputs, target_, reduction='mean')
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('loss:', loss.data)
            wandb.log({'loss':loss})

    model.eval()
    preds = np.empty((0,6), float)
    for i, (data_, target_) in enumerate(dl_test):
        outputs = model(data_.to(device))
        outputs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = np.append(preds, outputs, axis=0)
        if preds.shape[0] % 100 == 0:
            print('eval iteration', preds.shape[0])
    target = test_ds.df[test_ds.df.columns[1:7].values]
    f1 = f1_score(target, preds > 0.5, average='macro')
    auc = roc_auc_score(target, preds, average='macro')
    print(f1, auc)
    wandb.log({'test_f1_score':f1})
    wandb.log({'test_auc':auc})
        
    print('epoch', epoch, 'done')
    torch.save(model.state_dict(), checkpoint_path + '/' +'_epoch' + str(epoch) + '_' + '.pth')


