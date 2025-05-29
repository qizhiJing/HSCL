import torch.optim as optim
import numpy as np
import torch
import pandas as pd
from test import test
from train_val import train_val
from util.dataset import MultiModalDataset
from util.load_data import data_load
from util.seed import set_seed
from HSCL import HSCL
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# set_seed(42)

path = r'data\data.mat'
mri1, pet1, csf1, gnd1 = data_load(path=path, str='mync')
mri2, pet2, csf2, gnd2 = data_load(path=path, str='myad')
mri, pet, csf, gnd = [np.concatenate((data1, data2), axis=0) for data1, data2 in zip((mri1, pet1, csf1, gnd1), (mri2, pet2, csf2, gnd2))]

num_epochs = 250
excel = []

for times in range(5):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold1, (train_val_index, test_index) in enumerate(skf.split(mri, gnd)):
        model = []
        x_train_val_mri, x_test_mri, x_train_val_pet, x_test_pet, x_train_val_csf, x_test_csf, y_train_val, y_test = \
            mri[train_val_index], mri[test_index], pet[train_val_index], pet[test_index], csf[train_val_index], csf[test_index], gnd[train_val_index], gnd[test_index]
        test_dataset = MultiModalDataset(mri=x_test_mri, pet=x_test_pet, csf=x_test_csf, labels=y_test)
        test_loader = DataLoader(test_dataset, batch_size=len(y_test), shuffle=False)

        for fold, (train_index, val_index) in enumerate(skf.split(x_train_val_mri, y_train_val)):
            x_train_mri, x_val_mri, x_train_pet, x_val_pet, x_train_csf, x_val_csf, y_train, y_val = \
                x_train_val_mri[train_index], x_train_val_mri[val_index], x_train_val_pet[train_index], x_train_val_pet[val_index],\
                x_train_val_csf[train_index], x_train_val_csf[val_index],  y_train_val[train_index], y_train_val[val_index]

            train_dataset = MultiModalDataset(mri=x_train_mri, pet=x_train_pet, csf=x_train_csf, labels=y_train)
            train_loader = DataLoader(train_dataset, batch_size=len(y_train), shuffle=False)
            val_dataset = MultiModalDataset(mri=x_val_mri, pet=x_val_pet, csf=x_val_csf, labels=y_val)
            val_loader = DataLoader(val_dataset, batch_size=len(y_val), shuffle=False)

            hscl = HSCL().cpu()

            parameters = list(hscl.parameters())
            all_parameters = list(set(parameters))
            optimizer = optim.AdamW(all_parameters, lr=1e-3)
            train_val(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, model=hscl)
            hscl.load_state_dict(torch.load('checkpoint.pt'))

            model.append(hscl)

        result = test(test_loader=test_loader, model=model)
        excel.append(result)
        df = pd.DataFrame(excel)
        df.to_excel("result.xlsx", index=False)



