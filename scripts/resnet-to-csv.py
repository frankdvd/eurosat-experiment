
import torch

import torchvision.models as models

from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from dataset.eurosat import get_eurosat_dataloader

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print('==> Load pre-trained model..')
    assert os.path.isdir('pre-trained'), 'Error: no pre-trained model directory found!'
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 10)
    state_dict = torch.load('./pre-trained/ckpt_resnet_unbalanced.pth', map_location=torch.device('cpu'))['net']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.split('.')[0] == 'linear':
            name = 'fc.'+name.split('.')[1]
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)



    root_path = './data/2750'
    labels = ['Forest', 'River', 'Highway', 'AnnualCrop', 'SeaLake', 'HerbaceousVegetation', 'Industrial', 'Residential', 'PermanentCrop', 'Pasture']

    flat_data_arr = []
    target_arr = []

    trainloader, testloader = get_eurosat_dataloader('./data/')

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        logps = model.forward(inputs.to(device))
        for i in range(len(targets)):
            flat_data_arr.append(np.array(logps[:,:,0,0][i].cpu().detach()))
            target_arr.append(np.array(targets[i].cpu().detach()))

    for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):
        logps = model.forward(inputs.to(device))
        for i in range(len(targets)):
            flat_data_arr.append(np.array(logps[:,:,0,0][i].cpu().detach()))
            target_arr.append(np.array(targets[i].cpu().detach()))


    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data) #dataframe
    df['class']=target

    # X = df.iloc[:,:-1]
    # y = df.iloc[:,-1]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    # pd.concat([X_train, y_train], axis=1).to_csv("./resnet18features_train.csv")
    # pd.concat([X_test, y_test], axis=1).to_csv("./resnet18features_test.csv")

    df.to_csv("./data/resnet-output.csv", index=False)