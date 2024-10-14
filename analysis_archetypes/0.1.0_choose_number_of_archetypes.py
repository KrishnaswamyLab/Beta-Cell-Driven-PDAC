import sys
import numpy as np
import sklearn
import scprep
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import torch
from torch import nn, optim
from tqdm import tqdm

sys.path.append('~/bin/AAnet/') # replace with location of AAnet for replicating analysis
from AAnet_torch import *

condition = sys.argv[1]
f = open(f'{condition}_num_archetypes_loss.txt', 'a')
for run in range(3):
    np.random.seed(run) 
    for N_ARCHETYPES in range(9,11):
        print (N_ARCHETYPES)
        data_pc_op = pd.read_pickle(f'results/{condition}_magic_pc_op.pkl')
        data_magic_pc_norm = pd.read_pickle(f'results/{condition}_magic_pc_norm.pkl')

        print ('Density subsample...')
        distances, _ = sklearn.neighbors.NearestNeighbors(n_neighbors=2000).fit(data_magic_pc_norm).kneighbors()
        distances = distances.max(axis=1)
        p = distances / distances.sum()
        data_train = scprep.select.select_rows(data_magic_pc_norm, idx=np.random.choice(data_magic_pc_norm.shape[0],
                                                                        int(data_magic_pc_norm.shape[0] * 0.8), p=p, replace=False))
        data_test = data_magic_pc_norm.drop(data_train.index, inplace=False)

        data_train = data_train.values
        data_test = data_test.values
        
        extrema = torch.Tensor(utils.get_laplacian_extrema(data_train, n_extrema=N_ARCHETYPES, subsample=False))
        extrema = torch.Tensor(data_train[extrema.numpy().astype(int)])
        print ('Train...')
        device = torch.device('cpu')
        model = models.AAnet_vanilla(noise=0.05, layer_widths=[256, 128],
                                     n_archetypes=N_ARCHETYPES, 
                                     input_shape=data_train.shape[1],
                                     device=device, diffusion_extrema=extrema)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        data_loader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=True, num_workers=4)

        for i in tqdm(range(72)):
            if i == 0:
                r_loss = a_loss = 0
            else:
                loss, r_loss, a_loss=utils.train_epoch(model, data_loader, optimizer, epoch=i)
        
        print ('Reconstruct...')
        data_test_reconstructed = model.decode(model.encode(torch.Tensor(data_test))).detach().numpy()
        mse = mean_squared_error(data_test, data_test_reconstructed)
        f.write(f'{run} {condition} {mse}\n')
f.close()
