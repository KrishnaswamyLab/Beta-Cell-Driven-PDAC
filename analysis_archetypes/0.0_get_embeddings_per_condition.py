import scanpy, magic
from sklearn import decomposition
import numpy as np
import pandas as pd
import pickle

adata_beta = scanpy.read_h5ad('../data/beta_run_2.h5ad')
for sample in ['WT', 'HFD', 'OB/OB']:
    adata_beta_subset = adata_beta[adata_beta.obs['samples'] == sample]
    data_magic_op = magic.MAGIC(random_state=42, t=10, verbose=False)
    data_magic = data_magic_op.fit_transform(adata_beta_subset.to_df())
    
    data_pc_op = decomposition.PCA(n_components=20, random_state=42)
    data_magic_pc = data_pc_op.fit_transform(data_magic)
    data_magic_pc_norm = data_magic_pc / np.std(data_magic_pc[:, 0])
    data_magic_pc_norm = pd.DataFrame(data_magic_pc_norm,
                                      index=adata_beta_subset.obs_names,
                                      columns=[f'PC{i+1}' for i in range(data_magic_pc_norm.shape[1])])

    if '/' in sample:
        sample = sample.replace('/', '_')
        data_magic_pc_norm.to_pickle(f'results/{sample}_magic_pc_norm.pkl')
        with open(f'results/{sample}_magic_pc_op.pkl', 'wb') as f:
            pickle.dump(data_pc_op, f)