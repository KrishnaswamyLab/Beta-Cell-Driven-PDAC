import scanpy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import decomposition

samples = ['GSE101207', 'GSE86469', 'GSE81608', 'GSE83139', 'GSE154126']
adata_beta = scanpy.read_h5ad('../data/beta_run_2.h5ad')
data = adata_beta.to_df()
data.columns = adata_beta.var['human_name']

pc_op = decomposition.PCA(n_components=100, random_state=42)
data_pc= pc_op.fit_transform(adata_beta.to_df())

## SAVE MAPPED GENE AND PC SPACE TO EACH ANNDATA OBJECT
for sample in tqdm(samples):
    adata_ref = scanpy.read_h5ad(f'preprocess_from_geo/processed/{sample}_adata.h5ad')
    out = np.load(f'results/output_human_all_to_all_cycle_1_correspondence_15_corr_correspondence_{sample}_normal_t2d.npz')
    ctrl_magan = pd.DataFrame(out['sample_to_all'], index=adata_ref.obs_names, columns=adata_beta.var['human_name'])

    intersection_genes = np.array(list(set(adata_beta.var['human_name']).intersection(adata_ref.var_names)))
    print('Number of intersection genes', len(intersection_genes))

    X = ctrl_magan[intersection_genes] - ctrl_magan[intersection_genes].mean(axis=0) + data[intersection_genes].mean(axis=0)
    for gene in tqdm(intersection_genes):
        ctrl_magan[gene] = X[gene]

    ctrl_magan.columns = adata_beta.var_names
    ref_data_pc = pc_op.transform(ctrl_magan)

    adata_ref.obsm['X_scmmgan_pca'] = ref_data_pc
    adata_ref.obsm['X_scmmgan_gene'] = ctrl_magan.values
    adata_ref.write(f'preprocess_from_geo/processed/{sample}_adata.h5ad', compression='gzip')

## COMBINE ANNDATA OBJECTS
all_adatas = []
n_obs = []
for sample in tqdm(samples):
    adata_ref = scanpy.read_h5ad(f'preprocess_from_geo/processed/{sample}_adata.h5ad')
    adata_ref.obs_names_make_unique
    adata_ref.var_names_make_unique()
    all_adatas.append(adata_ref)
    n_obs.append(adata_ref.n_obs)
all_adatas = scanpy.concat(all_adatas, join='outer')

## CLEAN UP RELEVANT VARIABLES
all_adatas.obs['sample'] = [x for sublist in [np.repeat(sample, n_obs[i]) for i,sample in enumerate(samples)] for x in sublist]
all_adatas.obs['age'] = [np.float(x.split(' ')[0]) for x in all_adatas.obs['age'].astype(str)]
all_adatas.obs['BMI'] = all_adatas.obs['BMI'].astype(float)
all_adatas.obs['HbA1c'] = all_adatas.obs['HbA1c'].astype(float)
all_adatas.obs['isT2D'] = (all_adatas.obs['disease'] == 'T2D').astype(float)
all_adatas.obs['isMale'] = (all_adatas.obs['sex'] == 'male').astype(float)
all_adatas.obs['full_ID'] = all_adatas.obs['sample'] + '_' + all_adatas.obs['donor']

## SAVE METADATA
metadata = all_adatas.obs[['sample','donor', 'sex', 'age', 'BMI', 'HbA1c', 'isT2D', 'isMale']].drop_duplicates()
metadata.index = metadata['sample'] + '_' + metadata['donor']
metadata['HbA1c'] = metadata['HbA1c'].replace(0.071, 7.1) ## misinputted
metadata['HbA1c'] = metadata['HbA1c'].replace(0.054, 5.4) ## misinputted
metadata.loc['GSE83139_ABAF490', 'isMale'] = np.nan ## no entry

# KEEP DONORS WITH >= 20 CELLS
metadata = metadata[all_adatas.obs['full_ID'].value_counts() >= 20]
all_adatas = all_adatas[all_adatas.obs['full_ID'].isin(metadata.index)]

all_adatas.write('results/all_mapped_human_datasets.h5ad', compression='gzip')
metadata.to_csv('results/metadata.csv')
