import os
import scprep
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import phate
import scanpy

data_dir = '../data/'
samples = ['CG4_MMT_WT_cellranger/', 'CG5_MMT_OBOB_cellranger/',  'CG11-HFD_MMT_cellranger', 'CG10-WT_MMT_cellranger',]
sample_names = ['CG4_wt', 'CG5_ob/ob', 'CG11-hfd', 'CG10_wt']
data_suffix = 'filtered_feature_bc_matrix/'

data_sets = []
for sample in tqdm(samples):
    data_path = os.path.join(data_dir, sample, data_suffix)
    ds = scprep.io.load_10X(data_path, gene_labels='both', sparse=False)
    # This sample has twice as many reads
    if sample == 'CG10-WT_MMT_cellranger':
        ds = ds * 0.5
    data_sets.append(ds)

data, metadata = scprep.utils.combine_batches(data_sets, sample_names, append_to_cell_names=True)
metadata = pd.DataFrame(metadata)
data_filtered = scprep.filter.remove_rare_genes(data, min_cells=15)
data_filtered = scprep.filter.filter_library_size(data_filtered, cutoff=15000, keep_cells='below')
data_libnorm = scprep.normalize.library_size_normalize(data_filtered)
data_libnorm = scprep.transform.sqrt(data_libnorm)
libsize = scprep.measure.library_size(data_filtered)
data_libnorm = scprep.filter.filter_gene_set_expression(data_libnorm, exact_word='mt-Co1', cutoff=12.5, keep_cells='below')

data = data_libnorm
metadata = metadata.loc[data.index]

# Save in AnnData object
adata = scanpy.AnnData(data)
adata.obs['sample'] = metadata['sample_labels']
gnames = [s.split()[0] for s in adata.var.index]
gens = [s.split()[1][1:-1] for s in adata.var.index]
adata.var.index = gnames
adata.var["names"] = gnames
adata.var["ids"] = gens

# Drop highly expressed hormone genes, which overwhelmingly affect cell-cell distances
gene_subset_only = ["Ins1", 'Ins2', 'Sst', 'Ppy', 'Gcg']
phate_op = phate.PHATE(random_state=42, verbose=True)
adata.obsm['X_phate'] = phate_op.fit_transform(adata[:, ~adata.var['names'].isin(gene_subset_only)].to_df())
adata.write_h5ad('data/all_exocrine_endocrine_processed.h5ad'))