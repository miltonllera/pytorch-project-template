import numpy as np
import pandas as pd
from itertools import chain

import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import TSNE, MDS
from sklearn.linear_model import LogisticRegression
from itertools import product, combinations

from ignite.engine import Engine, Events, _prepare_batch
from .recording import create_activity_recorder


def temporal_pca(data, n_components=None, random_state=None, z_score=True):
    if z_score:
        data = data.groupby(['unit']).apply(
            lambda x: (x - x.mean())/x.std(ddof=1))

    X = data.groupby(level=['timestep', 'unit']).mean().values
    X = X.reshape(-1, len(data.index.unique(level='unit')))

    pca = PCA(
        n_components=n_components,
        copy=False, whiten=True,
        random_state=random_state
    ).fit(X=X)

    X_proj = pca.transform(X)

    components = range(1, pca.n_components_ + 1)    

    X_proj = pd.Series(
        name='activation projection',
        data=X_proj.reshape(-1),
        index=pd.MultiIndex.from_product(
            [data.index.unique(level='timestep'), components],
            names=['timestep', 'component']
        )
    )

    data = chain(pca.explained_variance_, pca.explained_variance_ratio_)
    expvar = pd.Series(
        name='value',
        data=list(data),
        index=pd.MultiIndex.from_product(
            [['expvar', 'expvar ratio'], components],
            names=['measurement', 'component']
        )
    )

    return X_proj, expvar, pca


def tsne_projection(data, n_components=2, perplexity=30.0):
    last_tstep = data.groupby(['input', 'unit']).last()
    units = sorted(data.index.unique(level='unit'))
    inputs = sorted(data.index.unique(level='input'))

    X = last_tstep.values.reshape(-1, len(units))
    X_proj = TSNE(n_components, perplexity).fit_transform(X)

    X_proj = pd.DataFrame(
        data=X_proj,
        index=pd.Index(inputs, names='inputs'),
        columns=['component #{}'.format(i + 1) for i in range(n_components)]
    )

    return X_proj


def get_encoding_act(model, data, device):
    recorder = create_activity_recorder(model, device)
    recorder.run(data)
    hidden_recs = np.concatenate(recorder.state.hidden, axis=0)
#     memcell_recs = np.concatenate(recorder.state.hidden, axis=0)
    return hidden_recs


def encoding_pca(data, n_components=0.95):
    pca = PCA(
        n_components=n_components,
        copy=False, whiten=True,
    ).fit(X=data)
    
    data_proj = pca.transform(data)
    
    components = range(1, pca.n_components_ + 1)    

    X_proj = pd.Series(
        name='activation projection',
        data=data_proj.reshape(-1),
        index=pd.MultiIndex.from_product(
            [list(range(data.shape[0])), components],
            names=['instances', 'component']
        ))
    
    return X_proj, pca


def similarity_matrix(recordings):
    corrcoef, p_values = [], []
    
    for X, Y in combinations(recordings, r=2):
        n_components = min(X.shape[1], Y.shape[1])
        cca = CCA(n_components=n_components, max_iter=1000).fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        X_c, Y_c = X_c.squeeze(), Y_c.squeeze()

        if len(X_c.shape) == 1:
            X_c = np.expand_dims(X_c, -1)
        if len(Y_c.shape) == 1:
            Y_c = np.expand_dims(Y_c, -1)

        cc = [pearsonr(x, y) for x, y in zip(X_c.T, Y_c.T)]
        corrcoef.append(np.mean(cc))
        
    m = len(recordings)
            
    corrmat = np.diag(np.ones(m))
    corrmat[np.triu_indices(m, k=1)] = corrcoef
    corrmat[np.tril_indices(m, k=-1)] = corrmat.T[np.tril_indices(m, k=-1)]
    
    return corrmat


def project_data(mat):
    mds = MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=12213,
                   dissimilarity="precomputed", n_jobs=1)
    embedding = mds.fit_transform(1.0-mat)
    return embedding
