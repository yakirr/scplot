import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statutils import vis

# TODO: create an object to store the count information so we don't have to pass it everywhere
# TODO: use numpy.digitize to assign cells to bins
def get_bins(x, numbins):
    return np.concatenate([
        np.linspace(x.min()-1e-5, np.percentile(x, 99), numbins),
        np.array([x.max()])])

# turn a vector into a series of indicators for whether each value lies in each bin
def count(x, bins):
    X = (x[:,None] <= bins[None,:]).astype(np.int)
    X = X[:,1:] - X[:,:-1]
    return X.astype(np.float64)

# compute marginal histograms for two columns
def count_xy(x, y, numbins):
    return count(x, get_bins(x, numbins)), \
        count(y, get_bins(y, numbins))

# compute the local average of a covariate across the bins defined by countsxy
def local_mean(z, countsxy=None, **kwargs):
    countsx, countsy = \
        count_xy(**kwargs) if countsxy is None else countsxy

    counts = countsx.T.dot(countsy).T
    total_f = (countsx * z[:,None]).T.dot(countsy).T
    return total_f / counts

# compute the local standard deviation of a covariate across the bins defined by countsxy
def local_std(z, countsxy=None, **kwargs):
    countsx, countsy = \
        count_xy(**kwargs) if countsxy is None else countsxy

    counts = countsx.T.dot(countsy).T
    total_f = (countsx * z[:,None]).T.dot(countsy).T
    total_f2 = (countsx * (z**2)[:,None]).T.dot(countsy).T
    result = total_f2 / counts - (total_f / counts)**2
    return result

# compute the local density of observations (i.e., cells) across the bins defined by countsxy
def local_density(countsxy, **kwargs):
    countsx, countsy = \
        count_xy(**kwargs) if countsxy is None else countsxy

    counts = countsx.T.dot(countsy).T
    return counts / counts.sum()

# compute the number of cells in each bin in a 2D grid built using 50 bins for each axis
def local_density_plt(cells, **kwargs):
    matshow_kw = {'cmap':'RdGy', 'height':6, 'colorbar':True}
    matshow_kw.update(kwargs)
    countsxy = count_xy(cells[:,0], cells[:,1], 50)
    p = local_density(countsxy)
    vis.matshow(p, origin='lower', **matshow_kw)
    return p

def local_mean_plt(cells, z, ax=None):
    countsxy = count_xy(cells[:,0], cells[:,1], 50)
    mu = local_mean(z, countsxy)
    vis.matshow(mu, origin='lower', colorbar=True, height=6, cmap='seismic', ax=ax)
    return mu
