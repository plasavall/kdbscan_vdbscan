# Density-based clustering in scenarios with variable density
This repository contains two algorithms devoted to solve different problems regarding variable-density scenarios.

## K-DBSCAN

Kernel-Density-Based Spatial Clustering of Applications with Noise (K-DBSCAN) aims at identifying arbitrarily-shaped groups of points within a significantly sparse sample-space, without previous knowledge of the amount of resulting clusters.

## V-DBSCAN

Variable-Density-Based Spatial Clustering of Applications with Noise (V-DBSCAN) is a multi-scale variation of DBSCAN that takes into account the variations in density when moving away from the centroid of the data.

## Installing

There is no need to install the algorithms specifically. However, it is worth mentioning that the following libraries are used:

```python
numpy
scipy.stats
skimage.feature
copy.deepcopy
scipy.spatial.distance
collections
sklearn.neighbors
sklearn.cluster
matplotlib.pyplot
termplot
``` 

## Implementation details

Both algorithms are structured as classes, and a fit() function is defined to perform the clustering. The input to this function is the feature matrix.

The algorithms are implemented to use euclidean distance by default. If another distance metric is to be used, this is to be set when first generating the instance of the algorithm (along with algorithm-specific parameters). Scale-space parameters (if default configuration does not apply to the particular problem) are to be set through the fit function.

For instance:
```python
from kdbscan import KDBSCAN
from vdbscan import VDBSCAN
from sklearn.datasets import load_iris
from scipy.spatial.distance import cosine

X = load_iris().data
alg1 = KDBSCAN(h=0.35,t=0.4,metric=cosine)
alg2 = VDBSCAN(kappa=0.005,metric=cosine)
kde = alg1.fit(X[:,1:3],return_kde = True)
alg2.fit(X,eta=0.5)
alg1_labels = alg1.labels_
alg2_labels = alg2.labels_
kdbscan.plot_kdbscan_results(kde)
```    
The code above will load a toy example (the Iris dataset) and use both algorithms to group them into a number of clusters previously unknown. Additionally, it will plot the the results of KDBSCAN, i.e., the Kernel Density Estimation, the points (coloured differently depending on their output cluster assignment) and the valid peaks (purple triangles) as well as the discarded peaks (black triangles). 

Important Note: The current implementation of KDBSCAN is restricted to a 2-Dimensional sample space.

## Python Version

Both algorithms were developed using Python 2.

## Help and Support

For any doubts regarding the algorithms presented here, you can contact the corresponding author of the paper for which it was developed (see Citing section in this document).

## Citing

If you have used this codebase in a scientific publication and wish to cite it, please use the [Expert Systems With Applications article](https://www.sciencedirect.com/science/article/pii/S0957417419300521?dgcid=author).
    
```bibtex
@article{pla2019finding,
  title={Finding landmarks within settled areas using hierarchical density-based clustering and meta-data from publicly available images},
  author={Pla-Sacrist{\'a}n, Eduardo and Gonz{\'a}lez-D{\'\i}az, Iv{\'a}n and Mart{\'\i}nez-Cort{\'e}s, Tom{\'a}s and D{\'\i}az-de-Mar{\'\i}a, Fernando},
  journal={Expert Systems with Applications},
  volume={123},
  pages={315-327},
  year={2019},
  publisher={Elsevier}
}
```
