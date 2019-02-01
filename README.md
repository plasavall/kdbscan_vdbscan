# Density-based clustering in scenarios with variable density
This repository contains two algorithms devoted to solve different problems regarding variable-density scenarios.

## K-DBSCAN

Kernel-Density-Based Spatial Clustering of Applications with Noise (K-DBSCAN) aims at identifying arbitrarily-shaped groups of points within a significantly sparse sample-space, without previous knowledge of the amount of resulting clusters.

## V-DBSCAN

Variable-Density-Based Spatial Clustering of Applications with Noise (V-DBSCAN) is a multi-scale variation of DBSCAN that takes into account the variations in density when moving away from the centroid of the data.

## Implementation details

Both algorithms are implemented to use a euclidean distance by default. If another distance metric is to be used, this is to be set when first generating the instance of the algorithm (along with algorithm-specific parameters). For instance:


.. code:: python
    
    from kdbscan import KDBSCAN
    from vdbscan import VDBSCAN
    from scipy.spatial.distance import cosine
    alg1 = KDBSCAN(h=0.35,t=0.4,metric=cosine)
    alg2 = VDBSCAN(kappa=0.0012,metric=cosine)
    

----------
Installing
----------

There is no need to install the algorithms specifically. However, it is worth mentioning that the following libraries are used:

.. code:: python

    numpy
    scipy.stats
    skimage.feature
    copy.deepcopy
    scipy.spatial.distance
    collections
    sklearn.neighbors
    sklearn.cluster
    termplot

--------------
Python Version
--------------

Both algorithms were developed using Python 2.

----------------
Help and Support
----------------

For any doubts regarding the algorithms presented here, you can contact the corresponding author of the paper for which it was developed (see Citing section in this document).

------
Citing
------

If you have used this codebase in a scientific publication and wish to cite it, please use the `Expert Systems With Applications article <https://www.sciencedirect.com/science/article/pii/S0957417419300521?dgcid=author>`_.

    PLA-SACRISTÁN, Eduardo, et al. Finding landmarks within settled areas using hierarchical density-based clustering and meta-data from publicly available images. Expert Systems with Applications, 2019.
    
.. code:: bibtex

    @article{pla2019finding,
      title={Finding landmarks within settled areas using hierarchical density-based clustering and meta-data from publicly available images},
      author={Pla-Sacrist{\'a}n, Eduardo and Gonz{\'a}lez-D{\'\i}az, Iv{\'a}n and Mart{\'\i}nez-Cort{\'e}s, Tom{\'a}s and D{\'\i}az-de-Mar{\'\i}a, Fernando},
      journal={Expert Systems with Applications},
      volume={123},
      pages={315-327},
      year={2019},
      publisher={Elsevier}
    }
