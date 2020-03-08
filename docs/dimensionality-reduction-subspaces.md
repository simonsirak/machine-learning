# Dimensionality Reduction and Subspaces
This lecture covers dimensionality reduction and subspaces.

## Principal Component Analysis (PCA)
PCA is an unsupervised learning technique that calculates *principal components* (dimensions defined by a linear combination 
of the different features), and uses them to represent data in a way that captures the vast majority of the variation in the data.

The principal components are thus found by finding the vector that maximizes the variance of the data along that vector.
It turns out that the principal component vectors are the eigenvectors of the covariance matrix! 

> Note: The covariance matrix can be estimated from the data matrices by subtracting the mean from each datapoint, 
> then multiplying the centered data matrix with its transpose (and then divided by n-1).

From the (normalized) eigenvectors (also called loadings), we can construct a matrix W which transforms the data matrix to eigen space.
The result of this transformation is the scores of each point, which is just a translation of the points into eigen space. The PC vectors 
give an idea of which feature is mostly responsible for each PC (since each element in the eigen vector says how much a particular feature 
weighs in the PC vector). We prefer it when one vector dominates.
