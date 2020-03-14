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


## Linear Discriminant Analysis (LDA)
PCA finds a way to describe the data along axes that maximize the variation in the *data*. Another form of analysis is LDA, which maximizes the separation of *known classes*. The key difference here is that LDA is based on labeled data, and tries to work based on the known categories, to separate them as much as possible. We can 
then classify by choosing the class which *maximizes the posteriori* (i.e the Bayes discriminant rule).

## Discriminant Functions
A common tool in classification is the concept of a discriminant function, which has a continuous range. The discriminant functions of different classes are then compared, and the class corresponding to the best (usually 
maximal) discriminant function output of a given input will be the given classification. For instance, in K-NN
we can construct 1 "prototype" vector per class which corresponds to the mean vector for that class ("centroid 
or class distribution"). Then, we can classify new points by measuring the distance from the new point to the 
prototype vectors, classifying based on the minimum euclidean distance. A discriminant function to this could be
<img src="https://render.githubusercontent.com/render/math?math=g^k(x)=||x-a||^2"> (which could then be simplified 
e.g by expanding and removing a term).

> If the minimum distance is above a threshold, then we can also choose to set a "Don't know"-category.

Another discriminant function is to calculate the cosine betewen input vector and the prototype (called "simple
similarity" since we only have 1 prototype per class).

## Subspaces
Imagine using a set of vectors as a prototype, instead of only a mean vector, for each class (i.e span a subspace
that represents possible inputs for that class). In a sense, we are compressing the possible inputs of that class 
into a subspace defined by prior analysis.

> One way to find the set of spanning vectors for a specific class is by performing PCA on the input data belonging to that class, finding the principal components, and choosing the first *p* principal components (i.e those axes that give the largest variation of the features within a class). By only considering a subset of the principal components, we effectively extract the feature combinations that give most variation in the data, i.e the feature combinations that explains most characteristics of an input belonging to that class. 
>
> It is important to remember to not have a too small *p*, but not too large *p* either (may lead to overlap of input vectors between different classes), since this will create 
> a too simple/too complex model, which directly relates to the Bias-Variance tradeoff. The principal components 
> are (or should be) chosen in order of largest eigenvalue (i.e largest importance for expressing variation).
> One way to choose how many to use is to use cross validation; investigate the cumulative contributions (M is the total number of eigenvalues, "i" denotes the i:th class) 
> <img src="https://render.githubusercontent.com/render/math?math=\alpha(p_i)=\frac{\sum_{j=1}^{p_i}\lambda_j}{\sum_{j=1}^{M}\lambda_j}">. For each class i, we choose <img src="https://render.githubusercontent.com/render/math?math=p_i"> so that <img src="https://render.githubusercontent.com/render/math?math=\alpha(p_i) \leq K \leq \alpha(p_i+1)">, where K is some value (the same for all classes) optimized via cross validation.

Once we have our subspaces for each class, we can calculate the *similarity* of a new input vector in relation to 
the subspace of each class. This is captured by the *similarity* value 
<img src="https://render.githubusercontent.com/render/math?math=S_k=\sum_{i=1}^p(x,u_i)^2">. The weird parenthesis just means the dot product. Since the subspace is spanned with an orthonormal basis, the dot product directly 
describes the scalar projection of the input vector onto a given subspace basis vector (no division by the norm 
of the basis vector is needed because it is already normalized). So we essentially square the lengths of the new 
input vector along each basis vector, then we sum that together and we get the similarity of input vector on the subspace.

> Specifically, we get the sum of squared components of each of the basis vectors for the subspace. The components describe the projection of the input vector onto each dimension of the subspace. So we have taken the squared distance of the projection of the input vector onto the subspace. Taking the square root of this input will thus 
give us the length of the projection onto the subspace, neat!

## Fisher's Discriminant Rule ("Fisher's method" in DD2421 slides)
This is a type of LDA.

We can get an idea of the overlap by taking the ratio of *within-class variance* (summing up the sampled variance within each class and dividing by total number of samples) and *between-class variance* (summing up sampled variance between each class mean and the overall mean, each term scaled by the number of samples in that class, then the sum is divided by the number of total samples. The ratio is *within/between*.

Fisher's rule is to maximize this ratio, which is done with the help of the Scatter matrix (an estimator for the 
covariance matrix). We can then imagine that we have some matrix A representing the basis vectors for our new subspace. The idea is to get the scatter matrix *as seen after in the basis represented by A* of each class by transforming the original scatter matrices of each class using A and its transpose. We then optimize the ratio described earlier, but using the transformed *within* and *between* scatter matrices. It turns out the eigenvectors of the final matrix (Sw inverse times Sb) are precisely the vectors that best expresses the separation of classes. 

## A Clarification of Usage of PCA and LDA
LDA and (class-wise) PCA both try to accomplish the same thing; find basis vectors that best represent the characteristics of the problem. PCA tries to do this by extracting the relevant features in a dataset (usually for a given class), whereas LDA tries to find vectors that separate the classes as much as possible according to some discriminant rule (usually variance/separation-related).

Another way to think about it is that PCA tries to maintain as much structural similarity as before the feature selection. LDA tries to cluster together data from the same class. This makes LDA good for classification (although PCA is still good for classification as well, [particularly if there is little data per class][1]

So in reality, PCA and LDA are just ways to perform dimensionality reduction. PCA takes the most important features into a subspace for each class, which can then be used to perform subspace methods or prototype-based classificaiton. LDA creates a *single* subspace using whatever dimension you want, which can be used to perform classification tasks on (so this single subspace becomes the domain of the classification problem, all input is transformed into this subspace first). 

[1]: https://sebastianraschka.com/Articles/2014_python_lda.html#principal-component-analysis-vs-linear-discriminant-analysis
