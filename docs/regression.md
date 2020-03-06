# Regression
This lecture will introduce the regression problem and different options to solve a regression problem.
In this lecture, it is assumed that the identity transformation is used on the input x:es, but you could
apply an arbitrary transformation to the input data before performing linear regression. The linearity is 
with respect to the coefficients, which linearly combines the different features to provide an output estimate.

## Linear Regression
Linear Regression (LR) is a parametric approach to regression. We assert that the function estimator is 
<img src="https://render.githubusercontent.com/render/math?math=\hat{f}(x)=w^Tx">
for some d-dimensional input x. The objective is to find a "w" such that the in-sample MSE (i.e training 
data) is minimized, i.e 
<img src="https://render.githubusercontent.com/render/math?math=E_{in}(\hat{f})=\frac{1}{N}\times\sum_{n=1}^N(\hat{f}(x_n)-y_n)^2">.
If we remove the initial fraction, the rest is called the Residual Sum of Squares (RSS). Replacing "f" with "w",
we get
<img src="https://render.githubusercontent.com/render/math?math=E_{in}(w)=\frac{1}{N}\times||Xw-Y||^2">,
where X is the input data (1 row = 1 input vector), and Y is the output vector. 

Luckily, the RSS is a convex function of "w", meaning we can take the gradient of the above expression and 
set it to 0 to obtain the minimizing w,
<img src="https://render.githubusercontent.com/render/math?math=w=(X^TX)^{-1}X^TY">.

## RANSAC
RANSAC stands for Random Sample Consensus, and it is a regression method used to deal with outliers. Outliers
destroy linear regression, but RANSAC is robust enough to handle outliers. Procedure is as follows:
1. Sample 2 points to estimate a line (or minimal number of points to estimate the hyperplane). We choose minimal
because this gives highest probability of not picking an outlier. 
2. Count the number of inliers that lie within a set threshold distance from the line. If number of inliers 
exceeds some parameter T, estimate line on these inliers and terminate.
3. If not, estimate a new 2-sample line.
4. After N trials, take largest set of inliers achieved from any 2-sample line in the N trials and estimate 
based on that set.

The probability of not picking two inliers after N trials is exponentially small; we only need relatively 
few trials.

The problem with vanilla RANSAC is that the chosen threshold distance can be too small (no inliers) or too 
large (all sampled lines are deemed equally good). This can be mitigated by extensions of RANSAC, e.g MLESAC
(don't know what this is though).

## k-NN Regression
This is a non-parametric regressor. Simply take the average of the responses of the k nearest neighbors and 
use it as output. Similar Bias-Variance tradeoff as KNN classifier.

## Parametric vs Non-Parametric
A parametric approach generally outperforms non-parametric if we have low amount of datasince parametric doesn't have 
to learn the fundamental structure as much as non-parametric does. Linear Regression is also easier to interpret
than KNN.

## Regularization in Ridge Regression
Regularization is the act of adding more information in order to reduce overfitting. Shrinkage is a 
common type of regularization, where you add a penalty to the (squared) coefficients of different features so 
that overly sensitive features get penalized. The total sum of penalties for the different features 
is then added on top of the RSS, and it is this total value that is minimized. The shrinkage 
parameter (usually lambda) is crucial in determining how aggressive the penalty should be; 
generally, you use Cross Validation (or Gradient Descent somehow) to find the best lambda.

This modified regression method is called Ridge Regression. When we penalize the squared 
coefficient of a feature, we are saying that we prefer that "if you don't greatly reduce the RSS, 
don't bother having a high coefficient". So essentially we are "shrinking" variables that aren't 
very important.

Another thing is that ridge regression allows estimation even if you do not have enough data points
to find one hyperplane (e.g only have 1 point in 2D, cannot span a line), since ridge regression 
asks about information aside from just the datapoints.

## Regularization in Lasso
Lasso (Least Absolute Shrinkage and Selection Operator) makes a small tweak to Ridge Regression;
instead of taking the squared coefficient, we take the absolute value of the coefficient (which becomes the l1 norm, i.e the norm we get from walking 1 feature at a time, like a taxicab taking 1 road at a time).

The big difference between Lasso and Ridge is that Ridge can never truly disregard a feature, only 
reduce the constant to a very small value for a feature; Lasso can set a feature coefficient to 0.
This is due to the fact that the function to minimize will have a derivative for which a large
enough lambda can cause the optimal coefficient to become 0.

In other words, Lasso provides both regularization and feature selection. As a general rule, Ridge 
is better than Lasso if most variables are relevant, and Lasso is better if most variables are 
irrelevant. And Lasso makes it easier to interpret the model as it becomes more sparse/less complex.
