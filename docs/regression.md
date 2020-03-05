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

## Regularization


