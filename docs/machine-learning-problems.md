# Machine Learning Problems
This lecture covered basic problems in ML. By "model", we mean a model in terms of the chosen method and hyperparameters,
not a specific trained instance of a model.

## Overfitting & Bias-Variance Tradeoff
Overfitting occurs when model is so complex that it is overly specialized to training data. Usually this 
leads to a high variance and low bias in the training data. Generally speaking, we want to find a sweet 
spot between a model that is too simple (high bias, low variance) and too complex (low bias, high variance). 

## Cross Validation
Cross Validation is a technique to estimate a test error without requiring additional data. We can write
the expected error (when seen over the universe of inputs and outputs) of a model as
<img src="https://render.githubusercontent.com/render/math?math=\epsilon_m=E(L(Y,f_m(X))">
, where "m" is the chosen model and X, Y are the inputs and outputs as taken from the true underlying (and
unknown) distribution. Cross validation aims to estimate 
<img src="https://render.githubusercontent.com/render/math?math=\hat{\epsilon_m}=\epsilon_m">
for each model under consideration. This is usually then used to choose the model with lowest estimated error.

K-folds cross validation is one of the simplest forms of cross validation. Partition the available data into 
K "folds". Then, estimate the expected error for a model by training it on every combination of K-1 folds, using the 
left-out fold to estimate the expected error. Then, take the average of these K estimates to receive your final 
estimation of the expected error for this model. Repeat the process for different models and choose the one with 
the smallest estimated expected error. The models could have simple differences like different hyperparameters, or 
they could have structural differences such as DT vs SVM.

If you have enough data, you could avoid Cross Validation altogether, and instead split the data into a training, validation 
and test set directly. Then we don't have to validate on the same data that we train on.

It is worthwhile to note that Cross Validation is simply a method for getting an idea of which model seems to be best.
Once the best model has been chosen by checking with Cross Validation, we train that model with *all* of the training 
data, not just with K-1 folds. This final trained model can then be evaluated on a separate test set.

## Curse of Dimensionality
The curse of dimensionality describes the problem that occurs when a problem requires high-dimensional data. In 
combinatorics, adding another dimension to the input (e.g another variable) of a combinatorial problem, the time needed
to find a solution increases exponentially. For machine learning, adding another dimension (i.e another feature) to 
the problem means we need exponentially more data to train sufficiently, usually due to the fact that every point seems 
"further away" in terms of euclidean distance. In reality, some features may not even be that relevant and just do more 
harm than good. Adding features may be helpful at first, but too many features may result in a decreased performance due
to the large amount of data needed to understand the proper interactions of each feature.

One can mitigate/avoid these effects by using techniques for dimensionailty reduction/feature selection.

## Bias-Variance Tradeoff
The bias of a model is the difference between the expected function estimation and the true function,
<img src="https://render.githubusercontent.com/render/math?math=E(\hat{f}_D(x)) - f(x)">
, where D is a dataset from the universe of datasets (the expected value iterates over all datasets).

The variance of a model is the expected error of any particular model with respect to the 
expected estimation of a model,
<img src="https://render.githubusercontent.com/render/math?math=\epsilon_m=E((\hat{f}_D(x) - E(\hat{f}_D(x)))^2)">.

A common tool used for estimating the error of a model is the MSE (Mean Squared Error). For a given x, 
the expected error taken over all possible models becomes
<img src="https://render.githubusercontent.com/render/math?math=E_D((\hat{f}_D(x) - f(x))^2)">.
This essentially is the expected error of an input x, with respect to all trained models. This can be 
decomposed in terms of bias and variance,
<img src="https://render.githubusercontent.com/render/math?math=E_D((\hat{f}_D(x) - f(x))^2)=Variance%2bBias^2">.

So now we have a way to describe the expected squared error for a sample x, seen over all possible trained models.
If we want the MSE for all possible input, we take the expected value of this with respect to all possible input x,
<img src="https://render.githubusercontent.com/render/math?math=E_x(Variance%2bBias^2)">.

Throughout this description, we have assumed there is no irreducible error. However there usually is, and this 
would reveal itself as a constant variance in the end resulting MSE.
