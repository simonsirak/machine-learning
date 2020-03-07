# Learning As Inference
This lecture covers how we can use learning as inference. In the previous lecture, we assumed that we knew the prior, 
likelihood and evidence (and could thus directly apply MAP or ML for classification). However, these are not always 
readily available and must be calculated before trying to infer new information. Specifically, we want to be able to infer
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,D)">. The big change is that we now want to 
infer based on the observation *and available data*. In order to know how this posterior looks like, we have to learn 
it.

## Different Learning Models
A *discriminative* model tries to learn 
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,D)">
directly. An example of this is Logistic Regression. A *generative* model tries to learn 
<img src="https://render.githubusercontent.com/render/math?math=Pr(y,x|D)">
and use this to work out the conditional probability. An example of this is Naive Bayes (specifically, Naive Bayes tries
to understand the numerator in Bayes Rule, which is precisely this joint probability.

A *parametric* model asserts a structure that is defined by a parameter, and the goal is to optimize based on the 
parameter, i.e 
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,\theta)"> 
where 
<img src="https://render.githubusercontent.com/render/math?math=\theta">
is estimated as 
<img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}">
using the dataset D. Examples of parametric model learning is ML estimation and MAP estimation.

A *non-parametric* model estimates the probability of getting *any* parameter given the dataset D, 
<img src="https://render.githubusercontent.com/render/math?math=Pr(\theta|D)">,
and then makes inference <img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,D)">
from 
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,\theta,D)Pr(\theta|D)">
by marginalizing out <img src="https://render.githubusercontent.com/render/math?math=\theta">. Note that this still 
uses 1 (or more) parameters "indirectly", but it does not claim that there exists an optimal parameter. Examples 
of a non-parametric model is a full Bayesian approach (i.e do not estimate parameter, but learn parameter distribution).

One core assumption that we make on the dataset is that the observations are i.i.d (Independent, Identically distributed).
Then we can factorize the joint probability of each observation occurring easily, and from that we can easily take the 
logarithm of that to get a sum of logarithmized probabilities (each probability = 1 observation).

## Maximum Likelihood Estimate
We want to optimize the likelihood of getting the dataset given the choice of parameter value, 
<img src="https://render.githubusercontent.com/render/math?math=\theta_{ML}=arg\ max\ P(D|\theta)">.
This is then used to estimate <img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,D)\approx Pr(y|x,\theta_{ML})">

In MLE for classification, we assume independence of classes; i.e samples belonging to one class do not influence the estimate 
obtained for another class. For instance, just because one class is more common than another, it should impact the probability 
of predicting a class, but we assume that it doesn't for the sake of parameter estimation. However we may still take the 
prior into consideration during classification (if we use MAP instead of ML for classification); the prior is however disregarded during parameter estimation.

From the MLE framework, you can justify/derive common estimates. For Bernoulli, the MLE estimate of 
<img src="https://render.githubusercontent.com/render/math?math=\lambda">
is simply the frequency of 1:s from the dataset. For Categorical, it is the same; just the frequency of 
number of datapoints belonging to a class.

It is crucial to note that MLE can be used to specify a distribution for both the Y:s and the X:s. Simply flip around 
the posterior to give the other perspective, i.e
<img src="https://render.githubusercontent.com/render/math?math=Pr(x|y,\theta)">
instead of 
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,\theta)">.
This way, you can explicitly learn in both directions (the training data has both Y and X, so data is not a problem).

## Naive Bayes Classifier
A Naive Bayes classifier mitigates the curse of dimensionality by viewing all features as conditionally independent given 
a class value. So, instead of modeling one D-dimensional distribution that takes all features into account given the 
output, we can simply estimate D one-dimensional distributions.

So a Naive Bayes classifier classifies based on Bayes Rule, but breaks down the D-dimensional joint likelihood into 
D 1-dimensional likelihoods due to the Bayes Assumption. We can then optimize parameters for each individual probability
using MLE or MAP, and then classify using MAP (or if you really want to, classify based on MLE). It is most successful 
when you have a good chunk of data and the features are reasonably independent.

One big problem in Naive Bayes is when no datapoint with class y has attribute x; then the product of D 1-dimensional 
data will be 0, all because we didn't have enough data. We can solve this by adding pseudocounts to all counts during 
estimation, so that no count is 0 (a form of regularization).

## Logistic Regression
Essentially, Logistic Regression is used to perform binary classification by checking whether predicted probability
is > or < than 0.5. The model is discriminative, it models the posterior directly. The lambda parameter (for the
underlying Bernoulli distribution) is optimized by assuming it follows a sigmoid function with a weight vector multiplied 
by an input observation. Optimizing for the weight w has no closed form solution, so gradient descent has to be used.

## Maximum A Posteriori
Maximum A Posteriori (MAP) is a point estimation method which maximizes the posterior, instead of the likelihood. This requires 
knowledge about the prior probability. For an analytical solution, you often choose the conjugate prior of the likelihood. To 
maximize with respect to the parameters, you can do a very similar approach as for ML; logarithmize the function to be optimized
(this does not change the arg max), then find the maximum by differentiating and setting the derivative to 0.

When using a gaussian prior (and gaussian for the error) in linear regression, we get ridge regression (if we substitute the rather
involved constant factor with lambda). If we use a laplacian prior, we get Lasso Regression.

The good thing with both ML and MAP is that once we've estimated the parameters, everything can be assumed to be known, assuming 
there is a true parameter value that represents the underlying model.

### Limitations of MAP
MAP essentially shifts the problem of estimating the parameters towards defining the parameters of the priors (this is the same as 
having to choose a good lambda in ridge regression; we've simply shifted the problem elsewhere). Furthermore, in the end we have 
calculated a posterior given parameters, but we actually want to calculate a posterior given the dataset; the parameters can only 
do so well in representing the dataset, since learning the posterior from the dataset in a parametric model assumes a certain 
structure to the dataset.

Another issue with MAP is that we have no measure of uncertainty (i.e how uncertain are we when we make a new classification? No way 
to tell). Bayesian estimation gives a measure of uncertainty!

## Non-Parametric Approaches: Bayesian Estimation
In this approach, there is no assumption that there is one true parameter value. Instead, we estimate  
<img src="https://render.githubusercontent.com/render/math?math=Pr(\theta|D)">
and use that to marginalize over theta to obtain posterior
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,\theta)">
which will look like 
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,D)=\int_{\theta \in \Theta}Pr(y|x,\theta,D)Pr(\theta|x, D)d\theta">. It is pretty safe to assume the conditional of Y given W is independent of D, i.e 
the data does not give more information once we have been given the weight. So we can drop D from the posterior in the integral. It is 
also pretty safe to say that w is independent of the x to-be-classified, so we can drop that from the right-most factor in the integral. 
So we get
<img src="https://render.githubusercontent.com/render/math?math=Pr(y|x,D)=\int_{\theta \in \Theta}Pr(y|x,\theta)Pr(\theta|D)d\theta">.

In order to calculate this, we would still need to model the parameters based on some assumed values (e.g parameter is distributed Gaussian
with assumed mean and covariance matrix). This will give us a prior of the parameter, which is used to obtain the parameter posterior in the integral. Furthermore, we need to model the conditional 
of Y given X to have a distribution that is based on the parameter. This will give us the Y posterior in the integral. 

From these two probabilities, it is raw probability theory and calculus (pretty brutal though) until you finally the final posterior that you want.
