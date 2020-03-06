# Probabilistic Reasoning
This lecture introduces probabilistic reasoning in ML. Probabilistic reasoning has a clear interpretation, works in data poor
environments, and is easily extended given new observations. It may be computationally costly to train on large datasets, and 
it is usually hard to define closed solutions, but nevertheless it is a very general and useful tool in ML.

## Probability Primer
This will not cover all probability, only the rules that I didn't remember.

A *conjugate prior* (with respect to a likelihood function) is a prior distribution which, when used together with its 
corresponding likelihood function, ensures that the posterior probability is of the same probability distribution family
as the prior probability distribution. Using conjugate priors ensures the posterior is of a closed form, and thus removes 
the need for numerical integration. This is another reason why Gaussian distributions are popular; they are self-conjugate!

Covariance describes how two variables vary together, and is a natural extension from variance (variance = covariance of variable with itself)
If an increase in X corresponds to increase in Y, then a positive covariance exists. 
If increase X means decrease Y, negative covariance exists. If X leaves 
Y unaffected, covariance is 0. Covariacne measures the direction of a relationship.
For the normal distribution, zero covariance is equivalent to independence between the variables.

Correlation is covariance divided by the square root of each of the variances of X and Y. This makes correlation
a unitless version of covariance. Because of this, correlation is essentially a normalized covariance which 
can clearly describe the strength of a correlation by looking at where the correlation is between -1 and 1.

### Common Distributions
The Bernoulli distribution acts on a binary variable and has a lambda parameter which describes the probability of the 
variable being 1. Its conjugate prior is the Beta distribution.

The Categorical distribution is an extension of Bernoulli, which allows the variable to take on K different values. The 
probability of getting each value is specified by a separate lambda, with the lambdas summing to 1. Its conjugate prior is
the Dirichlet distribution.

The Gaussian distribution models a distribution around a mean, with a given variance. This also extends to multivariate 
Gaussian. Gaussians are very common for describing the spread of data around a measured point, or to describe/summarize 
data from multiple trials. It is self-conjugate.

### Central Limit Theorem
The linear combination of a large number of independent, identically distributed variables tend towards 
a normal distribution regardless of the underlying distribution of the variables (both for discrete and 
continuous variables).

## Probabilistic Reasoning: Introduction
There are two types of probabilistic problems in ML; Learning, where we want to estimate the joint
probability distribution of two variables X and Y from observations, and Inference, where we want 
to estimate the conditional probability of Y given that X has some value.

### Probabilistic Regression
Regression could either be defined through a conditional probability (by finding a joint probability distribution 
between X and Y, computing the posterior of Y and predicting using the conditional expectation of Y), or 
by an explicit regression model (define a deterministic regression model as usual but with an added error 
term, then redefine Y with the usually normal-distributed error in mind, and estimate parameters based on this 
new distribution of Y).

### Probabilistic Classification
This is typically solved with maximization procedures such as 
* Maximum A Posteriori (choose y that maximizes probability of getting y given observed data)
* Maximum Likelihood (choose y that maximizes probability of getting observed data given y)

Maximum A Posteriori is more robust, but the prior is not always known. For instance, when training
parameters for a classifier, we usually use ML since the prior for having a correct parameter is not obvious
or known. However, once the parameters are chosen and we 
want to classify, we can now use the data to compute/estimate the prior, and thus use the MAP 
method for classification.
