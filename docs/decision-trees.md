# Decision Trees
This lecture covered decision trees, specifically classification trees. The nodes are created based on information 
gain, and we assume data is discrete with no order (so we have to split into every possible outcome for a feature).

If the data was ordered, we could constrain the tree to be binary, and add another parameter `s` to be used as 
part of the binary question for a node, `is feature F > value`. This would add another parameter to optimize
over. From now on, we will only discuss the non-ordered discrete case.

## Intuition
The intuition behind constructing a DT is that we choose to ask a series of questions about the features that gives 
as much information as possible, so that we at a leaf node can make an educated guess of the class of an input. For
any given question, we usually choose to ask about a single feature. We choose the feature that maximizes the 
expected amount of information this question gives.

## Entropy, Information Gain
The Shannon information content is measured as 
<img src="https://render.githubusercontent.com/render/math?math=\log{\frac{1}{p_i}}">
, where "i" 
stands for the i:th event. So if an event is very likely to happen, the information content of an outcome for that event will be very close 
to zero. On the other hand, if an event is very unlikely, getting a outcome of that event will give a large information content.

Entropy is a measure of uncertainty in a problem with different events. It is defined as 
<img src="https://render.githubusercontent.com/render/math?math=\sum_i{p_i\times\log{\frac{1}{p_i}}}">
, in other words the weighted sum of all information content. It is very closely related to the expected information content for an outcome
that is drawn from any of the i events. In short, we want to lower the entropy as much as possible through our series of quesitons. The 
entropy of a problem is often estimated by the entropy of a labeled dataset S `Ent(S)` (i.e where we know what the input and output is). 
This estimates the probability of events by calculating frequencies for each event from the dataset. If there is a probability distribution for the different events, that could also be used.

Information gain is the expected difference in entropy as a result of asking about the value of a feature F, and is often estimated by 
<img src="https://render.githubusercontent.com/render/math?math=Ent(S) - \sum_{v\in Values(F)}{\frac{|S_v|}{|S|}\times Ent(S_v)}">
. We choose the feature F that maximizes this difference, which is a greedy way to gain information the most amount of information for 
any single step.

Gini impurity is an alternative to entropy,
<img src="https://render.githubusercontent.com/render/math?math=1-\sum_i{p_i^2}">
, which peaks stronger at equal probabilities.

## Pruning
The above construction intuition would stop at a desired depth, or once we have only one class in the remaining subsets at the leafs. 
The classification is done by majority vote on the leaf nodes. However this may lead to complex trees that are overfit to the training 
data, leading to high variance. This could happen if the samples are non-representative or noisy, but also due to the overly complex 
tree. It can be solved by referring to Occam's razor (simplest explanation tends to be correct). In our case, we achieve a simpler DT
by pruning it.

We look at *reduced-error pruning*. The idea is to split the training data into a training set (to build the tree) and a validation set
(used to guide the pruning process). The motivation is that it is unlikely that both training and validation set has the same random 
errors.

The pruning will be performed as follows until the validation error starts to increase:
1. For each node, calculate the validation set error for pruning that node (which will remove child nodes as well) (prune = replace with majority vote leaf).
2. Remove node that had lowest validation set error.
