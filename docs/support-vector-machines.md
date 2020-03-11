# Support Vector Machines (SVM)
This lecture covered linear separation and SVMs. There is really nothing I can write that explains it better than the 
following lecture from MIT 6.034 (although this does not seem to take into account points outside the boundary, which it
should):

[![SVMs](http://img.youtube.com/vi/_PwhiWxHK8o/0.jpg)](http://www.youtube.com/watch?v=_PwhiWxHK8o)

## Elaboration on Alphas
The weight length we are minimizing is subject to the constraint that every datapoint is on or beyond the
correct margin (leading to an inequality). When we extend the method of Lagrange multipliers to handle inequalities,
we assert that the alphas (the multipliers) are greater than 0, in order to satisfy the Karush Kuhn Tucker (KKT)
conditions and have an optimal solution.

In the primal formulation of the SVM problem, there exist alphas for each input vector in the training set. These 
correspond to the Lagrange multipliers in a Lagrangian. Each alpha in a solution describes how much the minimized 
value (i.e length of weight vector) changes if the constant in the constraints (i.e "1") changes. If we increase it 
slightly, the weight vector increases roughly linearly with the alpha as coefficient (alpha is > 0). This means the 
margin will decrease. So the larger an alpha is, the more it will increase the weight if we increase the 
constant slightly. 

> Increasing the constant roughly corresponds to shifting the decision boundary along the weight vector 
> axis. So if we slightly shift the decision boundary in either direction, the vector with largest alpha
> will have the largest impact in the weight vector. Therefore, support vectors that "carry the boundary"
> the most typically have a larger alpha.

As such, a large alpha means that an input has a large impact on the weight.
