# Machine Learning Q&A

## Machine Learning Basics
1. [Bias Variance TradeOff](#bias-variance-tradeoff)
2. [Bayes classifier](#bayes-classifier)
3. [What is F test](#f-test)
4. [Why cannot use Least Square for Classification problem?](#why-cannot-use-least-square-for-classification-problem)
5. [How to decide logistic regression goodness of fit?](#how-to-decide-logistic-regression-goodness-of-fit)
6. [Logistic Regression for multi-class classification](#logistic-regression-for-multi-class-classification)
7. [Why ROC/AUC?](#why-rocauc)
8. [Why k-fold can have better test error estimation than LOO?](#why-k-fold-can-have-better-test-error-estimation-than-loo)
9. [Curse of Dim](#curse-of-dim)
10. [How to handle imbalanced data](#how-to-handle-imbalanced-data)
11. [Feature Selection](#feature-selection)
12. [Why not p-value to determine feature importance?](#why-not-p-value-to-determine-feature-importance)
13. [Why Ridge Works?](#why-ridge-words)
14. [More on Lasso](#more-on-lasso)
15. [How to prune a tree?](#how-to-prune-a-tree)
16. [How feature importance in trees is calculated?](#how-feature-importance-in-trees-is-calculated)
17. [Boosting Tree the tunning param](#boosting-tree-the-tuning-param)
18. [SVM](#svm)
19. [Kernel Tricks](#kernel-tricks)
20. [What is MLE and assumption](#what-is-mle-and-assumption)
21. [What is Overfitting and how to avoid it?](#what-is-overfitting-and-how-to-avoid-it)

<br/><br/>
## Deep Leaerning Basics
1. [What is NDCG? MAP vs NDCG?](#what-is-map-and-ndcg)
2. [L1 Norm vs L2 Norm](#l1-norm-vs-l2-norm)
3. [What is Vanishing Gradient?](#what-is-vanishing-gradient)
4. [What is batch normalization, and how does it address the vanishing gradient problem?](#what-is-batch-norm)
5. [What is optimizer? And how to choose?](#optimizer)
6. [What is dropout? Training vs inference](#dropout)
7. [what is the calibiration and how to use the calibration in ML?](#calibration)
8. [What’s the benefit of Transformer](#transformer)


### Bias Variance Tradeoff
It can be shown the $test error = bias  + variance + irreducible error$
Bias the amount by which the average of E[f(x0)] our estimate differs from the true mean, f(x0) is the out of sample prediction, if we fit f(.) using different training set, we may get different values of f(x0); 
The more flexible the model, the lower the bias.
Variance mean if we change the training data, how much the fitted model will be changed. The
expected squared deviation of ˆ f(x0) around its mean. The more flexible the model, the higher the variance.
So in order to reduce the test error, we need to decide the optimal model flexibility, giving us the lowest bias + variance.
<br><br><br>

### Bayes classifier
This is the best error rate we can achieve for classifier. If we know the distribution of each class, we assign the label to the class with the highest probability can give us the lowest error rate. This is the irreducible error rate we can get. max_g P(G = g|X=x).
<br><br><br>

### F test
F test = reduced Variance / unexplained Variance, where reduced $Variance = (RSS_H0 - RSS_Ha)/(n-p)$, unexplained Variance = RSS_Ha/(n - p). If reduced variance is big(F score high), than means the H0 restrict the model produce a negative effect on the fit, hence reject H0. Why F test, instead of checking the p value of each variable?
- When p >> 1, there is 5% chance of variables has p value > 0.05, while these variable is not significant actually. F test doesn't have this issue, since it penalize the number of variables by  devided by p.
<br><br><br>

### Why cannot use Least Square for Classification problem?
  1) if more than three classes, use 1, 2, 3, ... encoding implies the ordering.
  2) if binary problem, y can be viewed as a approximationg to P(y = 1|x). However, y can be outside range[0, 1], making it hard to interoperate. => Remediation: perform logistic transform on beta*x, s.t. output belongs to (0, 1).
  3) Finally, it can be shown that OLS + binary Y == LDA. 但若用logistic 来model multi-class =》可能出现masking.
<br><br><br>

### How to decide logistic regression goodness of fit?
Like OLS, we can also estimate SE(beta), so that beta_^ / SE(beta) follow t distribution, hence t-stats, p-value.
<br><br><br>

### Logistic Regression for multi-class classification.
可行， 但不如lda 普及。
<br><br><br>

### Why ROC/AUC?
In some scenario, we care more about False positive than true negative, or vise versa. But the Bayes decision boundary will only assign label to the class with >50% probability, minimize the total error rate. ROC give us the perform under different threshold. AUC is a good way to compare different model.
<br><br><br>

### Why k-fold can have better test error estimation than LOO?
When LOO, the training data for each fit is highly overlapped, therefore we get n highly correlated model, and the variance will be high. Since we are using almost all the training data, the bias is low. On the other hand, when k-fold, the data has lower variance and higher bias. 10-fold achieve a good tradeoff between bias and variance.
<br><br><br>

### Curse of Dim
- from ESL p22
when p is high, all sample points are close to the edge of the sample. 
Another way to interpret is to take fraction r of the space, average length of each dim is r^(1/p) which is very close to 1.
As a more concrete example, consider estimate f(x) = e^(-X_T*X) at x = 0(hill shape), the bigger the dim, the more close to the edge of each neighboring sample. Hence big bias.
To remedy, we need more samples, for example, each dim can take k discrete value, to numerate all possible combination we need k ^p. 
对于线性假设成立 的情况下， ols zero bias, low variance; otherwise, 1NN can be better at low bias.
-from ISL p244
线性模型下,当n < p, ols 会产生perfect fit, 此时需要feature selection.
<br><br><br>

### How to handle imbalanced data
a)	Try Changing Your Performance Metric - ROC Curves/F1/Confusion Matrix
b) Try Resampling Your Dataset
c) Try Different Algorithms - Tree Model
d)Try Penalized Models - Penalized SVM/LDA
e) Weighted Loss Function
<br><br><br>

### Feature Selection
除了forward/backward selection, 还有mixed selection(from ISL): 从空集开始，每步加变量， 
if at any point the p-value for one of the variables in the model rises above a certain
threshold, then we remove that variable from the model（类似于backward selection）.
<br><br><br>

### Why not p-value to determine feature importance?
1) correlated features may have both insigifinicant p-value
2) p-value is also impacted by sample size.	
<br><br><br>

### Why Ridge Works?
In general, in situations where the relationship between the response and the predictors is close to linear, the least squares estimates will have low bias but may have high variance. This means that a small change in the training data can cause a large change in the least squares coefficient estimates. In particular, when the number of variables p is almost as large as the number of observations n, ridge regression can still perform well by trading off a small increase in bias for a large decrease in variance. Hence, ridge regression works best in situations where the least squares estimates have high variance.
<br><br><br>

### More on Lasso.
The lasso is the posterior mode for β under a double-exponential prior.
若 X 为identity $matrix(y_j = b_j)$: In ridge regression, each least squares coefficient estimate is shrunken by the same proportion. 
In contrast, the lasso moves each least squares coefficient towards zero by a constant amount, λ/2; the least squares coefficients that are less than λ/2 in absolute value are shrunken entirely to zero.
<br><br><br>

### How to prune a tree?
For each value of α there corresponds a subtree T ⊂ T0 such that is as small as possible.
<br><br><br>

### How feature importance in trees is calculated?
In the case of bagging regression trees, we can record the total amount that the RSS (8.1) is decreased due to splits over a given predictor, averaged over all B trees. A large value indicates an important predictor. Similarly, in the context of bagging classification trees, we can add up the total amount that the Gini index (8.6) is decreased by splits over a given predictor, averaged over all B trees.
<br><br><br>

### Boosting Tree the tunning param:
Boosting has three tuning parameters:
1. The number of trees B. We use cross-validation to select B.
2. The shrinkage parameter λ, a small positive number. This controls the
rate at which boosting learns. Typical values are 0.01 or 0.001.
3. The number d of splits in each tree(depth)
<br><br><br>

### SVM
variance & bias trade off 主要由C决定， C代表了total violation budget. C越宽， 越多的店在margin之内， 更多support vectors => model 更robust.
<br><br><br>

### Kernel Tricks
We may want to enlarge our feature space in order to accommodate a non-linear boundary between the classes. The kernel approach that we describe here is simply an efficient computational approach for enacting this idea, where one need only compute C(n,2)
distinct pairs. A kernel is a function that quantifies the similarity of two observations.
For some kernels, such as the radial kernel (9.24), the feature space is implicit and infinite-dimensional, so we could never do the computations there anyway! The solution can always be written in the form of below.
<br><br><br>

### What is MLE and assumption.
In stats, maximum likelihood estimation (MLE) is a method of estimating the parameters of an assumed probability distribution, given some observed data. This is achieved by maximizing a likelihood function so that, under the assumed statistical model, the observed data is most probable.
<br><br><br>

### What is Overfitting and how to avoid it?
* More data
* More samples
* Data augmentation
<br\>
**Less complicated model:**
* reduce parameter, less layer
* Regularization
* Dropout
* Early stopping
<br><br><br>

### What is Overfitting and how to avoid it?
More data
More samples
Data augmentation
Less complicated model:
reduce parameter, less layer
Regularization
Dropout
Early stopping
<br><br><br>

## What is MAP and NDCG?
- MAP is the Average Precision across all users, the 
$$Average \\ Precision = PR \\ AUC$$
* Pros: 
  - one single metrics
  - Handle the ranking of lists naturally
  - Give more weights to error that appear at the top of the list
* Cons:
  - Binary(relevant/not relevant)
  - Not using fine-grained information


<img src="https://github.com/swchen1234/MLStudyNotes/assets/23247998/d37f75d7-f9ea-4252-8498-cad878b88c79" width="600">

- NDCG

<img src="https://github.com/swchen1234/MLStudyNotes/assets/23247998/695ee554-a6f6-4fbb-b1c5-fb43389364f1" width="600">
<img src="https://github.com/swchen1234/MLStudyNotes/assets/23247998/7c6eefa6-58dd-4b52-bd06-c2bcdd7f1d80" width="600">

* Pros:
  - Able to use binary/fine-grained(numerical) information(e.g. very relevent/medium relevant).
  - NDCG give still give some weight to the bottom of the list, while MAP give more weights to top rank.
  - MAP can only handle binary as well.

* ref [Rank Aware Recsys Evaluation Metrics](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)

<br><br><br>


### L1 Norm vs L2 Norm

* L1(Lasso Regression) will generate sparse feature and can be used for feature selection.
* L2(Ridge Regression) prevent overfitting and improve model’s generalization ability. 
<br><br><br>

### What is Vanishing Gradient?
When model is deep enough, certain activation functions such as sigmoid have a very small gradient, when stack them together will lead to a very small gradient, meaning a large change in input will only cause a small change in output. To overcome this,
* Relu has a larger gradient
* Batch Norm(#what-is-batch-norm)
* Initialization
* Residual Net
* ref (https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
<br><br><br>

### What is Batch Norm?
Batch normalization is a technique that normalizes the inputs to each layer within a mini-batch. By normalizing the inputs, it reduces the internal covariate shift and helps maintain a stable gradient flow. Batch normalization alleviates the vanishing gradient problem by ensuring that the gradients do not vanish or explode during training.
<br><br><br>

### Optimizer
A too small learning cate may be too slow to converge or do not converge due to diminishing gradient. A too big learning rate may overshoot the target and oscillate. 

Typically adaptive learning rate algorithm can be ADAM, RMSProp, Adagrad.
* ref [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam)
<br><br><br>

### Dropout
**Training**
The term “dropout” refers to dropping out the nodes (input and hidden layer) in a neural network (as seen in Figure 1). All the forward and backwards connections with a dropped node are temporarily removed, thus creating a new network architecture out of the parent network. The nodes are dropped by a dropout probability of p.
**Inference**
According to the original implementation (Figure 3b) during the inference, we do not use a dropout layer. This means that all the units are considered during the prediction step. But, because of taking all the units/neurons from a layer, the final weights will be larger than expected and to deal with this problem, weights are first scaled by the chosen dropout rate. With this, the network would be able to make accurate predictions.
To be more precise, if a unit is retained with probability p during training, the outgoing weights of that unit are multiplied by p during the prediction stage.
<br><br><br>

### Calibration
这个很笼统的问题，大概的意思，就是模型出来的score，不是服从你的数据的label的分布的，你需要重新map model output to the proper distribution.
When Calibrate?
To interpret the output of such a model in terms of a probability, we need to calibrate the model.
When not calibrate?
If you cares about the ranking of the result

To summarize, we would expect a calibrated model to have a lower log-loss than one that is not calibrated well.

* Calibration Method:
  - **_Isotonic Regression_** fits a non-decreasing line to a sequence of points in such a way as to make the line as close to the original points as possible.
* Evaluation
  - **_ECE_** is calculated as a weighted average of the accuracy/prediction error across the bins, weighted on the relative number of samples in each bin.
<br><br><br>

### Transformer
* Parallel Processing
* Better at long term memory (e.g. in a seq2seq model, traditionally, all the input information will be encoded in the final state and be the input of decoder, this is not good for long sentences. Attention solve this by utilization all the hidden states in encoder)
* Interpretable
<br><br><br>
