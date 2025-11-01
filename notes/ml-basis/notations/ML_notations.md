---
title: "ML Notations"
permalink: /notes/ml-basis/ml-notations
layout: single
author_profile: true
---


> [!tldr] 
> Just to define the notations


## Data points
If the data is quantitative, it can be presented using a matrix of size $(n, p)$ where
	- Each row represents a data point (observation / individual)
	- Each column represents a variable
Usually data are denoted using $\mathbf{X}$, with:
$$X = \begin{bmatrix} 
    x_{11} & x_{12} & \dots & x_{1p} \\
    x_{21} & x_{22} & \dots & x_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    x_{n1} & x_{n2} & \dots & x_{np} 
\end{bmatrix}$$

Each data point \\(i\\) can be represented by a vector \\(\mathbf{x}_{i}\in \mathbb{R}^{p}\\) such that

$$
\mathbf{X} = \begin{pmatrix} 
\mathbf{x}_1^T \\
\mathbf{x}_2^T \\
\vdots \\
\mathbf{x}_i^T \\
\vdots \\
\mathbf{x}_n^T 
\end{pmatrix} = 
\begin{pmatrix}
    x_1^1 & x_1^2 & \dots & x_1^j & \dots & x_1^p \\
    x_2^1 & x_2^2 & \dots & x_2^j & \dots & x_2^p \\
    \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
    x_i^1 & x_i^2 & \dots & x_i^j & \dots & x_i^p \\
    \vdots & \vdots & \ddots & \vdots & \ddots & \vdots \\
    x_n^1 & x_n^2 & \dots & x_n^j & \dots & x_n^p 
\end{pmatrix}
,\mathbf{x}_{i}= \begin{bmatrix}x_{i}^{1} \\ x_{i}^{2} \\ \vdots \\ x_{i}^{j}\\ \vdots\\ x_{i}^{p} \end{bmatrix}
$$

## Variable / features

Each variable \\(j\\) can also be represented as a vector \\(\mathbf{x}^{j} \in \mathbb{R}^{n}\\) containing the values taken by the data points,
$$
\mathbf{x}^{(j)} = \begin{bmatrix} x_{1j} \\ x_{2j} \\ \vdots \\ x_{nj} \end{bmatrix} \in \mathbb{R}^{n}
$$

$$




\mathbf{X} = 
\begin{matrix}  
  \begin{matrix} \mathbf{x}^{1} & \mathbf{x}^{2}  & \dots &  \mathbf{x}^{j} \dots  & \mathbf{x}^{p} \end{matrix}\\
  \begin{pmatrix}
  x_{1}^{1} & x_{1}^{2} & \dots & x_{1}^{j} & \dots  & x_{1}^{p}\\
  x_{2}^{1} & x_{2}^{2} & \dots & x_{2}^{j} & \dots & x_{2}^{p}\\
  \vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
  x_{i}^{1} & x_{i}^{2} & \vdots  & x_{i}^{j} & \vdots & x_{i}^{p} \\ 
  \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ 
  x_{n}^{1} & x_{n}^{2}  & \dots  & x_{n}^{j}  &  \dots & x_{n}^{p}
  \end{pmatrix}
 \end{matrix}
, \mathbf{x}^{j} = \begin{bmatrix} 
  x_{1}^{j} \\ x_{2}^{j} \\ \vdots  \\ x_{i}^{j} \\ \vdots  \\ x_{n}^{j}
\end{bmatrix}





$$

## Weighted data

It may be interesting to weights data points depending on their importance. Each data point is associated with a weights \\(w_{i} >0\\) such that \\(\sum_{i=1}^{n}w_{i}=1\\).

The weights matrix \\(\mathbf{D}_{w}\\) or \\(\mathbf{W}\\) (depending on the field) is a diagonal matrix,
$$
\mathbf{D}_{w}= \text{diag}(w_{1}, w_{2},\dots, w_{n})= \begin{bmatrix}
w_{1} & 0 & \dots & 0 \\ 
0 & w_2 & \dots & 0 \\ 
\vdots & \vdots & \vdots & \vdots \\ 
0 & 0 & \dots & w_{n}
\end{bmatrix}
$$

The most frequent case is the case of uniform weights, meaning all data points have the same weights, \\(w_{i}=\frac{1}{n}\\).

## Empirical mean

The empirical mean of the feature \\(j\\) is:
$$
\bar{x}^{j}= \sum_{i=1}^{n}w_{i}x_{i}^{j} \quad \text{weighted sum of the columns $j$ over all $n$ data points}
$$
Or, with matrices:
$$
\bar{x}^{j}= \mathbf{x}^{j}\mathbf{D}_{w}\mathbf{1}_{n}
$$

## Center of gravity

Given a set of data points, the centre of gravity \\(\mathbf{g}\\) is given by:
$$
\mathbf{g} = \begin{bmatrix}\bar{x}^{1} & \bar{x}^2 & \dots & \bar{x}^{j} & \dots & \bar{x}^{p} \end{bmatrix}^T
$$
or with matrices:
$$
\mathbf{g} = \mathbf{X}^{T}\mathbf{D}_{w}\mathbf{1}_{n}
$$
We can denote the centered data \\(\mathbf{\tilde{X}} = \mathbf{X} - \mathbf{1}_{n}\mathbf{g}^{T}\\)

## Empirical variance

The empirical variance of feature \\(j\\) is:
$$
s_{j}^{2}=\text{var}(\mathbf{x}^{j}) = \sum_{i=1}^{n}w_{i}(x_{i}^{j}-\bar{x}^{j})^{2}=\sum_{i=1}^{n}w_{i}(x_{i}^{j})^{2}-(\bar{x}^{j})^{2} = \sum_{i=1}^{n}w_{i}(\tilde{x}_{i}^{j})^{2}=\text{var}(\mathbf{\tilde{x}}^{j})
$$
or with matrices:
$$
s_{j}^{2}= \mathbf{\tilde{x}}^{j^T}\mathbf{D}_{w}\mathbf{\tilde{x}}^{j}
$$

## Empirical covariance

The empirical covariance of features \\(j\\) and \\(k\\) is:
$$
s_{jk}=\text{cov}(\mathbf{x}^{j},\mathbf{x}^{k}) = \sum_{i=1}^{n}w_{i}(x_{i}^{j}- \bar{x}^{j})(x_{i}^{k}-\bar{x}^{k})=\sum_{i=1}^{n}w_{i}x_{i}^{j}x_{i}^{k}-\bar{x}^{j}\bar{x}^{k}=\sum_{i=1}^{n}w_{i}\tilde{x}_{i}^{j}\tilde{x}_{i}^{k}=\text{cov}(\mathbf{\tilde{x}}^{j}, \mathbf{\tilde{y}}^{k})
$$
Or with matrices:
$$
s_{jk}= \mathbf{\tilde{y}}^{j^{T}}\mathbf{D}_{w}\mathbf{y}^{k}
$$

## Coefficient of empirical correlation

The empirical coefficient of correlation between features \\(j\\) and \\(k\\) is given by:
$$
r_{kj}= \text{cor}(\mathbf{x}^{j}, \mathbf{x}^{k})= \frac{\text{cov}(\mathbf{x}^{j}, \mathbf{x}^{k})}{\sqrt{\text{var}(\mathbf{x}^{j})}\sqrt{\text{var}(\mathbf{x}^{k})}} = \frac{s_{jk}}{s_{j}s_{k}} =\text{cor}(\mathbf{\tilde{x}^{j}}, \mathbf{\tilde{x}}^{k} )
$$

By construction,
* \\(-1 \leq r_{jk} \leq 1\\)
* \\(\lvert r_{jk}\rvert=1 \iff\\) The features are almost surely linked by an affine relation
* \\(\lvert r_{jk}\rvert =0 \iff\\) The features are uncorrelated (no affine relation)

## Useful matrices

### Matrix of empirical covariances, $$V= V_{\mathbf{X}} = V_{\mathbf{\tilde{X}}}$$

$$
\mathbf{V} = \begin{bmatrix}
s_{1}^{2} & s_12 & \dots & s_1k & \dots & s_{1p} \\
s_21 & s_{2}^{2} & \dots & s_2k & \dots & s_{2p}\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
s_{j1} & s_{j2} & \dots & s_{jk} & \dots & s_{jp}\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
s_{p1} & s_{p2} & \dots & s_{pk} & \dots & s_{p}^{2}
\end{bmatrix}
$$

### Matrix of empirical correlation $$\mathbf{R} = \mathbf{R}_{\mathbf{X}} = \mathbf{R}_{\mathbf{\tilde{X}}}$$

$$
\mathbf{R} = \begin{bmatrix}
1 & r_{12} & \dots & r_1k & \dots & r_{1p}\\
r_21 & 1 & \dots & r_2k & \dots & r_2p\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
r_{j1} & r_{j2} & \dots & r_{jk} & \dots & r_{jp}\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
r_{p1} & r_{p2} & \dots & r_{pk} & \dots  & 1
\end{bmatrix}
$$

### Standardised data

$$\mathbf{Z} = \mathbf{\tilde{X}}\mathbf{D}_{1/s}$$

## Metric space of data points

To characterise the structure, we focus on the proximity between data points in \\(\mathbb{R}^{p}\\) using 
$$
d_{M}^{2}(\mathbf{x}_{i}, \mathbf{x}_{l}) = \lVert \mathbf{x}_{i}- \mathbf{x}_{l}\lVert_{M}^{2}=(\mathbf{x}_{i}- \mathbf{x}_{l})^{T}\mathbf{M} (\mathbf{x}_{i}- \mathbf{x}_{l})
$$