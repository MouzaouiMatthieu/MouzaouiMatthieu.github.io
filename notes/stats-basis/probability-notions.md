---
title: "Probability Notions"
permalink: /notes/stats-basis/probability-notions
layout: single
author_profile: true
---
> [!tldr] Summary
> Probabilities 101, just to be clear later. Maybe a more precise note will come later.

> [!quote] Quote
> .

## Introduction

Randomness seems to be a part of our daily lives. Or at least, we think so. Maybe we call 'random' events for which we lack the true understanding of causes leading to it. Take a coin toss for instance. We usually agree the results (head or tail) is random. Each event occur with a certain probability. One could argue, if given the coin mass, exact shape and initial applied force to the system, we could deterministically deduce the result of the coin toss. When discussing randomness at a philosophy tea-bar (if this ever happens to you, you never know), quantum mechanic we be quickly mentioned. It seems, the explanation of quantum mechanic could be non determinist (*Bell's theorem)*. Or we might just be far too ignorant on the matter yet. Nevertheless, there is a need to model this randomness.

Probability theory offers a mathematical models to apprehend the world, causes and effects. It is an efficient theory which can be applied to various domains, and is thus popular. 

## Randomness and Models

Some experiment do not produce necessarily the same results each time, whiles still exhibiting some regularity if repeated multiple times with the same conditions. Examples include flipping a coin, rolling a dice etc.
Since the experiment can have different results, the first thing to do is to define the set of possible outcomes. 

### Kolmogorov axioms

##### Definition: Universe

The set $$\Omega$$ of all possible outcomes of an experiment specified by a given experimental protocol is called the universe. We also say that $$\Omega$$ is the state space or the sample space of the random experiment.

##### Definition: Measurable Space

A measurable space is a pair $$(\Omega, \mathcal{T})$$ where $$\Omega$$ is a set and $$\mathcal{T}$$ is a $$\sigma$$-algebra of $$\Omega$$, i.e., a set of subsets of $$\Omega$$ satisfying the following properties:

1. $$\Omega \in \mathcal{T}$$.
2. If $$A \in \mathcal{T}$$, then $$\bar{A} \in \mathcal{T}$$ where $$\bar{A}$$ is the complement of $$A$$ in $$\Omega$$.
3. If $$(A_n)_{n \in \mathbb{N}^*}$$ is a sequence of elements of $$\mathcal{T}$$, then $$\bigcup_{n=1}^{\infty} A_n \in \mathcal{T}$$.

The elements of $$\mathcal{T}$$ are called events. In particular, for any $$\omega \in \Omega$$, the singleton $$\{\omega\}$$ is called an elementary event.

The three axioms of a probabilizable space can be intuitively explained as follows. Since the elements of $$\mathcal{T}$$ are sets $$A$$ for which $$\omega \in A$$ corresponds to a possible event of the experiment, and since we trivially have $$\omega \in \Omega$$, $$\Omega$$ can only be an element of $$\mathcal{T}$$. We say that $$\Omega$$ is the certain event.

Regarding the second axiom (stability by complementarity), $$A \in \mathcal{T}$$ means that $$\omega \in A$$ is a possible event that the experiment can produce. The opposite event $$\omega \in \bar{A}$$ is also a possible event for the experiment considered.

The third axiom (stability by countable union) simply means that a choice between different events $$\omega \in A_n$$ is still an event that can occur during the experiment. It should be noted that this axiom considers a countable union of elements of $$\mathcal{T}$$, and one might wonder if considering a finite union would suffice in practice. In fact, no. It is enough to consider the random experiment consisting of counting the number of requests to a server during a given time period. This number cannot be bounded since we can have $$1, 2, 3, \ldots$$ requests.

##### Definition: Probability Space

A probability space is a triplet $$(\Omega, \mathcal{T}, \mathbb{P})$$ where $$(\Omega, \mathcal{T})$$ is a measurable space and $$\mathbb{P}$$ is a probability measure on $$\mathcal{T}$$, i.e., a function from $$\mathcal{T}$$ to $$[0, 1]$$ such that:

1. $$\mathbb{P}(\Omega) = 1$$, Normalization Condition
2. For a sequence of pairwise disjoint events $$(A_n)_{n \in \mathbb{N}^*}$$:

$$
\mathbb{P}\left(\bigcup_{n=1}^{\infty} A_n\right) = \sum_{n=1}^{\infty} \mathbb{P}(A_n) \quad [\sigma\text{-additivity}]
$$

In many applications, $$\Omega$$ is $$\mathbb{R}$$. In this case, we do not choose the set of all subsets of $$\Omega = \mathbb{R}$$ as the $$\sigma$$-algebra. Indeed, this set of subsets is much too large to define a probability $$\mathbb{P}$$ on it. We use the $$\sigma$$-algebra $$\mathcal{B}(\mathbb{R})$$ of the Borel sets of $$\mathbb{R}$$, i.e., the smallest $$\sigma$$-algebra containing all intervals of $$\mathbb{R}$$. This $$\sigma$$-algebra is already very large and sufficient for practical applications. In the following two examples, $$\Omega$$ is countable. In this case, we can choose the set of all subsets of $$\Omega$$ as the $$\sigma$$-algebra.

### Conditional Probability

In practice, it is very often useful to know how to calculate the probability of an event $$A$$ conditional on or knowing the event $$B$$. For example, in a game of rolling a six-sided die, what is the probability that the result is 6 given that the result is even? In this question, we seek to calculate the conditional probability of the event $$A = \{6\}$$ given the event $$B = \{2, 4, 6\}$$. Since the rolls are equiprobable and there is only one chance in three of rolling a 6 among $$\{2, 4, 6\}$$, intuition tells us that the conditional probability of $$A$$ given $$B$$ is $$1 / 3$$. The general definition that allows us to retrieve this result is the following Bayes' axiom.

##### Definition: Conditional Probability

Let a probability space $$(\Omega, \mathcal{T}, \mathbb{P})$$ and an event $$A$$ such that $$\mathbb{P}(A) \neq 0$$. For any $$B \in \mathcal{T}$$, we define the conditional probability of $$B$$ given $$A$$ by:

$$
\mathbb{P}(B \mid A) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(A)} \quad [\text{Bayes' Axiom}]
$$

The following result illustrates the validity of Bayes' axiom.

##### Proposition

With the previous notations, the function that assigns to any $$B \in \mathcal{T}$$ the value $$\mathbb{P}(B \mid A)$$ is a probability.

##### Bayes' Formula

Suppose that $$\Omega = \bigcup_{n=1}^{\infty} A_n$$ where the $$A_n$$ are pairwise disjoint. Bayes' formula allows us to calculate the posterior probabilities $$\mathbb{P}(A_n \mid B)$$ from the values $$\mathbb{P}(A_n)$$ and $$\mathbb{P}(B \mid A_n)$$:

$$
\mathbb{P}(A_n \mid B) = \frac{\mathbb{P}(A_n) \mathbb{P}(B \mid A_n)}{\sum_{k=1}^{\infty} \mathbb{P}(A_k) \mathbb{P}(B \mid A_k)}
$$

###### Example

35% of Starfleet officers specialize in Command track. Among these Command track officers, 25% achieve Captain rank within 10 years, whereas for other specializations this figure is only 15%. What is the probability that a randomly selected officer achieves Captain rank within 10 years?

The probability is determined by applying the total probability formula. Let $$B = \{ \text{officer acchieves Captain rank within 10 years} \}$$ and $$A = \{ \text{officer is in Command track} \}$$, then:
$$
\mathbb{P}(B) = \mathbb{P}(B \lvert A) \mathbb{P}(A) + \mathbb{P}(B \lvert \bar{A})\mathbb{P}(\bar{A}) =0.25 \times 0.35 + 0.15 \times 0.65 = 0.185
$$
### Independence

##### Definition: Independent Events

Let a probability space $$(\Omega, \mathcal{T}, \mathbb{P})$$. We say that two events $$A$$ and $$B$$—i.e., 2 elements of $$\mathcal{T}$$—are independent if $$\mathbb{P}(A \cap B) = \mathbb{P}(A) \mathbb{P}(B)$$.

##### Proposition: Independence and Conditional Probability

With the notations of the previous definition, assume that $$\mathbb{P}(B) \neq 0$$. The events $$A$$ and $$B$$ are independent if and only if $$\mathbb{P}(A \mid B) = \mathbb{P}(A)$$.

In other words, two events are independent (and omitting the special case where one has zero probability) if the occurrence of one does not influence the probability of the occurrence of the other.

## Random Variables

##### Definition:  Real Random Variable

A real random variable defined on a probability space $$(\Omega, \mathcal{T})$$ is a function:
$$
\begin{align}
X : \quad &\Omega \to \mathbb{R}\\
    & \omega \to X(\omega)
\end{align}
$$
such that $$X^{-1}(I)$$ is an element of $$\mathcal{T}$$ for every interval $$I$$ of $$\mathcal{B}(\mathbb{R})$$. We say that this function is measurable. The set $$X(\Omega)$$ of values taken by the random variable $$X$$ is called the range of $$X$$.

###### Example
Consider an experiment where we randomly select starships from a Federation shipyard. A random variable could be the function that associates to each starship its fuel efficiency (in warp cores per light-year), its maximum speed, or any other quantity defining it. This is therefore a function from $$\Omega$$ (the set of all possible starships) to $$\mathbb{R}$$. 


##### Definition: Probability Law

The probability law of $$X$$, denoted $$\mathbb{P}_{X}$$ is the image probability of $$\mathbb{P}$$ by $$X$$:
$$
\forall B \in \mathcal{B}, \mathbb{P}_{X}(B)
 = \mathbb{P}\left( X^{-1}(B) \right) = \mathbb{P}(X \in B)
$$

The distribution of a real random variable is entirely characterised by the cumulative distribution function of this variable

##### Cumulative Distribution Function

The cumulative distribution function $$\mathbb{F}_{X}$$ of a real random variable $$X$$ is defined by:
$$
\forall x \in \mathbb{R}, \mathbb{F}_{X}(x) = \mathbb{P}\left( X^{-1}(] - \infty, x) \right) = \mathbb{P}(X \leq x)
$$

The cumulative distribution function satisfies the following:
- it always exists.
- is increasing
- is right continuous
- $$\underset{x \to \infty}{\lim}\mathbb{F}_{X}(x) = 1$$ and $$\underset{x \to - \infty}{\lim}\mathbb{F}_{X}(x) = 0$$   

##### Definition: Probability Density Function (pdf)
A real valued random variable with a CDF $$\mathbb{F}_{X}$$ has a **density** if there exists a function $$f_{X} : \mathbb{R} \to [0, + \infty ]$$ such that: 
$$
\int_{\mathbb{R}}f_{X}(t)\mathrm{d}t = 1 \quad \text{and } \quad \mathbb{F}_{X}(x) = \int_{- \infty}^{x}f_{X}(t)\mathrm{d}t
$$

If a random variable $$X$$ has a density $$f_{X}$$, it is said the CDF $$\mathbb{F}_{X}$$ is absolutely continuous. Not all CDF are absolutely continuous. Informally, we can directly say $$X$$ is absolutely continuous. 

##### Proposition

If a random variable $$X$$ is absolutely continuous, then the CDF $$\mathbb{F}_{X}$$ is continuous and a density $$f_{X}$$ of $$X$$ is given by:
$$
f_{X}(x) = \mathbb{F}_{X}^{\prime} \quad \text{a.e.}
$$

##### Proposition

Let $$(a,b) \in \mathbb{R}^{2}$$ s.t. $$a \leq b$$. Any absolutely continuous real random variable $$X$$ with density $$f_{X}$$ and CDF $$\mathbb{F}_{X}$$ satistifes the following:

$$
\begin{align}
\mathbb{P}(X=a) &=0\\
\mathbb{P}(X \leq a) = \mathbb{P}(X < a) = \int_{- \infty}^{a} f_{X}(x)\mathrm{d}x
\mathbb{P}(a \leq X) &= \mathbb{P}(a < X) = \int_{a}^{\infty} f_{X}(x)\mathrm{d}x\\
\mathbb{P}(a < X \leq b) &= \mathbb{P}(a \leq X \leq b) = \mathbb{P}(a \leq X < b) = \mathbb{P}(a < X < b) = \int_{a}^{b} f(x)\mathrm{d}x = \mathbf{F}_{X}(b) - \mathbf{F}_{X}(a)
\end{align}
$$

### Moments

##### Definition

For  a random variable $$X$$, the $$k$$-th moment is defined as $$\mathbb{E}(X^{k})$$, provided it exists:
- For discrete variables: $$\mathbb{E}(X^{k}) = \sum_{i} x_{i}^{k}\mathbb{P}(X = x_{i})$$
- For continuous variables: $$\mathbb{E}(X^{k}) = \int_{- \infty}^{\infty} x^{k}f_{X}(x)\mathrm{d}x$$ 
The $$k$$-th moment exists if $$\mathbb{E}(\lvert X^{k} \lvert) < \infty$$. The first two moments are particularly importants in statistics: the **expectation** ($$k$$=1) and the **variance** ($$k$$=2). 

##### Variance

Variance measures the dispersion of $$X$$ and is defined as 
$$
\mathbb{V}(X) = \mathbb{E}((X - \mathbb{E}(X))^{2}) - \mathbb{E}(X^{2}) - \mathbb{E}^{2}(X)
$$
When $$\mathbb{E}(X) = 0$$ the random variable is said to be centered, and when $$\mathbb{V}(X) = 1$$ is is said to be standardised. 

##### Theorem (Transfer)

Let $$X$$ be a real random variable and $$g$$ a function such that $$\mathbb{E}(\lvert g(X) \lvert) < \infty$$:
- For a discrete case: $$\mathbb{E}(g(X)) = \sum_{i}\mathbb{P}(X = x_{i})$$
- For a continuous case: $$\mathbb{E}(g(X)) =\int_{-\infty}^{\infty} g(x)f(x)\mathrm{d}x$$. 

### Usual probability distributions and Random Variables


#### Bernoulli

A random variable $$X$$ is said to follow a Bernoulli distribution if there exists a number $$p \in [0,1]$$ scuh that the probability distribution of $$X$$ is given by $$\mathbb{P}(X=1) =p$$ and $$\mathbb{P}(X = 0) = 1-p$$. We denote the Bernoulli distribution as $$\mathcal{B}(0,1)$$. The expectation and variance of $$X$$ are $$\mathbb{E}(X)=p$$ and $$\mathbb{V}(X) = p(1-p)$$.
It is used to model 'success' or 'failure' experiments (with binary results).

#### Geometric Distribution

A random variable $$X$$ that takes value un $$\mathbb{N}^{*}$$ follows a geometric distribution with parameter $$p\in [0,1]$$ if its probability distribution is $$\mathbb{P}(N=k) = (1 - p)^{k-1}p$$. The expectation of $$X$$ is $$\mathbb{E}(X) = \frac{1}{p}$$ and its variance is $$\mathbb{V}(X) = \frac{1 -p}{p^{2}}$$. 
It  is used to rank the first success in a series of independent Bernoulli trials. 

#### Binomial Distribution

 A random variable $$X$$ follows a binomial distribution with parameters $$(n, p)$$ where $$n \in \mathbb{N}$$ and $$p \in [0,1]$$, if its probability distribution is $$\mathbb{P}(X=k) = C_{n}^{k}p^{k}(1-p)^{n-k}$$ for $$k\in \{0,1,\ldots,n\}$$. We denote the Binomial distribution as $$\mathcal{B}(n, p)$$. The expectation and variances of $$X$$ are $$\mathbb{E}(X) = np$$ and $$\mathbb{V}(X) = np (1 - p)$$.
The binomial distribution models the number of successes in a series of $$n$$ independent Bernoulli trials.

#### Poisson Distribution

 A random variable $$X$$ follows a Poisson distribution with parameter $$\lambda > 0$$ if its probability distribution is $$\mathbb{P}(X = k) = e^{- \lambda} \frac{\lambda^{k}}{k!}$$ for $$k = 0,1,2, \ldots,$$. We denote the Poisson distribution as $$\mathcal{P}(\lambda)$$. The expectation and variance of $$X$$ are $$\mathbb{E}(X) = \lambda = \mathbb{V}(X)$$. 

The Poisson distribution models the number of event occurring in a fixed time interval when events occur independently of the time elapsed since the last event. 

#### Uniform Distribution

A random variable $$X$$ follows a uniform distribution with parameters $$(a,b)$$ where $$- \infty < a < b < \infty$$ if it has the probability density function $$f$$ being 
$$
f_{X}(x) = 
\begin{cases}   
\frac{1}{b - a} \quad &\text{if}\ x \in [a, b] \\
0 \quad &\text{otherwise}
\end{cases}
$$
We denote the uniform distribution as $$\mathcal{U}([a,b])$$. The expectation and variances of $$X$$ are $$\mathbb{E}(X) = \frac{a+b}{2}$$ and $$\mathbb{V}(X) = \frac{(b - a)^{2}}{12}$$.

This distribution extends the concept of equiprobability from the discrete domain.

#### Normal or Gaussian distribution

A random variable $$X$$ follows a normal or a Gaussian distribution with parameters $$(\mu, \sigma^{2})$$ where $$\mu \in \mathbb{R}$$ and $$\sigma \in (0, + \infty)$$ if it has the probability density function $$f_X$$ defined as 
$$
f_{X} = \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{- \frac{(x-\mu)^{2}}{2 \pi^{2}}}
$$

We denote the normal distribution as $$\mathcal{N}(\mu, \sigma^{2})$$. The expectation and variance  of $$X$$ are $$\mathbb{E}(X) = \mu$$ and $$\mathbb{V}(X) = \sigma^{2}$$. 

This distribution is ubiquitous in statistics.


## Pairs of random variables


We often deal with situation involving multiple random variables. Specifically when we consider two random variables together, we refer to them as pair of random variables.

### Joint distribution and Expectation

Consider a probability space $$(\Omega, \mathcal{T}, \mathbb{P})$$ and two real random variables defined of $$(\Omega, \mathcal{T})$$. The joint distribution of the pair $$(X, Y)$$ is the probability measure defined on $$(\mathbb{R}^{2}, \mathcal{B}(\mathbb{R}^{2}))$$ by:
$$
\mathbb{P}_{X,Y}(C) = \mathbb{P}( (X,Y) \in C),\ \forall C \in \mathcal{B}(\mathbb{R}^{2})
$$

To know $$\mathbb{P}_{X,Y}$$ it is sufficient to know the probabilities $$\mathbb{P}_{X,Y}(A \times B) = \mathbb{P}((X, Y) \in A \times B)$$ for all events $$A$$ and $$B$$ of $$\mathcal{T}$$. Knowing the joint cumulative distribution function defined for all $$(x,y) \in \mathbb{R}^{2}$$ by:
$$
\mathbb{F}_{X,Y}(x,y) = \mathbb{P}((X,Y) \in (- \infty, x] \times - (\infty, y]) = \mathbb{P}(X \leq x \leq y)
$$
##### Theorem (transfer for pairs of random variables)

Let $$(X,Y)$$ be a pair of real random variables and $$g$$ be a function such that $$\mathbb{E}(\lvert g(X,Y) \lvert) < \infty$$, then:
- For discrete cases: $$\mathbb{E}(g(X,Y)) = \sum_{(i,j) \in K(C)} g(x_{i},y_{j}) \mathbb{P}(X = x_{i}, Y = y_{j})$$
- For continuous cases: $$\mathbb{E}(g(X,Y)) = \int \int_{\mathbb{R}^{2}} g(x,y) f_{X,Y} \mathrm{d}x\mathrm{d}y$$. 


### Marginal distributions

Consider a probability space $$(\Omega, \mathcal{T}, \mathbb{P})$$ and a pair $$X,Y$$ of real random variables defined on $$\Omega, \mathcal{T}$$. The distributions of $$X$$ and $$Y$$ are called the marginal distributions of the pair $$(X,Y)$$. 
For any $$A \in \mathcal{T}$$:
$$
\mathbb{P}_{X}(A) = \mathbb{P}(X \in A, Y \in A) = \mathbb{P}( (X,Y) \in A \times \mathbb{R}) 
$$
this allows to show that for all $$x \in \mathbb{R}$$:
$$
\mathbb{F}_{X}(x) = \underset{y \to \infty}{\lim} \mathbb{F}_{X,Y}(x,y)
$$

Similarly, for any $$B \in \mathcal{T}$$:
$$
\mathbb{P}_{y}(B):
\mathbb{P}
$$

##### Proposition:

Let $$X$$ and $$Y$$ be two real random variables with discrete values. We denote $$\{x_{i}: i \in I \}$$ and $$\{y_{j}: j \in J \}$$ as the respective ranges of these random variables, where $$I \subseteq \mathbb{N}^{*}$$ and $$J \subseteq \mathbb{N}^{*}$$. Assume that the values of $$x_{i}$$ and $$y_{j}$$ are distinct. 
$$
\mathbb{P}(X = x_{i}) = \sum\limits_{j \in J}\mathbb{P}(X = x_{i}, Y = y_{j})
$$

##### Proposition

In the case of an absolutely continuous pair $$(X,Y)$$, the random variables $$X$$ and $$Y$$ are also absolutely continuous. A density of $$X$$ is then:
$$
f_{X}(x)  = \int_{- \infty}^{\infty}f_{X,Y}(x,y) \mathrm{d}y
$$
and the density of $$Y$$ is:
$$
f_{Y}(y) = \int_{- \infty}^{\infty}f_{X,Y}(x,y)\mathrm{d}x
$$
These densities are called respectively marginal density of  $$X$$ and $$Y$$.


We now how to find the distributions of $$X$$ and $$Y$$ from the distribution of the pair $$(X,Y)$$. What about the converse? Without additional information, knowing only the distribution of $$X$$ and $$Y$$ is not sufficient to deduce that of $$(X,Y)$$. 


### Conditional distributions

The concept of conditional distribution helps describe how one random variable influences another. Specifically, given a probability space $$(\Omega, \mathcal{T}, \mathbb{P})$$ and two random variables $$X$$ and $$Y$$ defined on $$(\Omega, \mathcal{T})$$, we want to define and compute the conditional distribution $$\mathbb{P}_{Y}( \cdot \lvert X = x)$$ of $$Y$$ given $$\{ X = x \}$$. As long as $$\mathbb{P}(X = x) \neq 0$$ Bayes' axiom is sufficient and allows to define:
$$
\mathbb{P}_{Y}(A \lvert X = x) =
\frac{
\mathbb{P}( \{ Y \in A \} \cap \{X = x\} )
} 
{\mathbb{P}(X = x)}
$$
$$\forall A \in \mathcal{T}$$. 


##### Definitio:, conditional density

Let $$(\Omega, \mathcal{T}, \mathbb{P})$$ be a probability space and $$(X, Y)$$ be an absolutely continuous pair of random variables defined on $$(\Omega, \mathcal{T})$$. If $$f_{X}$$ is the marginal density of $$X$$, the conditional distribution of $$Y$$ given $$\{X = x \}$$ is defined for all $$x$$ where $$f_{X}(x) \neq 0$$ by its density:
$$
f_{Y \lvert X=x}(y) = \frac{f_{X,Y}(x,y)}{f_{X}(x)}
$$
##### Definition: Conditional expectation

We define the conditional expectation of $$Y$$ given $$\{X = x\}$$ as:
$$
\mathbb{E}(Y \lvert X =x) = \int_{-\infty}^{\infty}y f_{Y \lvert X = x}(y) \mathrm{d}y.
$$

##### Proposition: Law of iterated expectations
$$
\mathbb{E}(Y) = \int_{-\infty}^{\infty}\mathbb{E}(Y \lvert X = x)f_{X}(x)\mathrm{d}x
$$

### Independent Random Variables

##### Definition

Let $$(\Omega, \mathcal{T}, \mathbb{P})$$ be a probability space. We say that two random variables $$X$$ and $$Y$$ on the probability space $$(\Omega, \mathcal{T})$$ are independent if for all $$A$$ and $$B$$ of $$\mathcal{B}(\mathbb{R})$$, the events $$X \in A$$ and $$Y \in B$$ are independent.

The independence of $$X$$ and $$Y$$ is expressed by:
$$
\mathbb{P}(X \in A, Y \in B) = \mathbb{P}(X \in A) \mathbb{P}(Y \in B)
$$
also written as
$$
\mathbb{P}_{X,Y}(A \times B) = \mathbb{P}_X(A)\mathbb{P}_{Y}(B)
$$

##### Theorem:
With the previous notations:

1. $$X$$ and $$Y$$ are independent if and only if $$\mathbb{F}_{X,Y}(x,y) = \mathbb{F}_{X}(x)\mathbb{F}_{Y}(y)$$ for all $$(x,y) \in \mathbb{R}^{2}$$ where $$\mathbb{F}_{X}, \mathbb{F}_{Y}, \mathbb{F}_{X,Y}$$ are the respective CDF of $$X, Y$$ and $$(X,Y)$$
2. If $$X$$ and $$Y$$ are discrete with respective ranges $$\{ x_{i}: i \in I \}$$ and $$\{y_{j}: j \in J \}$$ where $$I \subseteq \mathbb{N}^{*}, \ J \subseteq \mathbb{N}^{*}$$ and the $$x_{i}$$ (resp. $$y_{i}$$) are distinct, $$X$$ and $$Y$$ are independent if and only if $$\mathbb{P}(X = x_{i}, Y = y_{j}) = \mathbb{P}(X = x_{i}) \mathbb{P}(Y = y_{j})$$ for all $$(i,j) \in I \times J$$
3. If the pair $$(X,Y)$$ is absolutely continuous, $$X$$ and $$Y$$ are independant if and only if $$f_{X,Y}(x,y)=  f_{X}(x)f_{Y}(y)$$ is a probability density of $$(X,Y)$$ for any density $$f_{X}$$ of $$X$$ and any density $$f_{Y}$$ of $$Y$$. 

##### Proposition

Let $$X + Y$$ be a sum of two independent random variables $$X$$ and $$Y$$. We have:

1. If $$X$$ and $$Y$$ are discrete and have a common range $$\mathbb{N}^{*}$$, then:
$$
\mathbb{P}(X+  Y = k) = \left( \mathbb{P}_{X} * \mathbb{P}_{Y}\right(k) ) = \sum\limits_{i=1}^{\infty}\mathbb{P}(X = k - i) \mathbb{P}(Y = i)
$$
2. If $$X$$ and $$Y$$ are absolutely continuous with respective densities $$f_X$$ and $$f_{Y}$$, then $$X+Y$$ is also absolutely continuous and has the convolution $$f_{X+Y} = f_{X} * f_{Y}$$ of the densities of $$X$$ and $$Y$$ as its density. Therefore, we have:
$$
\forall s \in \mathbb{R}, \quad f_{X+Y}(s) = \int_{-\infty}^{\infty}f_{X}(s-y)f_{Y}(y)\mathrm{d}y = \int_{-\infty}^{\infty}f_{X}(x)f_{Y}(s-x)\mathrm{d}x
$$

### Covariance and correlation coefficient

The independence between random variable is a strong assumption. This assumption is not always verified. A weaker notion is that of uncorrelatedness.

##### Definition: Covariance and correlation coefficient

Let two random variables $$X$$ and $$Y$$.

1. The covariance of $$X$$ and $$Y$$ is defined by:
	$$
	\mathbb{C}ov(X, Y) = \mathbb{E}\left( 
	( X - \mathbb{E}(X))
	\right)
	(
	Y - \mathbb{E}(Y)
	)
	= \mathbb{E}(XY) - \mathbb{E}(X)\mathbb{E}(Y)
$$
2. We say that $$X$$ and $$Y$$ are uncorrelated if $$\mathbb{C}ov(X,Y)=0$$
3. The correlation coefficient of $$X$$ and $$Y$$ with non-zero variances is defined by:
	$$
	\rho(X,Y) = \frac{\mathbb{C}ov(X,Y)}{\sqrt{\mathbb{V}(X) \mathbb{V}(Y)}}
$$

##### Proposition

1. If $$X$$ and $$Y$$ are independent, $$\mathbb{C}ov(X,Y) = \rho(X,Y) = 0$$. Independence is a stronger hypothesis than uncorrelatedness, as independence implies uncorrelatedness. The converse is generally false.
2. The correlation coefficient takes its values in the interval $$[-1,1]$$:
	$$ 
	\lvert \rho(X,Y) \lvert \leq 1
	$$
3. $$X$$ and $$Y$$ are almost surely linked by an affine relation of the form $$Y = aX + b$$ if and only if $$\lvert \rho(X,Y) \lvert = 1$$ 

##### Proposition

Consider the sum $$X+Y$$ of two uncorrelated random variables $$X$$ and $$Y$$. The variance of this sum is the sum of the variances of $$X$$ and $$Y$$:
$$
\mathbb{V}(X + Y) = \mathbb{V}(X) + \mathbb{V}(Y)
$$

## Convergences of sequences of real random variables

The convergences of sequences of random variables play an extremely important role in probability theory for at least two fundamental reasons. First, an important convergence result known as the "law of large numbers" allows us to establish a formal link between the theoretical notion of probability and its intuitive interpretation as the frequency of occurrence of an event, when a random experiment is repeated infinitely in an independent manner. The second reason is that, through the famous central limit theorem, the notion of convergence in distribution allows us to justify the omnipresence of the Gaussian probability distribution in statistics.

### Convergence in distribution

Convergence in distribution is the weakest mode of convergence in the sense that it does not imply any of the other modes of convergence presented in this chapter. It defines a neighborhood relationship not between the random variables themselves but between their cumulative distribution functions


##### Definition: Convergence in distribution

A sequence of random variables $$(X_n)_{n \in \mathbb{N}^{*}}$$ converges in distribution to $$X$$ if 
$$
\underset{n \to \infty}{\lim} \mathbb{F}_{X_{n}}(x) = \mathbb{F}_{X}(x)   
$$
at every point $$x$$ where $$\mathbb{F}_{X}$$ is continuous. This convergence is denoted $$X_{n} \overset{d}{\rightarrow} X$$. 

##### Theorem: Central Limite Theorem (CLT)

If $$(X_{n})_{n \in \mathbb{N}^{*}}$$ is a sequence if i.i.d. random variables with the same (finite) expectation $$\mu$$ and the same (finite) standard deviation $$\sigma >0$$, then, by setting $$\bar{X} = \frac{1}{n} \sum\limits_{i=1}^{n}X_{i}$$, we have:
$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}}\overset{d}{\rightarrow}\mathcal{N}(0,1)
$$


In practice, this theorem is very useful because it allows us to say that, for sufficiently large $$n$$, a sum of i.i.d. random variables approximately follows a normal distribution. 


The convergence in distribution of the sequence $$(X_n)$$ ​ to $$X$$ does not tell us anything about the gap that may or may not exist between $$X_n$$​ and $$X$$  when $$x$$ is large. In other words, it is not because the sequence  converges in distribution to  that the values of $$X_{n}$$ ​ are close to those of $$X$$. To describe this gap, we must define other modes of convergence.

### Convergence in Probability

A sequence of  random variables $$(X_n)_{n \in \mathbb{N}^{*}}$$ converges in probability to $$X$$ if for every real $$\varepsilon > 0$$, 
$$
\underset{n \to \infty}{\lim} \mathbb{P}(\lvert X_{n}- X \vert) \geq \varepsilon ) = 0. 
$$

This convergence is denoted: $$X_{n}\overset{\mathbb{P}}{\rightarrow} X$$. 

The convergence in probability is stronger than convergence in distribution because there is an implication relation between the two: 
$$(X_{n}\overset{\mathbb{P}}{\rightarrow} X) \implies (X_{n} \overset{d}{\rightarrow} X)$$. An example of this kind of convergence is the weak law of large numbers:

##### Theorem: Weak law of large numbers

If $$(X_{n})_{n \in \mathbb{N}^{*}}$$ is a sequence of i.i.d. random variables with the same expectation $$\mu$$ and same standard deviation $$\sigma >0$$, then, by setting $$\bar{X} = \frac{1}{n} \sum\limits_{i=1}^{n} X_{i}$$:
$$
\bar{X} \overset{\mathbb{P}}{\rightarrow} \mu
$$
This theorem plays a very important role in statistics because it allows us to justify the use of the empirical mean as an "estimator" of the expectation of a random variable

### Convergence in mean square

This kind of convergence appears regularly in the justification of the choice of an estimator. 

##### Definition: Convergence in Mean Square

A sequence of random variables $$(X_{n})_{n \in \mathbb{N}^{*}}$$ converges in mean square to $$X$$ if:
$$
\underset{n \to \infty}{\lim} \mathbb{E}((X_{n}-X)^{2})= 0
$$
This converge is denoted: $$X_{n}\overset{m.s.}{\rightarrow} X$$. 

This convergence is stronger than convergence in probability. 

### Other mode of Convergence

##### Definition: Convergence in Mean of Order $$q$$

A sequence of random variables $$(X_{n})_{n \in \mathbb{N}^{*}}$$ converges in mean of order $$q$$ to $$X$$ if, 
$$
\underset{n \to \infty}{\lim} \mathbb{E}( \lvert X_{n} - X \lvert^{q}) = 0
$$

##### Definition: Almost Sure Convergence

A sequence of random variables $$(X_{n})_{n \in \mathbb{N}^{*}}$$ converges almost surely if:
$$
\mathbb{P}( \underset{n \to \infty}{\lim} X_{n}= X) = 1
$$