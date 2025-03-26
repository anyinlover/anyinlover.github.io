---
title: Principles of Finite Precison Computation
pubDate: 2025-03-25 11:14:00
tags:
  - float
  - precision
---

# Principles of Finite Precison Computation

## Notation and Background

Evaluation of an expression in floating point arithmetic is denoted $fl(\cdot)$, and we assume that the basic arithmetic operations $op = +, -, *, /$ satisfy

$$fl(x \ op \ y) = (x \ op \ y)(1 + \delta), \quad |\delta| \le u.$$

Here, $u$ is the *unit roundoff* (or machine precision), which is typically of order $10^{-8}$ or $10^{-16}$ in single and double precision computer arithmetic.

## Relative Error

Let $\widehat{x}$ be an approximation to a real number $x$. The most useful measures of the accuracy of $\widehat{x}$ are its *absolute error*

$$E_{abs}(\widehat{x}) = |x - \widehat{x}|,$$

and its *relative error*

$$E_{rel}(\widehat{x}) = \frac{|x - \widehat{x}|}{|x|}$$

In scientific computation, where answers to problems can vary enormously in magnitude, it is usually the relative error that is of interest, because it is scale independent.

When $x$ and $\widehat{x}$ are vectors the relative error is most often defined with a norm, as $\|x - \widehat{x}\| / \|x\|$. For the commonly used norms $\|x\|_\infty := \max_i |x_i|$, $\|x\|_1 := \sum_i |x_i|$, and $\|x\|_2 := (x^Tx)^{1/2}$.

A relative error that puts the individual relative errors on an equal footing is the *componentwise relative error*

$$\max_i \frac{|x_i - \widehat{x}_i|}{|x_i|},$$

which is widely used in error analysis and perturbation analysis.

## Sources of Errors

There are three main sources of errors in numerical computation: rounding, data uncertainty, and truncation.

Rounding errors, which are an unavoidable consequence of working in finite precision arithmetic.

Uncertainty in the data is always a possibility when we are solving practical problems. It may arise in several ways:

- from errors of measurement or estimation,

- from errors in storing the data on the computer (rounding errors—tiny),

- from the result of errors (big or small) in an earlier computation if the data is itself the solution to another problem.

The effects of errors in the data are generally easier to understand than the effects of rounding errors committed during a computation, because data errors can be analysed using perturbation theory for the problem at hand, while intermediate rounding errors require an analysis specific to the given method.

Analysing truncation errors, or discretization errors, is one of the major tasks of the numerical analyst. Many standard numerical methods (for example, the trapezium rule for quadrature, Euler's method for differential equations, and Newton's method for nonlinear equations) can be derived by taking finitely many terms of a Taylor series. The terms omitted constitute the truncation error, and for many methods the size of this error depends on a parameter (often called $h$, "the step-size") whose appropriate value is a compromise between obtaining a small error and a fast computation.

## Precision Versus Accuracy

**Accuracy** refers to the absolute or relative error of an approximate quantity. **Precision** is the accuracy with which the basic arithmetic operations +, -, *, / are performed, and for floating point arithmetic is measured by the unit roundoff $u$. Accuracy and precision are the same for the scalar computation $c = a * b$, but accuracy can be much worse than precision in the solution of a linear system of equations, for example.

It is important to realize that accuracy is not limited by precision, at least in theory. Arithmetic of a given precision can be used to simulate arithmetic of arbitrarily high precision.

## Backward and Forward Errors

Instead of focusing on the relative error of $\widehat{y}$ we can ask, "For what set of data have we actually solved our problem?", that is, for what $\Delta x$ do we have $\widehat{y} = f(x + \Delta x)$? In general, there may be many such $\Delta x$, so we should ask for the smallest one. The value of $|\Delta x|$ (or $\min |\Delta x|$), possibly divided by $|x|$, is called the *backward error*. The absolute and relative errors of $\widehat{y}$ are called *forward errors*, to distinguish them from the backward error.

The process of bounding the backward error of a computed solution is called *backward error analysis*, and its motivation is twofold. 

1. It interprets rounding errors as being equivalent to perturbations in the data.
2. It reduces the question of bounding or estimating the forward error to perturbation theory, which for many problems is well understood.

A method for computing $y = f(x)$ is called *backward stable* if, for any $x$, it produces a computed $\widehat{y}$ with a small backward error, that is, $\widehat{y} = f(x + \Delta x)$ for some small $\Delta x$. The definition of "small" will be context dependent.

Most routines for computing the cosine function do not satisfy $\widehat{y} = \cos(x + \Delta x)$ with a relatively small $\Delta x$, but only the weaker relation $\widehat{y} + \Delta y = \cos(x + \Delta x)$, with relatively small $\Delta y$ and $\Delta x$. A result of the form

$$\widehat{y} + \Delta y = f(x + \Delta x), \quad |\Delta y| \le \epsilon |y|, \quad |\Delta x| \le \eta |x| $$

is known as a *mixed forward-backward error result*. In general, an algorithm is called *numerically stable* if it is stable in the mixed forward-backward error.

## Conditioning

The relationship between forward and backward error for a problem is governed by the *conditioning* of the problem, that is, the sensitivity of the solution to perturbations in the data. Continuing the $y = f(x)$ example of the previous section, let an approximate solution $\widehat{y}$ satisfy $\widehat{y} = f(x + \Delta x)$. Then, assuming for simplicity that $f$ is twice continuously differentiable,

$$\widehat{y} - y = f(x + \Delta x) - f(x) = f'(x)\Delta x + \frac{f''(x + \theta \Delta x)}{2!}(\Delta x)^2, \quad \theta \in (0,1),$$

and we can bound or estimate the right-hand side. This expansion leads to the notion of condition number. Since

$$\frac{\widehat{y} - y}{y} = \left(\frac{xf'(x)}{f(x)}\right) \frac{\Delta x}{x} + O((\Delta x)^2),$$

the quantity

$$c(x) = \left| \frac{xf'(x)}{f(x)} \right|$$

measures, for small $\Delta x$, the relative change in the output for a given relative change in the input, and it is called the *(relative) condition number* of $f$. If $x$ or $f$ is a vector then the condition number is defined in a similar way using norms, and it measures the *maximum* relative change, which is attained for some, but not all, vectors $\Delta x$.

If a method produces answers with forward errors of similar magnitude to those produced by a backward stable method, then it is called forward stable. Such a method need not be backward stable itself. Backward stability implies forward stability, but not vice versa.

## Cancellation

Cancellation is what happens when two nearly equal numbers are subtracted. Consider the subtraction (in exact arithmetic) $\widehat{x} = \widehat{a} - \widehat{b}$, where $\widehat{a} = a(1 + \Delta a)$ and $\widehat{b} = b(1 + \Delta b)$. The terms $\Delta a$ and $\Delta b$ are relative errors or uncertainties in the data, perhaps attributable to previous computations. With $x = a - b$ we have

$$\left| \frac{x - \widehat{x}}{x} \right| = \left| \frac{-a\Delta a + b\Delta b}{a - b} \right| \le \max(|\Delta a|, |\Delta b|) \frac{|a| + |b|}{|a - b|}.$$

The relative error bound for $\widehat{x}$ is large when $|a - b| \ll |a| + |b|$, that is, when there is heavy cancellation in the subtraction. This analysis shows that subtractive cancellation causes relative errors or uncertainties already present in $a$ and $b$ to be magnified. In other words, subtractive cancellation brings earlier errors into prominence.

It is important to realize that cancellation is not always a bad thing. 
1. The numbers being subtracted may be error free, as when they are from initial data that is known exactly. 
2. The second reason is that cancellation may be a symptom of intrinsic ill conditioning of a problem, and may therefore be unavoidable. 
3. The effect of cancellation depends on the role that the result plays in the remaining computation. For example, if $x \gg y \approx z > 0$ then the cancellation in the evaluation of $x + (y - z)$ is harmless.


## Accumulation of Rounding Errors

Most often, instability is caused not by the accumulation of millions of rounding errors, but by the insidious growth of just a few rounding errors.

As an example, let us approximate $e = \exp(1)$ by taking finite $n$ in the definition
$$e := \lim_{n \to \infty} \left( 1 + \frac{1}{n} \right)^n.$$
The approximations are poor, degrading as $n$ approaches the reciprocal of the machine precision. For $n$ a power of 10, $1/n$ has a nonterminating binary expansion. When $1 + 1/n$ is formed for $n$ a large power of 10, only a few significant digits from $1/n$ are retained in the sum.

## Instability Without Cancellation

Besides subtractive cancellation, there are other sources of instablity. One is overflow and underflow, another is sum a serial of numbers from largest to smallest.

## Increasing the Precision

When the only source of errors is rounding, a common technique for estimating the accuracy of an answer is to recompute it at a higher precision and to see how many digits of the original and the (presumably) more accurate answer agree. We would intuitively expect any desired accuracy to be achievable by computing at a high enough precision. This is certainly the case for algorithms possessing an error bound proportional to the precision. However, since an error bound is not necessarily attained, there is no guarantee that a result computed in *t*-digit precision will be more accurate than one computed in *s*-digit precision, for a given *t* > *s*; in particular, for a very ill conditioned problem both results could have no correct digits.**

## Cancellation of Rounding Errors

It is not unusual for rounding errors to cancel in stable algorithms, with the result that the final computed answer is much more accurate than the intermediate quantities.

## Rounding Errors Can Be Beneficial

In some algorithms rounding errors can even be beneficial.

## Stability of an Algorithm Depends on the Problem

An algorithm can be stable as a means for solving one problem but unstable when applied to another problem.

## Rounding Errors Are Not Random

Rounding errors, and their accumulated effect on a computation, are not random.

## Designing Stable Algorithms

There is no simple recipe for designing numerically stable algorithms. A few guidelines can be given.

1. Try to avoid subtracting quantities contaminated by error (though such subtractions may be unavoidable).
2. Minimize the size of intermediate quantities relative to the final solution.
3. Look for different formulations of a computation that are mathematically but not numerically equivalent.
4. It is advantageous to express update formulae as $new_value = old_value + small_correction$ if the small correction can be computed with many correct significant figures.
5. Use only well-conditioned transformations of the problem. In matrix computations this amounts to multiplying by orthogonal matrices instead of nonorthogonal, and possibly, ill-conditioned matrices, where possible.
6. Take precautions to avoid unnecessary overflow and underflow.
