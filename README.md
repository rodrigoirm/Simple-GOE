# Simple-GOE
A solution for one of Sethna's "Statistical Mechanics: Entropy, Order Parameters, and Complexity" problems.

Briefly, it's a program that creates a Gaussian Orthogonal Ensemble whose matrices' entries are independent gaussian random variables, and plots not only the probability density function of the histogram of gaps (i.e., the differences of central eigenvalues divided by the average over all differences), but also the the approximate probability distribution of those gaps (Wigner surmise). Here is an example of an ensemble made of 15000  4x4 matrices:

![](images/probdistN4.png)

Constructive criticism for the code is welcome, especially because I am not used to programming.
