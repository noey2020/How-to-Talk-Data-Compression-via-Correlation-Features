# How-to-Talk-Data-Compression-via-Correlation-Features

February 7, 2021

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

In the previous "How to Talk" installment, Building Good Training Datasets - Data Preprocessing,
we learned about the different approaches for reducing the dimensionality of a dataset
using different feature selection techniques. An alternative approach to feature
selection for dimensionality reduction is feature extraction. In this segment, you
will learn about three fundamental techniques that will help you to summarize the
information content of a dataset by transforming it onto a new feature subspace of
lower dimensionality than the original one. Data compression is an important topic
in machine learning, and it helps us to store and analyze the increasing amounts
of data that are produced and collected in the modern age of technology.

In this installment, we will cover the following topics:

- Principal component analysis (PCA) for unsupervised data compression

- Linear discriminant analysis (LDA) as a supervised dimensionality
  reduction technique for maximizing class separability
  
- Nonlinear dimensionality reduction via kernel principal component
  analysis (KPCA)
  
Similar to feature selection, we can use different feature extraction techniques to
reduce the number of features in a dataset. The difference between feature selection
and feature extraction is that while we maintain the original features when we use
feature selection algorithms, such as sequential backward selection, we use feature
extraction to transform or project the data onto a new feature space.

In the context of dimensionality reduction, feature extraction can be understood as
an approach to data compression with the goal of maintaining most of the relevant
information. In practice, feature extraction is not only used to improve storage space
or the computational efficiency of the learning algorithm, but can also improve the
predictive performance by reducing the curse of dimensionality, especially if we
are working with non-regularized models.

PCA is widely used across different fields, most prominently for feature
extraction and dimensionality reduction. Other popular applications of PCA include
exploratory data analyses and the denoising of signals in stock market trading, and
the analysis of genome data and gene expression levels in the field of bioinformatics.

PCA helps us to identify patterns in data based on the correlation between features.
In a nutshell, PCA aims to find the directions of maximum variance in highdimensional
data and projects the data onto a new subspace with equal or fewer
dimensions than the original one. The orthogonal axes (principal components) of
the new subspace can be interpreted as the directions of maximum variance given
the constraint that the new feature axes are orthogonal to each other.

Before looking at the PCA algorithm for dimensionality reduction in more detail,
let's summarize the approach in a few simple steps:

1. Standardize the d-dimensional dataset.

2. Construct the covariance matrix.

3. Decompose the covariance matrix into its eigenvectors and eigenvalues.

4. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.

5. Select k eigenvectors, which correspond to the k largest eigenvalues, where k is
   the dimensionality of the new feature subspace (k less than or equal to d).
   
6. Construct a projection matrix, W, from the "top" k eigenvectors.

7. Transform the d-dimensional input dataset, X, using the projection matrix, W,
   to obtain the new k-dimensional feature subspace.
   
import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)
                     
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)

X_test_std = sc.transform(X_test)

import numpy as np

cov_mat = np.cov(X_train_std.T)

eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print('\nEigenvalues \n%s' % eigen_vals)

# Make a list of (eigenvalue, eigenvector) tuples

eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low

eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))
               
print('Matrix W:\n', w)

Tune in to the next "How to Talk ...".

I included some jupyter notebooks to serve as study guide and to practice on real python code.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Building-Good-Training-Datasets

https://github.com/noey2020/How-to-Talk-scikit-learn-Machine-Learning-Library

https://github.com/noey2020/How-to-Talk-Linear-Regression-Optimizing-Loss-Function-Mean-Squared-Error

https://github.com/noey2020/How-to-Talk-an-Introduction-to-Linear-Regression

https://github.com/noey2020/Hpw-to-Talk-More-Generative-Models

https://github.com/noey2020/How-to-Talk-Gaussian-Generative-Models

https://github.com/noey2020/How-to-Talk-Multivariate-Gaussian

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-3

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-2

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-1

https://github.com/noey2020/How-to-Talk-2D-Generative-Modeling

https://github.com/noey2020/How-to-Talk-Probability-Review-3

https://github.com/noey2020/How-to-Talk-Probability-Review-2

https://github.com/noey2020/How-to-Talk-Generative-Modeling-in-One-Dimension

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
