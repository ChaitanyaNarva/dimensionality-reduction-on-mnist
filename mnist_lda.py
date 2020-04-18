# Loading libraries

from sklearn.datasets import load_iris
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Loading the IRIS datset

iris = load_iris()
iris.data.shape

#Step-1 : Data standarlization

scaler=StandardScaler().fit(iris.data)
data=scaler.transform(iris.data)

#Step-2 : Fitting the LDA model to the data

lda = LDA()
lda = LDA(n_components=2)
lda_result = lda.fit_transform(data, iris.target)

#Step-3 : Data Visualization

plt.scatter(lda_result[iris.target==0, 0], lda_result[iris.target==0, 1], color='r')
plt.scatter(lda_result[iris.target==1, 0], lda_result[iris.target==1, 1], color='g') 
plt.scatter(lda_result[iris.target==2, 0], lda_result[iris.target==2, 1], color='b') 
plt.title('LDA on iris')

# Principal Component Analysis Versus Linear Discriminant Analysis on IRIS dataset

pca = PCA(n_components=2)   #fitting PCA model to the data
pca_result = pca.fit_transform(iris.data)

# for PCA
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(pca_result[iris.target==0, 0], pca_result[iris.target==0, 1], color='r')
plt.scatter(pca_result[iris.target==1, 0], pca_result[iris.target==1, 1], color='g') 
plt.scatter(pca_result[iris.target==2, 0], pca_result[iris.target==2, 1], color='b') 
plt.title('PCA on iris')

# for LDA
plt.subplot(1,2,2)
plt.scatter(lda_result[iris.target==0, 0], lda_result[iris.target==0, 1], color='r')
plt.scatter(lda_result[iris.target==1, 0], lda_result[iris.target==1, 1], color='g') 
plt.scatter(lda_result[iris.target==2, 0], lda_result[iris.target==2, 1], color='b') 
plt.title('LDA on iris')
plt.show()
