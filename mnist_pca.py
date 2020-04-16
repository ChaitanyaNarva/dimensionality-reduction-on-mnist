#loading libraries
import numpy as np
from scipy.linalg import eigh
import seaborn as sns
from sklearn import decomposition

# ********************** METHOD-1 *************************

#step-1: finding covariance matrix

covariance_matrix=np.matmul(data.T,data)  #data we generated in preprocessing step
print(covariance_matrix.shape)

#step-2: finding eigen values and eigen vectors

values,vectors=eigh(covariance_matrix,eigvals=[782,783])
print(vectors.shape)
#calculating variance:
preserved_variance=values[1]/(values[0]+values[1])
print("The amount of variance we preserved after dimensionality reduction to 2 :",int(preserved_variance*100))

#step-3: dimensionality reduction

new_data=np.matmul(scalars,vectors)
new_data=new_data.T
print(new_data.shape)

#step-4: data visualization

#creating dataframe of new_data
new_data=np.vstack((new_data,label))
new_data=new_data.T
print(new_data.shape)
dataframe=pd.DataFrame(data=new_data,columns=["1st","2nd","label"])
dataframe.head(5)
sns.FacetGrid(dataframe,hue="label",height=6).map(plt.scatter,"1st","2nd").add_legend()  #plotting the data
plt.show()

# ************************* METHOD-2 ***************************

#step-1: dimensionality reduction

pca=decomposition.PCA()
pca.n_components=2
pca_data=pca.fit_transform(data).T
print(pca_data.shape)

#step-2: data visualization

#creating dataframe of pca_data
pca_new=np.vstack((pca_data,label)).T
print(pca_new.shape)
df=pd.DataFrame(data=pca_new,columns=["1st","2nd","labels"])
df.head(5)
sns.FacetGrid(df,hue="labels",height=6).map(plt.scatter,"1st","2nd").add_legend()
plt.show()

#plotting the percentage of variance explained by each dimension:
pca.n_components=784
pca_data=pca.fit_transform(scaler.T)
var_per=pca.explained_variance_/np.sum(pca.explained_variance_)
cum_sum=np.cumsum(var_per)
plt.figure(1,figsize=(6,4))
plt.clf()
plt.plot(cum_sum,linewidth=2)
plt.grid()
plt.show()
