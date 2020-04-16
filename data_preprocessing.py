#loading libraries

import pandas as pd
import numpy as np
import os
os.chdir("C:\\Users\\Chaitanya Narva\\Documents")   #path setting
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#loading data

data = pd.read_csv("train.csv")
label=mnist["label"]
data=mnist.drop("label",axis=1)
data.shape

#printing a random mnist datapoint

index
print("label of the image is",label[index])
plt.figure(figsize=(3,3))
matrix=np.matrix(data.iloc[index]).reshape(28,28)   #reshaping 784 dimensions to 24*24 as an image format
plt.imshow(matrix,cmap='gray')
plt.axis('off')
plt.show()

#data standarlization

scaler = StandardScaler().fit(data)
data=scaler.transform(data)
data.shape
