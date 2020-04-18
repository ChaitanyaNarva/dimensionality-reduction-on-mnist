# loading libraries

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Picking the top 1000 points as TSNE takes a lot of time for 15K points
data_1000 = standard_data[0:1000,:]   # data we generated in preprocessing step
labels_1000 = label[0:1000]

# Fitting T-SNE model to the mnist data with perplexity =30

tsne = TSNE(n_components=2, random_state=0) #perplexity=30 max_iter=1000 learning_rate=200
tsne_data = tsne.fit_transform(data_1000).T
print(tsne_data.shape)
print(len(labels_1000))

new_data = np.vstack((tsne_data,labels_1000)).T
tsne_df = pd.DataFrame(data=new_data, columns=("1st", "2nd","label"))
tsne_df['label']=labels_1000
# Ploting the result of tsne
sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, '1st', '2nd').add_legend()
plt.show()

# Fitting T-SNE model to the mnist data with perplexity = 100 and n_iter = 1000

tsne=TSNE(n_components=2, random_state=0, perplexity=100, n_iter=1000, learning_rate=1000)
tsne_data=tsne.fit_transform(data_1000)

tsne_data=np.vstack((tsne_data,label))
df=pd.DataFrame(data=tsne_data,columns=["1st","2nd","label"])

#plotting dataframe
sns.FacetGrid(df,hue="species",size=6).map(plt.scatter,"1st","2nd").add_legend()
plt.show()
