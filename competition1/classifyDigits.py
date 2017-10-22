import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
sorted(seeds, key = lambda x: x[1])


data = np.genfromtxt('Extracted_features.csv', delimiter=',')
#pca = PCA(n_components=102)
#data = pca.fit_transform(data)
train_x = []
labels = []
[train_x.append(data[seed[0]]) for seed in seeds]
[labels.append(seed[1]) for seed in seeds]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, labels)

kmeans = KMeans(n_clusters=10)
kmeans.cluster_centers_ = np.array([data[seeds[6*i][0]] for i in range(10)])

#test_x = [data[i] for i in range(6000,10000)]
test_x = data[6000:10000]


#preds = neigh.predict(test_x)
preds = kmeans.predict(test_x)

preds = [int(pred) for pred in preds]

print(data[0])
print(preds[0])
id = [i for i in range(6001,10001)]

#np.savetxt("svmpred.csv", np.column_stack((ids, ytest)), delimiter=",", fmt='%s', header='Id,Prediction', comments='')

np.savetxt('output.csv',np.column_stack((id,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
