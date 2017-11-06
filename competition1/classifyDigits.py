import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC


seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
sorted(seeds, key = lambda x: x[1])

labeled_indices = [seed[0]-1 for seed in seeds]
labeled_set = set(labeled_indices)
indices = set(np.arange(0,10000))
unlabeled_set = indices - labeled_set

data = np.genfromtxt('Extracted_features.csv', delimiter=',')
#pca = PCA(n_components=102)
#data = pca.fit_transform(data)
# train_x = []
# labels = []
# [train_x.append(data[seed[0]]) for seed in seeds]
# [labels.append(seed[1]) for seed in seeds]
#
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(train_x, labels)
#
# kmeans = KMeans(n_clusters=10)
# kmeans.cluster_centers_ = np.array([data[seeds[6*i][0]] for i in range(10)])
# #test_x = [data[i] for i in range(6000,10000)]
# test_x = data[6000:10000]
# #preds = neigh.predict(test_x)
# preds = kmeans.predict(test_x)
# preds = [int(pred) for pred in preds]
#
id = [i for i in range(6001,10001)]

y_train = np.arange(0,10000)
y_train[list(unlabeled_set)] = -1
for seed in seeds:
    y_train[seed[0]-1] = seed[1]

# #lp_model = label_propagation.LabelSpreading(gamma=0.25)
# lp_model = LabelSpreading()
# lp_model.fit(data, y_train)
# predicted_labels = lp_model.predict(data)
# #predicted_labels = lp_model.transduction_[list(unlabeled_set)]
# #preds = predicted_labels[5940:9941]
# preds = predicted_labels[6000:]

simscores = np.genfromtxt('Graph.csv', delimiter=',', dtype=(int, int))
similarity_matrix = [[1 for _ in range(10000)] for _ in range(10000)]
for sim in simscores:
    id1 = int(sim[0]-1)
    id2 = int(sim[1]-1)
    similarity_matrix[id1][id2] = 0.5
    # if(id1 in labeled_set):
    #     y_train[id2] = y_train[id1]
    # elif(id2 in labeled_set):
    #     y_train[id1] = y_train[id2]

# posindices = y_train > -1
# y_train = y_train[posindices]
# train_data = data[posindices]
# svm = SVC()
# svm.fit(train_data, posindices)
# preds = svm.predict(data[6000:])

lp_model = LabelPropagation(max_iter=1000, gamma=0.02)
lp_model.fit(data, y_train)
preds = lp_model.predict(data)
preds = preds[6000:]

np.savetxt('output3.csv',np.column_stack((id,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
