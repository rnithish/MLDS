import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cross_decomposition import CCA
from sklearn.manifold import SpectralEmbedding
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture


seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
sorted(seeds, key = lambda x: x[1])

data = np.genfromtxt('Extracted_features.csv', delimiter=',')
# pca = PCA(n_components=1024)
# data = pca.fit_transform(data)

simscores = np.genfromtxt('Graph.csv', delimiter=',', dtype=(int, int))
X_matrix = [[0 for _ in range(6000)] for _ in range(6000)]
for sim in simscores:
    id1 = int(sim[0]-1)
    id2 = int(sim[1]-1)
    X_matrix[id1][id2] = 1
    X_matrix[id2][id1] = 1

spec_embed = SpectralEmbedding(affinity="precomputed", n_components=1084)
embeddings = spec_embed.fit_transform(X_matrix)
cca = CCA()
result = cca.fit_transform(data[:6000],embeddings)
to_predict = cca.transform(data[6000:])

cents = {}
for y in seeds:
    if y[1] in cents:
        cents[y[1]] += result[0][y[0]-1]
    else:
        cents[y[1]] = result[0][y[0]-1]

x_train = []
# ids = [i for i in range(0,10)]
for i, _ in sorted(cents.items()):
    for j, _ in enumerate(cents[i]):
        cents[i][j] /= 6
    x_train.append(cents[i])

kmeans = KMeans(n_clusters=10, init = np.array(x_train))
preds = kmeans.fit_predict(to_predict)
####MAPPING BULLSHIT
# mappings = {}
# for i in range(6000):
#     if(predicted_labels[i] in mappings):
#         mappings[predicted_labels[i]].add(i)
#     else:
#         mappings[predicted_labels[i]] = set([i])
#
# true_labels = [[0 for i in range(10)] for _ in range(10)]
# for seed in seeds:
#     for i in range(10):
#         if(seed[0]-1 in mappings[i]):
#             true_labels[i][seed[1]-1] += 1
# true_label_mappings = []
# for i in range(10):
#     true_label_mappings.append(np.argmax(true_labels[i]))
# for i in range(len(predicted_labels)):
#     predicted_labels[i] = true_label_mappings[predicted_labels[i]]
######MAPPING BULLSHIT END

# train_x = []
# [train_x.append(result[0][seed[0]-1]) for seed in seeds]
# labels = []
# [labels.append(seed[1]) for seed in seeds]
#
# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(train_x, labels)
# preds = neigh.predict(to_predict)


ids = [i for i in range(6001,10001)]
np.savetxt('output2.5.csv',np.column_stack((ids,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
