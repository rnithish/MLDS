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
import hypertools as hyp
from sklearn.mixture import GaussianMixture


seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
sorted(seeds, key = lambda x: x[1])

seed_dict = {}
for seed in seeds:
    seed_dict[seed[0]-1] = seed[1]-1
labeled_indices = [seed[0]-1 for seed in seeds]
labeled_set = set(labeled_indices)
indices = set(np.arange(0,10000))
unlabeled_set = indices - labeled_set

data = np.genfromtxt('Extracted_features.csv', delimiter=',')
#pca = PCA(n_components=102)
#data = pca.fit_transform(data)
train_x = []
labels = []
[train_x.append(data[seed[0]-1]) for seed in seeds]
[labels.append(seed[1]) for seed in seeds]

#plot_data = hyp.load(train_x)
#hyp.plot(train_x, '.', ndims=2, labels=labels)

# neigh = KNeighborsClassifier(n_neighbors=1)
# neigh.fit(train_x, labels)

# gmm = GaussianMixture(n_components=10)
# gmm.fit(data)
# predicted_labels = gmm.predict(data[6000:])
#
# mappings = {}
# for i in range(4000):
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
# true_label_mappingss = [1,0,2,5,4,6,3,7,8,9]
# for i in range(10):
#     true_label_mappings.append(np.argmax(true_labels[i]))
# for i in range(len(predicted_labels)):
#     predicted_labels[i] = true_label_mappingss[predicted_labels[i]]

ids = [i for i in range(6001,10001)]
# np.savetxt('output3.csv',np.column_stack((ids,predicted_labels)),delimiter=',', header="Id,Label", fmt='%s', comments='')

# kmeans = KMeans(n_clusters=10)
# kmeans.cluster_centers_ = np.array([data[seeds[6*i][0]-1] for i in range(10)])
#test_x = [data[i] for i in range(6000,10000)]
# test_x = data[6000:]
# preds = neigh.predict(test_x)
#preds = kmeans.predict(test_x)
# preds = [int(pred) for pred in preds]

# ids = [i for i in range(6001,10001)]
# np.savetxt('output.csv',np.column_stack((ids,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')

# y_train = np.arange(0,10000)
# y_train[list(unlabeled_set)] = -1
# for seed in seeds:
#     y_train[seed[0]-1] = seed[1]

#lp_model = label_propagation.LabelSpreading(gamma=0.25)
# lp_model = LabelSpreading()
# lp_model.fit(data, y_train)
# predicted_labels = lp_model.predict(data)
#predicted_labels = lp_model.transduction_[list(unlabeled_set)]
#preds = predicted_labels[5940:9941]
#preds = predicted_labels[6000:]

simscores = np.genfromtxt('Graph.csv', delimiter=',', dtype=(int, int))
#similarity_matrix = [[0 for _ in range(10)] for _ in range(10000)]
X_matrix = [[0 for _ in range(6000)] for _ in range(6000)]
for sim in simscores:
    id1 = int(sim[0]-1)
    id2 = int(sim[1]-1)
    X_matrix[id1][id2] = 1
    X_matrix[id2][id1] = 1
    # if(id1 in labeled_set):
    #     #y_train[id2] = y_train[id1]
    #     similarity_matrix[id2][seed_dict[id1]] += 1
    # elif(id2 in labeled_set):
    #     #y_train[id1] = y_train[id2]
    #     similarity_matrix[id1][seed_dict[id2]] += 1

# for id in range(10000):
#     y_train[id] = np.argmax(similarity_matrix[id])+1 if(np.max(similarity_matrix[id])>0) else -1
#
# posindices = y_train > 0
# y_train = y_train[posindices]
# train_data = data[posindices]
# svm = SVC()
# svm.fit(train_data, y_train)
# preds = svm.predict(data[6000:])

# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(train_data, y_train)
# preds = neigh.predict(data[6000:])

#sc_model = SpectralClustering(n_clusters=10, kernel_params={"metric":"precomputed"}, affinity=similarity_matrix)
#sc_model = SpectralClustering(n_clusters=10, affinity="precomputed", gamma=1)
#predicted_labels = sc_model.fit_predict(X_matrix)
spec_embed = SpectralEmbedding(affinity="precomputed")
embeddings = spec_embed.fit_transform(X_matrix)
cca = CCA()
result = cca.fit_transform(data[:6000],embeddings)
to_predict = cca.transform(data[6000:])
kmeans = KMeans(n_clusters=10)
predicted_labels = kmeans.fit_predict(result[0])
mappings = {}
for i in range(6000):
    if(predicted_labels[i] in mappings):
        mappings[predicted_labels[i]].add(i)
    else:
        mappings[predicted_labels[i]] = set([i])

true_labels = [[0 for i in range(10)] for _ in range(10)]
for seed in seeds:
    for i in range(10):
        if(seed[0]-1 in mappings[i]):
            true_labels[i][seed[1]-1] += 1
true_label_mappings = []
for i in range(10):
    true_label_mappings.append(np.argmax(true_labels[i]))
for i in range(len(predicted_labels)):
    predicted_labels[i] = true_label_mappings[predicted_labels[i]]

print(len(predicted_labels))

train_x = []
[train_x.append(result[0][seed[0]-1]) for seed in seeds]
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_x, labels)
preds = neigh.predict(to_predict)



# svm = SVC()
# svm.fit(data[:6000], predicted_labels)
# preds = svm.predict(data[6000:])
# neigh = KNeighborsClassifier(n_neighbors=7)
# neigh.fit(data[:6000], predicted_labels)
# preds = neigh.predict(data[6000:])

np.savetxt('output2.csv',np.column_stack((ids,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
