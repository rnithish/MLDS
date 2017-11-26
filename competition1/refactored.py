import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cross_decomposition import CCA
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.semi_supervised import label_propagation
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
import hypertools as hyp
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib

colors=["red", "gold", "limegreen","blue","silver","green","violet","magenta","cyan","black"]
cmap = matplotlib.colors.ListedColormap(colors)


seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
sorted(seeds, key = lambda x: x[1])

data = np.genfromtxt('Extracted_features.csv', delimiter=',')

simscores = np.genfromtxt('Graph.csv', delimiter=',', dtype=(int, int))
X_matrix = [[0 for _ in range(6000)] for _ in range(6000)]
for sim in simscores:
    id1 = int(sim[0]-1)
    id2 = int(sim[1]-1)
    X_matrix[id1][id2] = 1
    X_matrix[id2][id1] = 1

spec_embed = SpectralEmbedding(affinity="precomputed")
embeddings = spec_embed.fit_transform(X_matrix)
cca = CCA(n_components=22)
result = cca.fit_transform(data[:6000],embeddings)
to_predict = cca.transform(data[6000:])
kmeans = KMeans(n_clusters=10)
predicted_labels = kmeans.fit_predict(result[0])

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

train_x = []
[train_x.append(result[0][seed[0]-1]) for seed in seeds]
labels = []
[labels.append(seed[1]) for seed in seeds]

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_x, labels)
preds = neigh.predict(to_predict)


tsnepts = TSNE().fit_transform(data[6000:])
plt.title("Spectral Clustering")
plt.scatter(tsnepts[:,0], tsnepts[:,1], c=preds, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()


# N = 10
# cmap = plt.cm.jet
# #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# scat = ax.scatter(tsnepts[:][0],tsnepts[:][1],c=preds,s=np.random.randint(100,500,10),cmap=cmap)
# # create the colorbar
# cb = plt.colorbar(scat, spacing='proportional')
# cb.set_label('Custom cbar')
# ax.set_title('Discrete color mappings')
# plt.show()
#
# colors=["red", "gold", "limegreen","blue","silver","green","violet","magenta","cyan","black"]
# cmap = matplotlib.colors.ListedColormap(colors)
# #cmap = matplotlib.colors.L
# plt.subplot(2, 2, 1)
# plt.scatter(tsnepts[:][0], tsnepts[:][1], c=preds,cmap=cmap, label='first 500', marker='.')
# plt.plot()
# plt.title('View 2 forming 2 distinct clusters through CCA, without PCA')
#
# plt.tight_layout()
# plt.show()

ids = [i for i in range(6001,10001)]
np.savetxt('output2.csv',np.column_stack((ids,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
