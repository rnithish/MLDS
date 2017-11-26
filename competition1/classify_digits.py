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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import hypertools as hyp
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib


seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
sorted(seeds, key = lambda x: x[1])

data = np.genfromtxt('Extracted_features.csv', delimiter=',')
seed_labels = [seed[1] for seed in seeds]
seed_pts = [data[seed[0]-1] for seed in seeds]
tsnepts = TSNE().fit_transform(seed_pts)
seed_cca1 = [tsnept[0] for tsnept in tsnepts]
seed_cca2 = [tsnept[1] for tsnept in tsnepts]

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

seed_cca1 = [result[0][seed[0]-1][0] for seed in seeds]
seed_cca2 = [result[0][seed[0]-1][1] for seed in seeds]
seed_labels = [seed[1] for seed in seeds]

kmeans = KMeans(n_clusters=10, init = np.array(x_train), random_state=1)
preds = kmeans.fit_predict(to_predict)


plt.title("centroids(seeds) after CCA")
plt.scatter(seed_cca1, seed_cca2, c=seed_labels, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()


plt.title("CCA Clustering")
plt.scatter(to_predict[:,0], to_predict[:,1], c=preds, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()


ids = [i for i in range(6001,10001)]
np.savetxt('output.csv',np.column_stack((ids,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
