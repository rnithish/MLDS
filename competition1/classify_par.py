import numpy as np
from sklearn.cluster import *
from sklearn.neighbors import *

#create seeds and fix for one
seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
seedmap = {}
for seed in seeds:
    seed[0] -= 1
    seedmap[seed[0]] = seed[1]

data = np.genfromtxt('Extracted_features.csv', delimiter=',')

x_train = []
y_train = []
cents = {}

#create training x 60x1084 and training y 60x1
for seed in seeds:
    x_train.append(data[seed[0]])
    y_train.append(seed[1])

for idx, y in enumerate(y_train):
    if y in cents:
        cents[y] += x_train[idx]
    else:
        cents[y] = x_train[idx]

x_train = []
for i,_ in sorted(cents.items()):
    for j,_ in enumerate(cents[i]):
        cents[i][j] /= 6
    x_train.append(cents[i])

# ids = [i for i in range(0,10)]
# knn = KNeighborsClassifier(n_neighbors=1)
# knn.fit(x_train, ids)
# preds = knn.predict(data[6000:])
x_train = np.array(x_train)
kmeans = KMeans(n_clusters=10, init=x_train).fit(data[:6000])
preds = kmeans.predict(data[6000:])
# kmeans = KMeans(n_clusters=10, random_state=3)
# ydata = kmeans.fit_predict(data)
#
# ydata = ydata[6000:]
# ids = []
# trueLab = []
# fakeLab = []
#
# for idx, y in enumerate(ydata):
#     if idx in seedmap:
#         ids.append(idx)
#         trueLab.append(seedmap[idx])
#         fakeLab.append(y)

# for y in ydata:
#     y = clustmap[y]

ids = [i for i in range(6001,10001)]
# save to csv
np.savetxt('outkmeans2.1.csv',np.column_stack((ids,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')
