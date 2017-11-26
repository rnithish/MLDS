import numpy as np
from numpy import genfromtxt
from hmmlearn.hmm import GaussianHMM

label_data = genfromtxt('Label', delimiter=',')
observation_data = genfromtxt('Observations.csv', delimiter=',')
runs = [[] for _ in range(6000)]
for i in range(len(label_data)):
    tuple = label_data[i]
    run = int(tuple[0])-1
    step = int(tuple[1])-1
    angle = observation_data[run][step]
    x = tuple[2]
    y = tuple[3]
    runs[run].append([x,y,angle])

model2 = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(runs[0])
preds = model2.predict(runs[1])

print(preds)


