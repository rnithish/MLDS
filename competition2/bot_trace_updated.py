import numpy as np
from numpy import genfromtxt
from hmmlearn.hmm import GaussianHMM

label_data = genfromtxt('Label', delimiter=',')
observation_data = genfromtxt('Observations.csv', delimiter=',')

num_blocks = 10

xmin = np.amin(label_data[:,2])
xmax = np.amax(label_data[:,2])

ymin = np.amin(label_data[:,3])
ymax = np.amax(label_data[:,3])

y_increments = []
y_stepsize = (ymax - ymin) / num_blocks
for i in range(num_blocks):
    if i == 0:
        y_increments.append(ymin)
    else:
        y_increments.append(y_increments[-1] + y_stepsize)

x_increments = []
x_stepsize = (xmax - xmin) / num_blocks
for i in range(num_blocks):
    if i == 0:
        x_increments.append(xmin)
    else:
        x_increments.append(x_increments[-1] + x_stepsize)


def getid_fromcoordinates(x, y):
    id = 0
    i = 0
    while i < len(y_increments) and y_increments[i] <= y:
        i+=1
    id += (i-1) * num_blocks

    i = 0
    while i < len(x_increments) and x_increments[i] <= x:
        i += 1
    id += (i - 1)

    return id

preds2 = []
runs = [[] for _ in range(6000)]
for i in range(len(label_data)):
    tuple = label_data[i]
    run = int(tuple[0])-1
    step = int(tuple[1])-1
    angle = observation_data[run][step]
    x = tuple[2]
    y = tuple[3]
    id = getid_fromcoordinates(x,y)
    if run == 1:
        preds2.append(id)
    runs[run].append([id,angle])

#runs_tuple = ([runs[0], runs[1]])
runs_tuple = ([runs[i] for i in range(len(runs))])
runs_fit = np.concatenate(runs_tuple)
#runs_lengths = [len(runs[0]), len(runs[1])]
runs_lengths = [len(runs[i]) for i in range(len(runs))]
model2 = GaussianHMM(n_components=100, covariance_type="diag", n_iter=2).fit(runs_fit, runs_lengths)

#observation_pred = [[x] for x in observation_data[7000]]
preds = model2.predict(runs[1])
print(preds)
print(preds2)

z = [preds[i] == preds2[i] for i in range(len(preds))]
print(z)
