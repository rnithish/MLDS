import numpy as np
from keras.models import Sequential
from keras.layers import *

#label to onehot
def onehot(n ,lab):
    x = [0 for _ in range(n)]
    x[lab] = 1
    return x

#onehot to label
def classify(onehotv):
    i = 0
    maxi = 0
    for idx, val in enumerate(onehotv):
        if val>i:
            i = val
            maxi = idx
    return maxi

#create seeds and fix for one
seeds = np.genfromtxt('Seed.csv', delimiter=',', dtype=(int, int))
for seed in seeds:
    seed[0] -= 1

data = np.genfromtxt('Extracted_features.csv', delimiter=',')

x_train = []
y_train = []

#create training x 60x1084 and training y 60x1
for seed in seeds:
    x_train.append(data[seed[0]])
    y_train.append(onehot(10,seed[1]))

#convert to numpy array
x_train = np.array(x_train)
y_train = np.array(y_train)

#model architecture
model = Sequential()
# model.add(Dense(60, input_dim=1084, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))
model.add(Dense(4096, activation='relu', input_dim=1084))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.add(Convolution1D(32, 3, activation='relu', input_shape=(60, 1084)))
# model.add(Convolution1D(32, 3, activation='relu'))
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

#compile and fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=8)

#basic self scoring
scores = model.evaluate(x_train, y_train)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#predict and convert to apt form
predictions = model.predict(data[6000:])
preds = []
for i in predictions:
    preds.append(classify(i))

#save to csv
id = [i for i in range(6001,10001)]
np.savetxt('outnn.fix.csv',np.column_stack((id,preds)),delimiter=',', header="Id,Label", fmt='%s', comments='')