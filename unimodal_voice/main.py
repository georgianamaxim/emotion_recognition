import numpy as np
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
np.random.seed(1671)
NB_EPOCH = 25
VERBOSE = 1
NB_CLASSES = 6
N_HIDDEN = 1000
N_LINES = 400
RESHAPD = N_LINES * 12
N_INPUTFILES = 200
N_TESTFILES = 30

(X,Y) = load_data()

Max = np.amax(X)
Min = np.amin(X)
X = (X-Min) / (Max - Min)
Y = np_utils.to_categorical(Y, NB_CLASSES)

X = X.reshape(N_INPUTFILES, N_LINES, 12, 1)

BATCH_SIZE = 10
model = Sequential()
model.add(Conv2D(20, kernel_size=5, padding="same", input_shape= (N_LINES, 12, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Flatten())
model.add(Dense(N_HIDDEN))
model.add(Activation("relu"))

model.add(Dense(NB_CLASSES))
model.add(Activation("softmax"))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE)

(X_test, Y_test) = load_testdata()
Max = np.amax(X_test)
Min = np.amin(Y_test)
X_test = (X_test-Min) / (Max - Min)

Y_test = np_utils.to_categorical(Y, NB_CLASSES)
X_test = X_test.reshape(N_INPUTFILES, N_LINES, 12, 1)

scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
