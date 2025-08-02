#import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential

X = np.linspace(1,10,100)
Y = 2*X+10
print(X)
print(Y)

X = np.linspace(1,100,100)
Y = 2*X+10+np.random.randn(100)
print(X)
print(Y)

#define architectue of the model
model = Sequential()
model.add(Dense(1,input_dim=1,activation='linear'))

#compile the model
model.compile(optimizer='sgd',loss='mse')
#Train the model
model.fit(X,Y,epochs=100)
#make predictions
pred = model.predict(X)

plt.scatter(X,Y,label='original data')
plt.plot(X,pred,label='predicted data')
plt.show()

