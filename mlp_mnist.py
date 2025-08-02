#import libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils import to_categorical


#load data
(X_train, y_train),(X_test, y_test) = mnist.load_data()
print(X_train.shape)
#normalize the data

# normalize the images
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# one-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


#build the architecture

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

#compile the model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

#train the model
model.fit(X_train,y_train,epochs=10,batch_size=64)

#evaluate
model.evaluate(X_test,y_test)


#predictions
 
sample_images = X_test[:5]
sample_labels=y_test[:5]

predictions = model.predict(sample_images)

print(predictions)

result = np.argmax(predictions, axis = 1)

print(result)

for i in range(5):
    actual = np.argmax(sample_labels[i])  # Actual digit label
    predicted = result[i]                 # Predicted digit
    plt.subplot(1, 5, i + 1)
    plt.title(f'Actual: {actual}\nPredicted: {predicted}')
    plt.imshow(sample_images[i], cmap='gray')
    plt.axis('off')
plt.show()

