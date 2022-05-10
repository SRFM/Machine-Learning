import tensorflow as tf
import numpy as np
from sklearn import metrics
from sklearn import svm

# import dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# train/test images/labels are used in training and testing respectively
# they are in numpy array form, to see their dimension do train_images.shape e.c. 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# there are 10 labels, with these names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# process the values of dataset
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape training/testing images from 3d to 2d arrays
nsamples, nx, ny = train_images.shape
d2_train_dataset = train_images.reshape((nsamples,nx*ny))
samples, x, y = test_images.shape
d2_test_dataset = test_images.reshape((samples,x*y))

# keep only a small part of the whole dataset, because it takes too much time
x_train = d2_train_dataset[0:10000]
y_train = train_labels[0:10000] 
x_test = d2_test_dataset[0:10000]
y_test = test_labels[0:10000]

# Support Vector Machine model with rbf (Gaussian) kernel and one-vs-rest stategic (ovr)
# it is the same with the one-vs-all
model = svm.SVC(kernel = 'rbf', decision_function_shape = 'ovr')

# feed data to train the model
model.fit(x_train,y_train)

# test the trained model and get the predictions
prediction = model.predict(x_test)

# print metrics needed
accuracy = metrics.accuracy_score(y_test, prediction)
precision = metrics.precision_score(y_test, prediction, average = 'macro')
recall = metrics.recall_score(y_test, prediction, average = 'macro')
f1 = 2*(precision*recall)/(precision + recall)

print('\nAccuracy :', accuracy)
print('\nF1 score :', f1)