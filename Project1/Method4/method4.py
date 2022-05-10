import tensorflow as tf
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

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
x_train = train_images.reshape((nsamples,nx*ny))
samples, x, y = test_images.shape
x_test = test_images.reshape((samples,x*y))

# gaussian naive bayes model
model = GaussianNB()

# feed data to train the model
model.fit(x_train,train_labels)

# test model and get predictions
predictions = model.predict(x_test)

# print metrics needed
accuracy = metrics.accuracy_score(test_labels, predictions)
precision = metrics.precision_score(test_labels, predictions, average = 'macro')
recall = metrics.recall_score(test_labels, predictions, average = 'macro')
f1 = 2*(precision*recall)/(precision + recall)

print('\nAccuracy :', accuracy)
print('\nF1 score :', f1)