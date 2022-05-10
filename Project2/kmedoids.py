import tensorflow as tf
import numpy as np
import cv2
from math import log2
from sklearn.metrics import f1_score
from sklearn import metrics

# install an extension to help with the k-medoids model
!pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids

# import dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# train/test images/labels are used in training and testing respectively
# they are in numpy array form, to see their dimension do train_images.shape e.c. 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# custom function to calculate Kullback-Libler distance
def kl_divergence(p, q):
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

# for each image calculate the histogram and keep it in an np array
# we used 16 bins
new_images = []

for i in train_images:
  current_image = train_images[i]
  histogram = cv2.calcHist([current_image],[0],None,[32],[0,256])
  new_images.append(histogram)

final_train_images = np.array(new_images)

samples, x, y = final_train_images.shape

# final training images to use
x_train = final_train_images.reshape((samples,x*y))

# keep a smaller part because it runs out of memory
x_train = x_train[0:1000]
train_labels = train_labels[0:1000]

# create a k-medoids model with 10 clusters, the same as data labels
# and using the kl distance
model = KMedoids(n_clusters = 10, metric = kl_divergence)
# train the k-medoids model
model.fit(x_train)
# predict clusters
prediction = model.predict(x_train)

# calculate the f_measure based on predictions and the true data labels
print("F1-Measure = ",f1_score(train_labels,prediction,average = 'micro'))

# custom function to calculate cluster purity
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
# calculate cluster purity based on predictions and the true data labels
print("Purity = ",purity_score(train_labels,prediction))