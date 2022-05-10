import tensorflow as tf
import numpy as np
import cv2
from sklearn.metrics import f1_score
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

# import dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

# train/test images/labels are used in training and testing respectively
# they are in numpy array form, to see their dimension do train_images.shape e.c. 
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# for each image get histogram and keep it in an np array
# we used 16 bins
new_images = []

for i in train_images:
  current_image = train_images[i]
  histogram = cv2.calcHist([current_image],[0],None,[16],[0,256])
  new_images.append(histogram)

final_train_images = np.array(new_images)

samples, x, y = final_train_images.shape

# final training images to use
x_train = final_train_images.reshape((samples,x*y))

# keep a smaller part because it runs out of memory
x_train = x_train[0:10000]
train_labels = train_labels[0:10000]

# create a hierarchical clustering model, aka agglomerative
# run 3 times with affinity = euclidean/manhattan/cosine
model = AgglomerativeClustering(n_clusters = 10, affinity = 'euclidean', linkage = 'complete')
# train the model and get cluster predictions
prediction = model.fit_predict(x_train)

# calculate the f_measure based on predictions and the true data labels
print("F1-Measure = ",f1_score(train_labels,prediction,average = 'micro'))

# custom function to calculate cluster purity
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
# calculate cluster purity based on predictions and the true data labels
print("Purity = ",purity_score(train_labels,prediction))