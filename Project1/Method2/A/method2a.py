import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

# neural network model with one hidden layer with 500
# nodes and sigmoid activation function
# input and output layer has fixed size
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(500, activation='sigmoid'),
    tf.keras.layers.Dense(10)
])

# using stohastic gradient descent (sgd) and taking accuracy into consideration
model.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# feed data to train the model
model.fit(train_images, train_labels, epochs=10)

# evaluate model and get metrics needed
loss, accuracy = model.evaluate(test_images,  test_labels, verbose=2)

# apply softmax function on the output layer
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

# test model and get predictions
predictions = probability_model.predict(test_images)
labels_predicted = np.argmax(predictions,axis = 1)
f1 = metrics.f1_score(test_labels,labels_predicted,average='macro')

# functions to plot and display the results about probabilities for each image
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 4
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

print('\nTest accuracy:', accuracy)
print('\nF1 score: ', f1)