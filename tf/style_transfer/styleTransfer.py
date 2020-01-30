import tensorflow as tf
import numpy as np
import matplotlib as mpl
import cv2
import png

import matplotlib.pyplot as plt
from pdb import set_trace as bp
from PIL import Image


original_img = cv2.imread('me.jpg' , cv2.IMREAD_COLOR)    
#plt.imshow(original_img, cmap='gray', vmin=0, vmax=1)
#plt.show()
#img = Image.fromarray(original_img, 'RGB')
#img.show()

x = tf.keras.applications.vgg19.preprocess_input(original_img*(255))
print(x)
x = tf.image.resize(x, (224, 224))
print(x)
#x = x[tf.newaxis, :]
#bp()
plt.imshow(x, cmap='gray', vmin=0, vmax=1)
plt.show()

print(np.shape(x.numpy()))
#bp()
#img = Image.fromarray(x, 'RGB')
img = Image.fromarray(x.numpy(), 'RGB')
img.show()

vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
x = x[tf.newaxis, :]
prediction_probabilities = vgg(x)
print(prediction_probabilities.shape)
predicted_top_5= tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
[(class_name, prob) for (number, class_name, prob) in predicted_top_5]
print(predicted_top_5)
#bp()
