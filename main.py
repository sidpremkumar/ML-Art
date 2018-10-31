import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

#Global Variables here:
content_path = 'img/Green_Sea_Turtle_grazing_seagrass.jpg'
style_path = 'img/The_Great_Wave_off_Kanagawa.jpg'



def main():
    #enabling eager execution
    enableEagerExecution()

    #configure the plt
    plt.figure(figsize=(10, 10))

    #load content and style images
    content = load_img(content_path).astype('uint8')
    style = load_img(style_path).astype('uint8')

    #display the images
    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')
    plt.show()

def enableEagerExecution():
    tf.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))


def load_img(path_to_img):
    max_dim = 512.0
    img = Image.open(path_to_img)
    long = max(img.size)
    scale = max_dim / long
    width, height = img.size

    #scale and resize the images, so that they are the same
    img = img.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)
    # We need to broadcast the image array such that it has a batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)

def load_and_process_img(path_to_img):
  #we want to load and preprocess our images
  #we will follow the VGG training method
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img

main()