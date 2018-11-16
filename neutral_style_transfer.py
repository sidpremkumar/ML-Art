import matplotlib
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False

import numpy as np
from tqdm import tqdm
import IPython.display
from PIL import Image
import time
import functools

# TODO: Figure out a way to save and write clipped variables

# https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python

import pickle

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

# Global Variables here:
content_path = 'img/c3.jpg'
style_path = 'img/s3.jpg'
style_name = 'rick'
# Size of cropped image
SIZE = 1080
TRAINABLE = False

# The intermidiate layers that we are going to be looking for:
# Content layer where will pull our feature maps
content_layers = ['block5_conv2']

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1'
                ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def main_init():
    # GPU Config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.Session(config=config)

    # #configure the plt
    # plt.figure(figsize=(10, 10))
    #
    # #load content and style images
    # content = load_img(content_path).astype('uint8')
    # style = load_img(style_path).astype('uint8')

    # If Using notebook:
    # #display the images
    # plt.subplot(1, 2, 1)
    # imshow(content, 'Content Image')
    #
    # plt.subplot(1, 2, 2)
    # imshow(style, 'Style Image')
    # plt.show() --- best,

    best, best_loss = driver(content_path, style_path, num_iterations=1000)
    Image.fromarray(best)


def enableEagerExecution():
    tf.enable_eager_execution()
    print("Eager execution: {}".format(tf.executing_eagerly()))


def load_img(path_to_img):
    max_dim = SIZE
    img = Image.open(path_to_img)
    # long
    x = max(img.size)
    scale = float(float(max_dim) / float(x))
    width, height = img.size
    # scale and resize the images, so that they are the same
    img = img.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
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
    # we want to load and preprocess our images
    # we will follow the VGG training method
    img = load_img(path_to_img)

    # Guarentees compatabilty of the image
    # Adequare image to the format the model requires (bc the model will look at multiple dimension)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                               "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_model():
    # Actually creating the VCG19 model
    # we will access intermidiate layers

    # actually load the download the model
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')

    # set it's trainability
    #False in order to get the layers we are interested in
    vgg.trainable = TRAINABLE

    #from docs:
    #Tensor programs work by first building a graph of tf.Tensor objects, detailing how each
    #tensor is computed based on the other available tensors and then by running
    #parts of this graph to achieve the desired results.

    #Get corresponding intermiditate layer
    #This is why we need to enable eager execution
    #style layers that we are interested in, global variable
    #We get these tensors,
    # Note: evaluate them to get a result

    style_outputs = [vgg.get_layer(name).output for name in style_layers]

    # print("Style_outputs type", type(style_outputs[0]))
    # print(style_outputs[0])
    # output = vgg.get_layer('block1_conv1').output
    # print output
    #content layers we are interested in, global variable
    content_outputs = [vgg.get_layer(name).output for name in content_layers]

    #TODO: Average insted of adding
    #averaged = keras.layers.average()([style+_output, content_outputs])

    model_outputs = style_outputs + content_outputs

    # return and build the model
    # from documentation:
    # model = Model(inputs=[a1, a2], outputs=[b1, b2, b3])
    # Train the model with the vgg input
    # set the output to be the ones produced by the style+picture
    return models.Model(vgg.input, model_outputs)


def get_content_loss(base_content, target):
    # if we think of the base content as p (with x-y-z) and target as x (with x-y-z)
    # then this function is returning the equlicdian distance between the two
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    # returning a gram matrix version of the intermidiate layers
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    # compares the gram matrixes of the two, one is the style image,
    height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    #reduce mean goes from a layer with n nodes -> n-1 nodes
    #square, squares the resulting tensor.
    return tf.reduce_mean(tf.square(gram_style - gram_target))


def get_feature_representations(model, content_path, style_path):
    # helper function for gradient decent
    # get the images loaded and processed,

    # load and process out images [ensure that the image is compatable with VGG & return image]
    content_image = load_and_process_img(content_path)
    style_image = load_and_process_img(style_path)

    # compute via the passed model
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    #output is list

    # get the style and content feature representation from out model
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]

    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]


    return style_features, content_features


def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):

    # returns the total loss, stlye loss, content loss, and total variational loss

    style_weight, content_weight = loss_weights

    # pass in our pre-processed image into the model
    #with the vgg as input, and our image as outputs
    # only getting back the layers we are interested in
    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    # get total loss
    loss = style_score + content_score
    return loss, style_score, content_score


def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    # Compute gradients wrt input image
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss


def driver(content_path, style_path, num_iterations=10, content_weight=1e3, style_weight=1e-2):
    # we dont want to train or mess with any layers except the ones we're interested in, so set their trinable to false
    model = get_model()
    #disable training in the model
    for layer in model.layers:
        layer.trainable = False

    # get the style and feature representations, for our interested layers (intermidieate)
    #put the pictures through the model, get the output of the layers we are interested in
    style_features, content_features = get_feature_representations(model, content_path, style_path)

    #turn the output in gram style feature
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # load and process inital image, convert it
    init_image = load_and_process_img(content_path)
    # make the image into a tensor
    init_image = tfe.Variable(init_image, dtype=tf.float32)

    # create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

    # tracker for displaying intermediate images

    iter_count = 1

    # store out best result, initilize variables here
    best_loss, best_img = float('inf'), None

    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }


    # # for displaying
    num_rows = 2
    num_cols = 5

    display_interval = num_iterations / (num_rows * num_cols)
    start_time = time.time()
    global_start = time.time()

    # what we want to normilize the mean around, this is what the VGG networks are trained on
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means
    imgs = []
    #computing the next seetp in gradient decent.
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss

    #apply adam optimization
    opt.apply_gradients([(grads, init_image)])
    #clipped is the init_image tensors clipped by min/max
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)

    init_image.assign(clipped)
    print(type(init_image))
    #Image in tensor form -> init_image -> whats its loss -> all_loss ->
    # gradient decent -> change values -> pass it through the model -> whats its loss?
    again = True
    while (again):
        for i in tqdm(range(num_iterations - 1)):
            # computing the next seetp in gradient decent.
            loss, style_score, content_score = all_loss
            grads, all_loss = compute_grads(cfg)

            # apply adam optimization - version of gradient decent
            opt.apply_gradients([(grads, init_image)])

            # clipped is the init_image tensors clipped by min/max
            # to allow to deprocessing
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            # image = Image.fromarray(plot_img)
            # plot_img = image.numpy()
            # plot_img = deprocess_img(plot_img)
            # imgs.append(plot_img)
            # image.save('outputs/gif/' + str(style_name) + '-' + str(i) + '.bmp')
            if loss < best_loss:
                # updates best loss
                best_loss = loss
                best_img = deprocess_img(init_image.numpy())
            #if i % display_interval == 0:
            if True:
                # Use the .numpy() method to get the concrete numpy array
                plot_img = init_image.numpy()
                plot_img = deprocess_img(plot_img)
                imgs.append(plot_img)
                # IPython.display.clear_output(wait=True)
                # IPython.display.display_png(Image.fromarray(plot_img))
                final_image = Image.fromarray(plot_img)
                # Show the image
                # final_image.show()
                # Save the image
                final_image.save('outputs/' + str(style_name) + '-' + str(i) + '.bmp')

        print('Total time: {:.4f}s'.format(time.time() - global_start))
        # IPython.display.clear_output(wait=True)
        # plt.figure(figsize=(14, 4))
        # for i, img in enumerate(imgs):
        #     plt.subplot(num_rows, num_cols, i + 1)
        #     plt.imshow(img)
        #     plt.xticks([])
        #     plt.yticks([])
        text = raw_input("Do you want to keep going?: 1 - yes, 0 - no")
        if(text == 0):
            again = False






    print("Done!")
    return best_img, best_loss
