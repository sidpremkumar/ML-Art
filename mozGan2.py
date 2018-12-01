# -*- coding: utf-8 -*-
"""
Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the
MNIST dataset. See https://arxiv.org/abs/1610.09585 for more details.
You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. Prefer the TensorFlow backend if you plan on iterating,
as the compilation time can be a blocker using Theano.
Timings:
Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min
Consult https://github.com/lukedeo/keras-acgan for more information and
example output
"""
from __future__ import print_function
import os
from neutral_style_transfer import *
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np
from tensorflow.python.keras.preprocessing import image as kp_image

np.random.seed(1337)
num_classes = 10


def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 28, 28, 1)
    model = Sequential()

    model.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
    model.add(Reshape((3, 3, 384)))

    # upsample to (7, 7, ...)
    model.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())

    # upsample to (14, 14, ...)
    model.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())

    # upsample to (28, 28, ...)
    model.add(Conv2DTranspose(3, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal'))

    # this is the z space commonly referred to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Embedding(num_classes, latent_size,
                    embeddings_initializer='glorot_normal')(image_class)

    # hadamard product between z-space and a class conditional embedding
    input = layers.multiply([latent, cls])
    #what is generated is the input of the model
    fake_image = model(input)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    model = Sequential()
    model.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, 3)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, 3, padding='same', strides=1))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, 3, padding='same', strides=2))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, 3, padding='same', strides=1))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())

    image = Input(shape=(28, 28, 3))

    features = model(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    #is the image fake or not?
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    #class the image belongs too
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])

def neural_transfer(x_train, x_test):
    xtrain = 0
    xtest = 0
    for image in tqdm(x_train):
        #content_path, style_path='img/s1.jpg', num_iterations=10, content_weight=1e3, style_weight=1e-2, SAVE_ITERATION  = True
        img = Image.fromarray(image)
        img = img.convert('RGB')
        img = driver(img, num_iterations=30, SAVE_ITERATION=False)
        img.save('x_train/' + str(xtrain) + '.bmp')
        xtrain += 1
    # for image in tqdm(x_test):
    #     #content_path, style_path='img/s1.jpg', num_iterations=10, content_weight=1e3, style_weight=1e-2, SAVE_ITERATION  = True
    #     img = Image.fromarray(image)
    #     img = img.convert('RGB')
    #     img = driver(img, num_iterations=30, SAVE_ITERATION=False)
    #     img.save('x_test/'  +str(xtest) + '.bmp')
    #     xtest += 1

    print("CONVERTED IMAGES")
    exit()
    return x_train, x_test

def load_transfer_data():
    x_train = np.empty((2000,28,28,3))
    print("Loading images...")
    for x in tqdm(range(2000)):
        file = Image.open('x_train/' + str(x) + '.bmp')
        img = np.array(file)
        x_train = np.append(x_train,[img], axis=0)
    (_, y_train), (x_test, y_test) = mnist.load_data()
    # x_test = np.squeeze(x_test, axis=(4,))
    # print(x_test.shape)

    return (x_train, y_train), (x_test, y_test)


    return x_train, x_test
if __name__ == '__main__':
    i = 0
    #enable eager execution
    # enableEagerExecution()
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = load_transfer_data()

    # x_train, x_test = neural_transfer(x_train, x_test)

    # batch and latent size taken from the paper
    epochs = 100
    batch_size = 10
    latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # build the generator
    generator = build_generator(latent_size)
    #latent z vector
    latent = Input(shape=(latent_size, ))
    #the number of classes (in MNIST this is 10)
    image_class = Input(shape=(1,), dtype='int32')

    # generate a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    #retuns if the image is fake or not, and the class it belongs too
    fake, aux = discriminator(fake)
    #create a combined model (G vs D)
    combined = Model([latent, image_class], [fake, aux])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # get our mnist data, and force it to be of shape (..., 28, 28, 1)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #
    # #If we want to first stylize our input
    # x_train, x_test = neural_transfer(x_train, x_test)
    #normalize the input to -1 to 1
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    #flatten
    x_train = np.expand_dims(x_train, axis=-1)


    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    #extract relevent info
    num_train, num_test = x_train.shape[0], x_train.shape[0]


    train_history = defaultdict(list)
    test_history = defaultdict(list)
    x_train = np.squeeze(x_train, axis=(4,))
    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(np.ceil(x_train.shape[0] / float(batch_size)))
        progress_bar = Progbar(target=num_batches)

        #for saving information
        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # get a batch of real images
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]

            label_batch = y_train[index * batch_size:(index + 1) * batch_size]
            #print(image_batch[0][2][5]*127.5+127.5)
            # temmp = Image.fromarray((image_batch[0]*127.5+127.5).astype(np.uint8))
            # temmp.save('plots/' + str(label_batch[0]) + '.bmp')

            # generate a new batch of noise
            noise = np.random.uniform(-1, 1, (len(image_batch), latent_size))

            # sample some labels from p_c --- random classes
            sampled_labels = np.random.randint(0, num_classes, len(image_batch))

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (len(image_batch), 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)


            x = np.concatenate((image_batch, generated_images))

            # use one-sided soft real/fake labels
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            # essentially changing 1 and 0 to 0 and 0.95. The idea is
            # that it doesnt let extenrious numbers to mess with the model
            soft_zero, soft_one = 0, 0.95
            y = np.array(
                [soft_one] * len(image_batch) + [soft_zero] * len(image_batch))
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # we don't want the discriminator to also maximize the classification
            # accuracy of the auxiliary classifier on generated images, so we
            # don't train discriminator to produce class labels for generated
            # images (see https://openreview.net/forum?id=rJXTf9Bxg).
            # To preserve sum of sample weights for the auxiliary classifier,
            # we assign sample weight of 2 to the real images.
            # for the generated images it's random
            disc_sample_weight = [np.ones(2 * len(image_batch)),
                                  np.concatenate((np.ones(len(image_batch)) * 2,
                                                  np.zeros(len(image_batch))))]

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            noise = np.random.uniform(-1, 1, (2 * len(image_batch), latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * len(image_batch))

            # we want to train the generator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = np.ones(2 * len(image_batch)) * soft_one

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

            progress_bar.update(index + 1)

        print('Testing for epoch {}:'.format(epoch))

        # evaluate the testing loss here

        # generate a new batch of noise
        noise = np.random.uniform(-1, 1, (num_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        x = np.concatenate((x_train, generated_images))
        y = np.array([1] * num_test + [0] * num_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # see if the discriminator can figure itself out...
        # discriminator_test_loss = discriminator.evaluate(
        #     x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        #test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        # print(ROW_FMT.format('discriminator (test)',
        #                      *test_history['discriminator'][-1]))

        # save weights every epoch
        generator.save_weights(
            'plots/weights/params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'plots/weights/params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        num_rows = 40
        noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)),
                        (num_classes, 1))

        sampled_labels = np.array([
            [i] * num_rows for i in range(num_classes)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)
        print(generated_images[0][5][6]*127.5 + 127.5)
        generated_image = Image.fromarray((generated_images[0]*127.5 + 127.5).astype(np.uint8))

        generated_image.save('plots/' + str(i) + '_' + str(sampled_labels[0]) + '.bmp')
        i += 1

        # # prepare real images sorted by class label
        # real_labels = y_test[(epoch - 1) * num_rows * num_classes:
        #                       epoch * num_rows * num_classes]
        # indices = np.argsort(real_labels, axis=0)
        # real_images = x_test[(epoch - 1) * num_rows * num_classes:
        #                       epoch * num_rows * num_classes][indices]
        #
        # # display generated images, white separator, real images
        # img = np.concatenate(
        #     (generated_images,
        #      np.repeat(np.ones_like(x_test[:1]), num_rows, axis=0),
        #      real_images))
        #
        # # arrange them into a grid
        # img = (np.concatenate([r.reshape(-1, 28)
        #                        for r in np.split(img, 2 * num_classes + 1)
        #                        ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        #
        # Image.fromarray(img).save(
        #     'plot_epoch_{0:03d}_generated.png'.format(epoch))

    with open('acgan-history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)
