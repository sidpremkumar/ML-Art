from keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, LeakyReLU, BatchNormalization, Input, Embedding
from keras.layers import multiply, Dropout, ZeroPadding2D, Reshape, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.initializers import RandomNormal
import keras.backend as K
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from keras.models import model_from_json


import matplotlib.pyplot as plt
import seaborn as sns
# from dcgan import DCGAN
from keras.datasets import mnist
from keras.optimizers import Adam, RMSprop
plt.switch_backend('agg')


#Create a class for the DCGAN

class DCGAN:
    #Params:
    #image_shape -> Shapte of input image
    #z_size -> Size of the latent z vector -> http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/

    def __init__(self, image_shape=(28,28,1), n_filters=64, z_size=(1,1,100),
                 alpha=0.2, lr=5e-5, extra_layers=0, clamp_lower=-0.01,
                 clamp_upper=0.01, disc_iters_per_gen_iters=5, num_classes=10, latent_dim=100):
        # assert image_shape[0] % 8 == 0, "Image shape must be divisable by 8."

        self.num_classes = num_classes
        self.latent_dim = 100

        self.image_shape = image_shape
        self.channels = self.image_shape[2]
        self.n_filters = n_filters
        self.z_size = z_size

        #LeakyReLU value -> Alpha
        self.alpha = alpha

        #Learning rate -> lr
        self.lr = lr
        self.extra_layers = extra_layers
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper

        #DCGAN ->
        self.disc_iters_per_gen_iters = disc_iters_per_gen_iters
        self.weight_init = RandomNormal(mean=0., stddev=0.02)

    def discriminator(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.image_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))



        #Flaten the output
        model.add(Flatten())
        model.summary()
        # #Kinda like adding noise, not sure -> https://keras.io/layers/core/#dense
        # model.add(Dense(1, use_bias=False))


        #Inputs to the model
        img = Input(shape=self.image_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [validity, label])


    def generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))


        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    # def adversarial(self,generator,discriminator):
    #     model = Sequential()
    #     model.add(generator)
    #     model.add(discriminator)
    #     # For the combined model we will only train the generator
    #     self.discriminator.trainable = False
    #     # The generator takes noise and the target label as input
    #     # and generates the corresponding digit of that label
    #     noise = Input(shape=(self.latent_dim,))
    #     label = Input(shape=(1,))
    #     img = self.generator([noise, label])
    #
    #     # The discriminator takes generated image as input and determines validity
    #     # and the label of that image
    #     valid, target_label = self.discriminator(img)
    #
    #     # The combined model  (stacked generator and discriminator)
    #     # Trains the generator to fool the discriminator
    #
    #     return Model([noise, label], [valid, target_label])

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true*y_pred)


#Create a class that will train the generator

class Trainer:
    def __init__(self, dcgan, optimizer='adam', plot_path='plots'):
        assert optimizer.lower() in ['adam', 'rmsprop'], "Optimizer is not supported :("

        #reformating the plot_paths directory string
        if plot_path.endswith('/'):
            plot_path = plot_path[:-1]

        #If the plot does not exist, make it !
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)

        #Assign our DCGAN
        self.dcgan = dcgan
        #define the latent z vector size
        self.z_size = dcgan.z_size
        self.latent_dim = dcgan.latent_dim
        #same with learning rate
        self.lr = dcgan.lr
        #build out this network, create each of the CNN
        #if we have previously computed values check there first:
        with tqdm(total=6) as pbar:
            if(os.path.exists('temp/discriminator_architecture.json')):
                #Load what we have previously computed!
                with open('temp/discriminator_architecture.json', 'r') as f:
                    self.discriminator = model_from_json(f.read())
                    pbar.update(1)
                if(os.path.exists('temp/discriminator_weights.h5')):
                    self.discriminator.load_weights('temp/discriminator_weights.h5')
                    pbar.update(1)
                if(os.path.exists('temp/adversarial_architecture.json')):
                    with open('temp/adversarial_architecture.json', 'r') as f:
                        self.adversarial = model_from_json(f.read())
                        pbar.update(1)
                if(os.path.exists('temp/adversarial_weights.h5')):
                    self.adversarial.load_weights('temp/adversarial_weights.h5')
                    pbar.update(1)

                if(os.path.exists('temp/generator_architecture.json')):
                    with open('temp/generator_architecture.json', 'r') as f:
                        self.generator = model_from_json(f.read())
                        pbar.update(1)
                if(os.path.exists('temp/generator_weights.h5')):
                    self.generator.load_weights('temp/generator_weights.h5')
                    pbar.update(1)
                print("Pre-Loaded from saved files!")
            else:
                self.discriminator = dcgan.discriminator()
                pbar.update(3)
                self.generator = dcgan.generator()
                pbar.update(3)
                # self.adversarial = dcgan.adversarial(self.generator, self.discriminator)

        (self.x_train, self.x_train_class), (self.x_test, self.x_test_class) = self.dataset()
        self.model_compiler(optimizer)
        self.plot_path = plot_path


        # -->>>
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
        self.combined.compile(loss=losses,
            optimizer=self.opt)
        self.discriminator.trainable = True


    def model_compiler(self,optimizer):
        if optimizer.lower() == 'adam':
            self.opt = Adam(lr=self.lr, beta_1=0.5,beta_2=0.9)
        elif optimizer.lower() == 'rmsprop':
            self.opt = RMSprop(lr=self.lr)

        #build the generator and discriminator
        self.discriminator.compile(optimizer=self.opt, loss=self.dcgan.wasserstein_loss,
                                    metrics=['accuracy'])
        # self.generator.compile(optimizer=opt, loss=self.dcgan.wasserstein_loss)

        #ensure the discriminator is not trainable and then create the adversarial (G vs D)

        # self.discriminator.trainable = False
        # self.adversarial.compile(optimizer=opt, loss=self.dcgan.wasserstein_loss)
        # self.discriminator.trainable = True

    def interval_mapping(image, from_min, from_max, to_min, to_max):
        # map values from [from_min, from_max] to [to_min, to_max]
        # image: input array
        from_range = from_max - from_min
        to_range = to_max - to_min
        scaled = np.array((image - from_min) / float(from_range), dtype=float)
        return to_min + (scaled * to_range)

    def dataset(self):
        #Load in and convert the data set

        (x_train, x_train_class), (x_test,x_test_class) = mnist.load_data()

        # x_train = np.reshape(x_train, (-1,28,28,1))
        # train_padded = np.zeros((np.shape(x_train)[0], 32, 32, 1))
        # train_padded[:, 2:30, 2:30, :] = x_train
        # train_padded /= np.max(train_padded)
        # train_padded *= 2
        # train_padded -= 1
        #
        # x_test = np.reshape(x_test, (-1, 28, 28, 1))
        # test_padded = np.zeros((np.shape(x_test)[0], 32, 32, 1))
        # test_padded[:, 2:30, 2:30, :] = x_test
        # test_padded /= np.max(test_padded)
        # test_padded *= 2
        # test_padded -= 1

        #normalize the input data 0-1
        #scaling 0-255 -> -1-+1
        #First subtract it by 127.5
        #Then drive by 127.5
        #

        x_train = (x_train.astype(np.float32) - 127.5) / 127.5

        x_train = np.expand_dims(x_train, axis=3)

        # x_trainnew = []
        # for x in x_train:
        #     print(x.shape)
        #     new = self.interval_mapping(x, 0, 255, -1, 1)
        #     x_trainnew.append(new)
        #     print(x_trainnew)


        #Repeat for the rest of the data
        x_train_class = x_train_class.reshape(-1, 1)

        x_test = (x_test.astype(np.float32) - 127.5) / 127.5
        x_test = np.expand_dims(x_test, axis=3)
        x_test_class = x_test_class.reshape(-1, 1)


        return (x_train, x_train_class), (x_test,x_test_class)



    def get_batch(self,batch_size, train=True):
        if train:
            #get a random batch from the training data set
            idx = np.random.randint(0, self.x_train.shape[0], batch_size)
            idy = np.random.randint(0, self.x_train_class.shape[0], batch_size)
            return self.x_train[idx], self.x_train_class[idx]
        else:
            #get a random batch from the test data set
            idx = np.random.randint(0, self.x_test.shape[0], batch_size)
            idy = np.random.randint(0, self.x_test_class.shape[0], batch_size)
            return self.x_train[idx], self.x_test_class[idx]

    def make_noise(self,batch_size):
        return np.random.normal(loc=0,scale=1, size=(batch_size,self.z_size[2]))

    def gen_batch(self, batch_size):
        noise = np.random.normal(0, 1, (batch_size, self.z_size[2]))
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))
        # latent_vector_batch = self.make_noise(batch_size)
        gen_output = self.generator.predict([noise, sampled_labels])



        return gen_output, sampled_labels

    def plot_dict(self, dictonary):
        for key, item in dictonary.items():
            plt.close()
            plt.plot(range(len(item)), item)
            plt.title(str(key))
            #TODO: Save figure
            plt.savefig(self.plot_path+'/{}.png'.format(key), bbox_inches='tight')


    #Not sure about this function - > ??
    def make_images(self, epoch, num_images=3):
        print("Saving Images! Epoch: ", epoch)
        noise = self.make_noise(num_images)
        label = np.array([1,2,3])
        digits = self.generator.predict([noise, label])
        digits = (127.5 * digits) + 127.5
        # for x in range(28):
        #     for y in range(28):
        #         print(digits[0,y,x,0])
        m = 0
        while m < num_images:
            img = Image.fromarray(digits[m,:,:,0])
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(self.plot_path+'/epoch_{}-image_{}.png'.format(epoch, m), bbox_inches='tight')
            else:
                img.save(self.plot_path+'/epoch_{}-image_{}.png'.format(epoch, m), bbox_inches='tight')
            # #used to show the heatmap of the image. Shows concentration of prediction
            plt.close()
            image = sns.heatmap(digits[m,:,:,0], cbar=False, square=True)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout(pad=0)
            plt.savefig(self.plot_path+'/heatmap_epoch_{}-image_{}.png'.format(epoch, m), bbox_inches='tight')
            m += 1

    def train(self, num_epochs=25, batch_size=32):
        batches_per_epoch = np.shape(self.x_train)[0] // batch_size
        stats = {'wasserstein_distance': [], 'generator_loss':[]}

        gen_iterations = 0



        for epoch in tqdm(range(num_epochs)):
            valid = []
            for x in range(batch_size):
                valid.append(1)

            fake = np.zeros((batch_size, 1))

            #fake = [0,0,0...]
            #valid =
            if (epoch) % 5 == 0:
                print("Checkpoint!")
                self.make_images(epoch+1, num_images=3)

                #Save all our models!
                self.discriminator.save_weights('temp/discriminator_weights.h5')
                with open('temp/discriminator_architecture.json', 'w') as f:
                        f.write(self.discriminator.to_json())

                # self.adversarial.save_weights('temp/adversarial_weights.h5')
                # with open('temp/adversarial_architecture.json', 'w') as f:
                #         f.write(self.adversarial.to_json())

                self.generator.save_weights('temp/generator_weights.h5')
                with open('temp/generator_architecture.json', 'w') as f:
                        f.write(self.generator.to_json())

            for i in tqdm(range(batches_per_epoch)):
                #train the discriminator
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    disc_iterations = 100
                else:
                    disc_iterations = self.dcgan.disc_iters_per_gen_iters

                for j in tqdm(range(disc_iterations)):

                    for l in self.discriminator.layers:
                        #in the models layers
                        weights = l.get_weights()
                        #clamp the weights
                        weight = [np.clip(w, self.dcgan.clamp_lower, self.dcgan.clamp_upper) for w in weights]
                        #Set the new clampped weights
                        l.set_weights(weights)

                    #train with real data:
                    (data_batch, data_batch_class) = self.get_batch(batch_size)
                    #np.ones means that this is 1 -> true -> real numbers

                    disc_loss_real = self.discriminator.train_on_batch(data_batch, [valid, data_batch_class])

                    #train with batch of generator (fake) data
                    #-np.ones means that this is -1 -> false -> fake
                    #gen_batch_class is an array of 10's, indicating fake image
                    gen_batch, gen_batch_class = self.gen_batch(batch_size)
                    disc_loss_fake = self.discriminator.train_on_batch(gen_batch, [fake, gen_batch_class])

                #train the generator now

                #make random noise
                # noise = self.make_noise(batch_size)
                #
                # #Disable the trainablity of the discriminator
                # self.discriminator.trainable = False
                #
                # #train the adversarial (G vs D) with np.ones 1 -> True
                #
                # gen_loss = self.adversarial.train_on_batch(noise, np.ones(batch_size))
                # self.discriminator.trainable = True
                #
                # stats['generator_loss'].append(gen_loss)
                # stats['wasserstein_distance'].append(-(disc_loss_real + disc_loss_fake))

                gen_iterations += 1

        self.plot_dict(stats)

if __name__ == '__main__':
    #Check that we are using the GPU & Configure it

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.Session(config=config)
    #Check that we are using the GPU
    K.tensorflow_backend._get_available_gpus()

    dcgan = DCGAN()
    trainer = Trainer(dcgan)
    trainer.train()