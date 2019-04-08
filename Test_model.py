
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Reshape, Flatten, \
    Dropout,BatchNormalization, Activation, ZeroPadding2D,LeakyReLU,UpSampling2D, Conv2D
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
import matplotlib.pyplot as plt
import os
import numpy as np

weight_path="./models_and_weights/"

class DCGAN():

    def __init__(self):

        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # load weight
        self.load_weights_from_file()


    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        # model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def load_weights_from_file(self):
        # discriminator weight
        if self.discriminator != None and os.path.isfile(weight_path + "dcgan_discriminator_weight.h5"):
            print("discriminator_weights exists")
            self.discriminator.load_weights(weight_path + "dcgan_discriminator_weight.h5")
        # generator weight
        if self.generator != None and os.path.isfile(weight_path + "dcgan_generator_weight.h5"):
            print("generator_weights exists")
            self.generator.load_weights(weight_path + "dcgan_generator_weight.h5")

    def test_model(self):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
        return


if __name__ == "__main__":
    dcgan = DCGAN()
    dcgan.test_model()