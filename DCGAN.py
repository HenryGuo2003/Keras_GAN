import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dropout, Flatten
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('tf')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = (X_train.astype(np.float32) - 127.5)/127.5
X_train = X_train[:, :, :, None]

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)
sgd = SGD(lr=0.0005, momentum=0.9, nesterov=True)

generator = Sequential()
generator.add(Dense(input_dim=100, output_dim=1024))
generator.add(Activation('tanh'))
generator.add(Dense(128*7*7))
generator.add(BatchNormalization())
generator.add(Activation('tanh'))
generator.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, (5, 5), padding='same'))
generator.add(Activation('tanh'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, (5, 5), padding='same'))
generator.add(Activation('tanh'))
generator.compile(loss='binary_crossentropy', optimizer='SGD')
generator.summary()

discriminator = Sequential()
discriminator.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
discriminator.add(Activation('tanh'))
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Conv2D(128, (5, 5)))
discriminator.add(Activation('tanh'))
discriminator.add(MaxPooling2D(pool_size=(2, 2)))

discriminator.add(Flatten())
discriminator.add(Dense(1024))
discriminator.add(Activation('tanh'))
discriminator.add(Dense(1))
discriminator.add(Activation('sigmoid'))

discriminator.compile(loss='binary_crossentropy', optimizer=sgd)
discriminator.summary()

# Combined network
gan = Sequential()
gan.add(generator)
discriminator.trainable = False
gan.add(discriminator)
gan.compile(loss='binary_crossentropy', optimizer=sgd)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    #plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images_gan_loss_epoch_%d.png' % epoch)
    plt.close()

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images_gan_generated_image_epoch_%d.png' % epoch)
    plt.close()

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models_gan_generator_epoch_'+str(epoch)+'.h5')
    discriminator.save('models_gan_discriminator_epoch_'+str(epoch)+'.h5')

def train(epochs=1, batchSize=100):
    batchCount = X_train.shape[0] / batchSize
    print 'Epochs:', epochs
    print 'Batch size:', batchSize
    print 'Batches per epoch:', batchCount

    for e in xrange(1, epochs+1):
        print '-'*15, 'Epoch %d' % e, '-'*15
        for _ in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
	    #batchSize= number of picture, randomDim = size of a picture
            noise = np.random.uniform(-1, 1, size=[batchSize, randomDim])
	    #number from 0 to shape choose random images
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize),:,:,:]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise, verbose=0)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.uniform(-1, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

	print ('dloss:')
        print (dloss)
        print ('gloss:')
        print (gloss)
	
        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 20 == 0:
            plotGeneratedImages(e)
            saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)

if __name__ == '__main__':
    train(500, 128)
