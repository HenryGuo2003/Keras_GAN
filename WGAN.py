#This GAN is for Image Classification
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dropout, Flatten
from keras.layers import Dense, Embedding
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.optimizers import *
from keras.initializers import *
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('tf')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
Class_Num = 10
randomDim = 100

# Load MNIST data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
#X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = (X_train.astype(np.float32) - 127.5)/127.5
#X_train = X_train[:, :, :, None]
X_train = np.expand_dims(X_train, axis=-1)
X_test = (X_test.astype(np.float32) - 127.5)/127.5
#X_test = X_test[:, :, :, None]
X_test = np.expand_dims(X_test, axis=-1)

#initializer
weight_init = RandomNormal(mean=0., stddev=0.02)

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)
rms = RMSprop(lr=0.00005)
sgd = SGD(lr=0.0005, momentum=0.9, nesterov=True)

#WGAN self define loss
def d_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

gen = Sequential()
gen.add(Dense(1024, input_dim=randomDim))
gen.add(LeakyReLU())
gen.add(Dense(128*7*7))
gen.add(LeakyReLU())
gen.add(Reshape((7, 7, 128)))
gen.add(UpSampling2D(size=(2, 2)))
gen.add(Conv2D(64, (5, 5), padding='same'))
gen.add(LeakyReLU())
gen.add(UpSampling2D(size=(2, 2)))
gen.add(Conv2D(1, (5, 5), padding='same'))
gen.add(Activation('tanh'))
gen.summary()

Input_Z = Input(shape=(randomDim, )) #noise
Input_Class = Input(shape=(1,), dtype='int32') #class
#Class_Num is Input Dim, randomDim is Output Dim
#Embed Input_Class from Class_Num to randomDim
e = Embedding(Class_Num, randomDim, embeddings_initializer='glorot_normal')(Input_Class)
Embedded_Class = Flatten()(e)
h = layers.multiply([Input_Z, Embedded_Class])

Fake_Image = gen(h)
generator = Model([Input_Z, Input_Class], Fake_Image)

dis = Sequential()
dis.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
dis.add(LeakyReLU(0.2))
dis.add(Dropout(0.3))
dis.add(Conv2D(128, (5, 5)))
dis.add(LeakyReLU(0.2))
dis.add(Dropout(0.3))

dis.add(Flatten())
im = Input(shape=(28,28,1))
features = dis(im)

Fake = Dense(1)(features)
Classify = Dense(Class_Num, activation='softmax')(features)

discriminator = Model(im, [Fake, Classify])
discriminator.compile(optimizer=rms, loss=[d_loss, 'sparse_categorical_crossentropy'])

# Combined network
ganInput = Input(shape=(randomDim,))
Gan_Class = Input(shape=(1,), dtype='int32')
Fake = generator([ganInput, Gan_Class])
discriminator.trainable = False
Fake_Image, Output_Class = discriminator(Fake)
gan = Model(inputs=[ganInput, Gan_Class], outputs=[Fake_Image, Output_Class])
gan.compile(loss=[d_loss, 'sparse_categorical_crossentropy'], optimizer=rms)

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
    noise = np.random.normal(0., 1., size=[examples, randomDim])
    sampled_labels = np.random.randint(0, Class_Num, size=128)
    generatedImages = generator.predict([noise, sampled_labels])
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

def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print 'Epochs:', epochs
    print 'Batch size:', batchSize
    print 'Batches per epoch:', batchCount

    for e in xrange(1, epochs+1):
        print '-'*15, 'Epoch %d' % e, '-'*15
        for _ in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
	    #batchSize= number of picture, randomDim = size of a picture
            noise = np.random.normal(0., 1., size=[batchSize, randomDim])
	    Random_Num = np.random.randint(0, X_train.shape[0], size=batchSize)
	    #number from 0 to shape choose random images and labels
            imageBatch = X_train[Random_Num]
	    labelBatch = Y_train[Random_Num]

            discriminator.trainable = True
	    for l in discriminator.layers: l.trainable = True
	    for l in discriminator.layers:
		weights = l.get_weights()
		#set weight to -0.01~0.01
		weights = [np.clip(w, -0.01, 0.01) for w in weights]
		l.set_weights(weights)

            # Generate fake MNIST images
	    sampled_labels = np.random.randint(0, Class_Num, size=batchSize)
            generatedImages = generator.predict([noise, sampled_labels.reshape((-1,1))], verbose=0)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
	    #multiply with -1 because we want loss to be minimum
	    yDis = -np.ones(2*batchSize)
            # One-sided label smoothing
	    yDis[batchSize:] = 1
	    class_y = np.concatenate((labelBatch, sampled_labels), axis=0)

            # Train discriminator
            dloss = discriminator.train_on_batch(X, [yDis, class_y])

            # Train generator
            noise = np.random.normal(0., 1., size=[2 * batchSize, randomDim])
	    sampled_labels = np.random.randint(0, Class_Num, size=2 * batchSize)
            yGen = -np.ones(2 * batchSize)
            discriminator.trainable = False
	    for l in discriminator.layers: l.trainable = False
            gloss = gan.train_on_batch([noise, sampled_labels.reshape((-1,1))], [yGen, sampled_labels])

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 20 == 0:
            plotGeneratedImages(e)
            saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)

    Y_is_fake = np.ones(X_test.shape[0])
    score = discriminator.evaluate(X_test, [Y_is_fake, Y_test])
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    train(300, 128)
