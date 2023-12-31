import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate


KERNEL_INITIALIZER = "HeNormal"
KERNEL_INITIALIZER = "he_normal"


# --------------------------------------------
#
# U-Net Encoder Block
#
# --------------------------------------------
def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer=KERNEL_INITIALIZER)(inputs)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer=KERNEL_INITIALIZER)(conv)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


# --------------------------------------------
#
# U-Net Decoder Block
#
# --------------------------------------------
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
        n_filters,
        (3, 3),    # Kernel size
        strides=(2, 2),
        padding='same'
    )(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters,
                  3,     # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer=KERNEL_INITIALIZER)(merge)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer=KERNEL_INITIALIZER)(conv)
    return conv


# --------------------------------------------
#
#  UNET model
#
# --------------------------------------------


def UNet(input_size=(128, 128, 3), n_filters=32, n_classes=3, dropout_prob=0.3):
    """
        Combine both encoder and decoder blocks according to the U-Net research paper
        Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8, dropout_prob=dropout_prob, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=dropout_prob, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used with the upsampling
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# --------------------------------------------
#
#  Shallow UNET model
#
# 1------>5
#  \     /
#   2-->4
#    \ /
#     3
# --------------------------------------------


def ShallowUNet(input_size=(128, 128, 3), n_filters=32, n_classes=3, dropout_prob=0.3):
    """
        Combine both encoder and decoder blocks according to the U-Net research paper
        Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=dropout_prob, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    ublock4 = DecoderMiniBlock(cblock3[0], cblock2[1], n_filters * 2)
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used with the upsampling
    ublock5 = DecoderMiniBlock(ublock4, cblock1[1], n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock5)

    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


if __name__ == "__main__":
    SHALLOW_UNET = True  # True to use shallow Unet (3 levels), False (5 levels)

    if SHALLOW_UNET:
        unet = ShallowUNet(input_size=(128, 128, 3), n_filters=32, n_classes=3)
        print(unet.summary())
    else:
        unet = UNet(input_size=(128, 128, 3), n_filters=32, n_classes=3)
        print(unet.summary())
=======
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

# for bulding and running deep learning model
import tensorflow as tf


KERNEL_INITIALIZER = "HeNormal"
KERNEL_INITIALIZER = "he_normal"


# --------------------------------------------
#
# U-Net Encoder Block
#
# --------------------------------------------
def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,   # Kernel size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer=KERNEL_INITIALIZER)(inputs)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,   # Kernel size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer=KERNEL_INITIALIZER)(conv)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = tf.keras.layers.BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


# --------------------------------------------
#
# U-Net Decoder Block
#
# --------------------------------------------
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = tf.keras.layers.Conv2DTranspose(
        n_filters,
        (3, 3),    # Kernel size
        strides=(2, 2),
        padding='same'
    )(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = tf.keras.layers.concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,     # Kernel size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer=KERNEL_INITIALIZER)(merge)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,   # Kernel size
                                  activation='relu',
                                  padding='same',
                                  kernel_initializer=KERNEL_INITIALIZER)(conv)
    return conv


# --------------------------------------------
#
#  UNET model
#
# --------------------------------------------


def UNet(input_size=(128, 128, 3), n_filters=32, n_classes=3, dropout_prob=0.3):
    """
        Combine both encoder and decoder blocks according to the U-Net research paper
        Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters * 8, dropout_prob=dropout_prob, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters * 16, dropout_prob=dropout_prob, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used with the upsampling
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(ublock9)

    conv10 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# --------------------------------------------
#
#  Shallow UNET model
#
# 1------>5
#  \     /
#   2-->4
#    \ /
#     3
# --------------------------------------------


def ShallowUNet(input_size=(128, 128, 3), n_filters=32, n_classes=3, dropout_prob=0.3):
    """
        Combine both encoder and decoder blocks according to the U-Net research paper
        Return the model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters * 4, dropout_prob=dropout_prob, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filters
    # Observe the skip connections from the encoder are given as input to the decoder
    ublock4 = DecoderMiniBlock(cblock3[0], cblock2[1], n_filters * 2)
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used with the upsampling
    ublock5 = DecoderMiniBlock(ublock4, cblock1[1], n_filters)

    # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = tf.keras.layers.Conv2D(n_filters,
                                   3,
                                   activation='relu',
                                   padding='same',
                                   kernel_initializer='he_normal')(ublock5)

    conv10 = tf.keras.layers.Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


if __name__ == "__main__":
    SHALLOW_UNET = True  # True to use shallow Unet (3 levels), False (5 levels)

    if SHALLOW_UNET:
        unet = ShallowUNet(input_size=(128, 128, 3), n_filters=32, n_classes=3)
        print(unet.summary())
    else:
        unet = UNet(input_size=(128, 128, 3), n_filters=32, n_classes=3)
        print(unet.summary())
