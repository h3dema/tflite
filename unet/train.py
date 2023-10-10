#
# Train the segmentation model (UNet)
#
# TODO: select between CPU or GPU
#
import argparse
import os
import random
import numpy as np  # for using np arrays

# for reading and processing images
try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio

from PIL import Image

# for visualizations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

import tensorflow as tf
from unet import UNet, ShallowUNet


def LoadData(path1, path2, limit: int = None):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively

    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)

    # Make a list for images and masks filenames
    orig_img = []
    for file in image_dataset:
        orig_img.append(file)

    if limit is not None:
        print("Limit dataset to {} samples".format(limit))
        orig_img = random.sample(orig_img, limit)

    # Sort the lists to get both of them in same order
    orig_img.sort()

    # the dataset has exactly the same name for images and corresponding masks
    # mask_img = [os.path.join(path2, os.path.basename(fname).replace(".jpg", ".png")) for fname in orig_img]
    mask_img = [os.path.basename(fname).replace(".jpg", ".png") for fname in orig_img]
    # for file in mask_dataset:
    #     mask_img.append(file)
    # mask_img.sort()

    return orig_img, mask_img


def PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2):
    """
    Processes the images and mask present in the shared list and path
    Returns a NumPy dataset with images as 3-D arrays of desired size
    Please note the masks in this dataset have only one channel
    """
    # Pull the relevant dimensions for image and mask
    m = len(img)                     # number of images
    i_h, i_w, i_c = target_shape_img   # pull height, width, and channels of image
    m_h, m_w, m_c = target_shape_mask  # pull height, width, and channels of mask

    # Define X and Y as number of images along with shape of one image
    X = np.zeros((m, i_h, i_w, i_c), dtype=np.float32)
    y = np.zeros((m, m_h, m_w, m_c), dtype=np.int32)

    # Resize images and masks
    for file in img:
        # convert image into an array of desired shape (3 channels)
        index = img.index(file)
        path = os.path.join(path1, file)
        single_img = Image.open(path).convert('RGB')
        single_img = single_img.resize((i_h, i_w))
        single_img = np.reshape(single_img, (i_h, i_w, i_c))
        single_img = single_img / 256.
        X[index] = single_img

        # convert mask into an array of desired shape (1 channel)
        single_mask_ind = mask[index]
        path = os.path.join(path2, single_mask_ind)
        single_mask = Image.open(path)
        single_mask = single_mask.resize((m_h, m_w))
        single_mask = np.reshape(single_mask, (m_h, m_w, m_c))
        single_mask = single_mask - 1  # to ensure classes #s start from 0
        y[index] = single_mask
    return X, y


def plot_bias_variance(results, out_fname: str = None):
    # High Bias is a characteristic of an underfitted model and we would observe low accuracies for both train and validation set
    # High Variance is a characterisitic of an overfitted model and we would observe high accuracy for train set and low for validation set
    # To check for bias and variance plit the graphs for accuracy
    # I have plotted for loss too, this helps in confirming if the loss is decreasing with each iteration - hence, the model is optimizing fine
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(results.history["loss"], color='r', label='train loss')
    axis[0].plot(results.history["val_loss"], color='b', label='dev loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    if "accuracy" in results.history:
        # tensorflow 2+
        axis[1].plot(results.history["accuracy"], color='r', label='train accuracy')
        axis[1].plot(results.history["val_accuracy"], color='b', label='dev accuracy')
    else:
        # tensorflow 1.15
        axis[1].plot(results.history["acc"], color='r', label='train accuracy')
        axis[1].plot(results.history["val_acc"], color='b', label='dev accuracy')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()

    # RESULTS
    # The train loss is consistently decreasing showing that Adam is able to optimize the model and find the minima
    # The accuracy of train and validation is ~90% which is high enough, so low bias
    # and the %s aren't that far apart, hence low variance
    if out_fname is None:
        plt.show()
    else:
        fig.savefig(out_fname)
    plt.close(fig)


def visualize_output(X, y, index, out_fname: str = None):
    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(X[image_index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y[image_index, :, :, 0])
    arr[1].set_title('Processed Masked Image ')
    if out_fname is None:
        plt.show()
    else:
        fig.savefig(out_fname)
    plt.close(fig)


def VisualizeResults(index, X_valid, y_valid, model):
    # show results of Validation Dataset
    img = X_valid[index]
    img = img[np.newaxis, ...]
    pred_y = model.predict(img)
    pred_mask = tf.argmax(pred_y[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_valid[index, :, :, 0])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:, :, 0])
    arr[2].set_title('Predicted Masked Image ')
    fig.savefig(f"visualize_results_{index}.png")
    plt.close(fig)


def train(X_train, X_valid, y_train, y_valid,
          model,
          epochs=1,
          ):
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        verbose=1
    )

    # There are multiple optimizers, loss functions and metrics that can be used to compile multi-class segmentation models
    # Ideally, try different options to get the best accuracy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    # Run the model in a mini-batch fashion and compute the progress for each epoch
    results = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=epochs,
        validation_data=(X_valid, y_valid),
        callbacks=[cp_callback],  # Pass callback to training
    )

    # save final model
    model.save_weights(checkpoint_path.format(epoch=0))

    #
    # Bias Variance Check
    #
    plot_bias_variance(results, out_fname="bias_variance_check.png")

    #
    # View Predicted Segmentations
    #
    #
    model.evaluate(X_valid, y_valid)

    # Add any index to contrast the predicted mask with actual mask
    index = min(X_valid.shape[0] - 1, 700)
    VisualizeResults(index, X_valid, y_valid, model)

    return model


def NN(input_size, n_units=[128], n_classes=3, dropout: float=0):
    assert 0 <= dropout < 1, "Range of dropout should be between 0 and 1"

    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(shape=input_size)
    x = inputs
    for i, n in enumerate(n_units):
        x = tf.keras.layers.Dense(n, activation="relu", name=f"layer{i+1}")(x)
        if dropout > 0:
            x = tf.keras.layers.Dropout(dropout)(x, training=False)

    # create last layer.
    # remember to increment "i" -> thus i + 2
    x = tf.keras.layers.Dense(n_classes, activation="softmax", name=f"layer{i+2}")(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def Conv(input_size, filters=[3], kernels=[128], max_pool: list = None, n_classes=3, dropout: float=0, batch_normalization: bool = True):
    assert 0 <= dropout < 1, "Range of dropout should be between 0 and 1"
    assert len(filters) == len(kernels)
    assert max_pool is None or len(max_pool) == len(kernels)

    if max_pool is None:
        max_pool = [None] * len(kernels)

    if filters[-1] != n_classes:
        filters[-1] = n_classes

    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = tf.keras.layers.Input(shape=input_size)
    x = inputs
    for i, [_filter, _kernel, _max] in enumerate(zip(filters, kernels, max_pool)):
        x = tf.keras.layers.Conv2D(
            _filter, _kernel,
            activation="relu" if i < len(filters) - 1 else "softmax",
            padding='same')(x)
        if _max is not None:
            x = tf.keras.layers.MaxPooling2D(
                pool_size=_max,
                strides=1,
                padding='same')(x)

        if i < len(filters) - 1:
            if batch_normalization:
                x = tf.keras.layers.BatchNormalization()(x, training=False)

            if dropout > 0:
                x = tf.keras.layers.Dropout(dropout)(x, training=False)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


if __name__ == "__main__":
    #
    # data
    #
    path1 = 'images/original/'
    path2 = 'images/masks/'
    SHALLOW_UNET = True

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output-dir', type=str, default="output_dir")
    parser.add_argument('--epochs', type=int, default=20)

    parser.add_argument("--model-name", type=str, default="shallow_unet",
                        choices=["unet", "shallow_unet", "nn", "conv2d"]
                        )

    parser.add_argument('--dropout', type=float, default=0)

    parser.add_argument('--batch-normalization', action="store_true")
    parser.add_argument('--filters', nargs="+", type=int, default=None)
    parser.add_argument('--kernels', nargs="+", type=int, default=None)
    parser.add_argument('--max-pool', nargs="+", type=int, default=None)

    args = parser.parse_args()

    # Call the apt function
    img, mask = LoadData(path1, path2, limit=args.limit)

    # View an example of image and corresponding mask
    show_images = 1
    for i in range(show_images):
        img_view = imageio.imread(path1 + img[i])
        mask_view = imageio.imread(path2 + mask[i])
        print(f"#{i} img: {img_view.shape} mask: {mask_view.shape}")
        fig, arr = plt.subplots(1, 2, figsize=(15, 15))
        arr[0].imshow(img_view)
        arr[0].set_title('Image ' + str(i))
        arr[1].imshow(mask_view)
        arr[1].set_title('Masked Image ' + str(i))

    # Define the desired shape
    target_shape_img = [128, 128, 3]
    target_shape_mask = [128, 128, 1]

    # Process data using apt helper function
    X, y = PreprocessData(img, mask, target_shape_img, target_shape_mask, path1, path2)

    # QC the shape of output and classes in output dataset
    print("X Shape:", X.shape)
    print("Y shape:", y.shape)
    # There are 3 classes : background, pet, outline
    print("Classes:", np.unique(y))

    # Visualize the output
    image_index = 0
    visualize_output(X, y, image_index, out_fname="output.png")

    # Use scikit-learn's function to split the dataset
    # Here, I have used 20% data as test/valid set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

    #
    # Net Architecture
    #
    #
    # Call the helper function for defining the layers for the model, given the input image size
    # Note: the output will be [128, 128, n_classes]
    input_size = target_shape_img
    if args.model_name == "shallow_unet":
        print("Using shallow model")
        model = ShallowUNet(input_size=input_size, n_filters=32, n_classes=3)
    elif args.model_name == "unet":
        model = UNet(input_size=input_size, n_filters=32, n_classes=3)
    elif args.model_name == "nn":
        model = NN(input_size=input_size, n_classes=3, dropout=args.dropout)
    elif args.model_name == "conv2d":
        model = Conv(
            input_size=input_size,
            n_classes=3,
            dropout=args.dropout,
            filters=args.filters,
            kernels=args.kernels,
            batch_normalization=args.batch_normalization,
            max_pool=args.max_pool,
        )
    else:
        raise Exception("Model {args.model_name} not implemented")
    # Check the summary to better interpret how the output dimensions change in each layer
    print("Model:\n", model.summary())

    model = train(
        X_train, X_valid, y_train, y_valid,
        model=model,
        epochs=args.epochs
    )
    #
    # to convert to tflite, the model needs to be saved using saved_model.save()
    #
    tf.saved_model.save(model, args.output_dir)
