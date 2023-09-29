#
# Train the segmentation model (UNet)
#
# TODO: select between CPU or GPU
#
import os
import numpy as np  # for using np arrays

# for reading and processing images
import imageio.v2 as imageio
from PIL import Image

# for visualizations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs 

import tensorflow as tf
from unet import UNet, ShallowUNet


def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively

    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        orig_img.append(file)
    for file in mask_dataset:
        mask_img.append(file)

    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    orig_img.sort()
    mask_img.sort()

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
    axis[1].plot(results.history["accuracy"], color='r', label='train accuracy')
    axis[1].plot(results.history["val_accuracy"], color='b', label='dev accuracy')
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
          epochs=20,
          input_size=(128, 128, 3),
          shallow_unet: bool = True  # True to use shallow Unet (3 levels), False (5 levels)
         ):
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    #
    # Net Architecture
    #
    #
    # Call the helper function for defining the layers for the model, given the input image size
    # Note: the output will be [128, 128, n_classes]
    if shallow_unet:
        print("Using shallow model")
        model = ShallowUNet(input_size=input_size, n_filters=32, n_classes=3)
    else:
        model = UNet(input_size=input_size, n_filters=32, n_classes=3)
    # Check the summary to better interpret how the output dimensions change in each layer
    print("Model:\n", model.summary())

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
    index = 700
    VisualizeResults(index, X_valid, y_valid, model)

    return model


if __name__ == "__main__":
    #
    # data
    #
    path1 = 'images/original/'
    path2 = 'images/masks/'
    SHALLOW_UNET = True
    
    # Call the apt function
    img, mask = LoadData(path1, path2)

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

    model = train(X_train, X_valid, y_train, y_valid, 
                  epochs=1,
                  shallow_unet=SHALLOW_UNET)
    #
    # to convert to tflite, the model needs to be saved using saved_model.save()
    #
    tf.saved_model.save(model, "output_dir") 