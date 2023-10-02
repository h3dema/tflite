import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove some tensorflow annoying warnings

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np

if __name__ == "__main__":
    #
    # configuration
    #
    model_path = "../unet/output_dir/tflite-regression.tflite"
    image_file = "../unet/images/original/Abyssinian_1.jpg"

    # create an interpreter to run the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()  # input
    output_details = interpreter.get_output_details()

    # get the size of the input image
    # notice that we are removing the 0-dimension, which contains the number of batches
    #
    size = input_details[0]["shape"][1:3].tolist()
    # load and resize the image to the correct input size
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
    # load interpreter with input data.
    input_shape = input_details[0]['shape']
    input_data = np.array([np.array(image)], dtype=np.float32)  # batch = 1
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # evaluate model with the image
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # notice that output_data contains the probabilities for each of the three classes,
    # thus we need to find the most likely class using tf.argmax()
    pred_mask = tf.argmax(output_data[0], axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 9))
    _ = ax1.imshow(image)
    _ = ax1.axis("off")
  
    _ = ax2.imshow(pred_mask)
    _ = ax2.axis("off")
    plt.savefig("prediction.png")
    plt.close(fig)