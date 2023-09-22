"""

    uses pre-trained MobileNet, read the keras model,
    convert it to tflite and
    save the converted model to disk

"""

import tensorflow as tf
import pathlib
import os


output_dir = "/tmp/saved_model"

# load pre-trained model
model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# optimizations (quantization)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.optimizations = [tf.lite.Optimize.DEFAULT]


tflite_model = converter.convert()

# save it
tflite_model_file = pathlib.Path(
    os.path.join(output_dir, "tflite-mobilenet-1.tflite")
)
tflite_model_file.write_bytes(tflite_model)
