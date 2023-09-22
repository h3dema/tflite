"""

    uses pre-trained MobileNet, read the keras model,
    generates a concrete function from the model,
    convert the function to tflite model and
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

# use concrete function
run_model = tf.function(lambda x: model)
concrete_function = run_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
)


# convert to tflite
converter = tf.lite.TFLiteConverter.from_concrete_function([concrete_function])
tflite_model = converter.convert()

# save it
tflite_model_file = pathlib.Path(
    os.path.join(output_dir, "tflite-mobilenet-1.tflite")
)
tflite_model_file.write_bytes(tflite_model)
