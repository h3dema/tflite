import os
import tensorflow as tf
import pathlib


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide some warning messages from Tensorflow

    output_dir = "/tmp/saved_model"
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(units=1, input_shape=[1])]
    )
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(x, y, epochs=500)

    tf.saved_model.save(model, output_dir)  # save the trained model

    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
    tflite_model = converter.convert()
    # save
    tflite_model_file = pathlib.Path(
        os.path.join(output_dir, "tflite-regression.tflite")
    )
    tflite_model_file.write_bytes(tflite_model)
