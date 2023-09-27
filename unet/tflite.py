import tensorflow as tf
import pathlib
import os

if __name__ == "__main__":
    output_dir = "output_dir"  # where the tensorflow model is
    
    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
    #
    # see valid optimizations at https://www.tensorflow.org/api_docs/python/tf/lite/Optimize
    #
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    # save
    tflite_model_file = pathlib.Path(
        os.path.join(output_dir, "tflite-regression.tflite")
    )
    nbytes = tflite_model_file.write_bytes(tflite_model) 
    print(f"Saved tflite model at {tflite_model_file}. Size = {nbytes} bytes")
    print(f"with optimizations = {converter.optimizations}")
