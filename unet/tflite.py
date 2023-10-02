"""

Known error on coralmicro:
> Didn't find op for builtin opcode 'CONV_2D' version '5'. An older version of this builtin might be supported.
> Are you using an old TFLite binary with a newer model?
>
> Failed to get registration from op code CONV_2D


- https://github.com/tensorflow/tensorflow/issues/43232

"""
import argparse
import pathlib
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    # where the tensorflow model is, and we are going to save the tflite model in the same folder
    parser.add_argument('--datafolder', type=str, default="output_dir")
    args = parser.parse_args()

    # convert to tflite
    converter = tf.lite.TFLiteConverter.from_saved_model(args.datafolder)
    #
    # see valid optimizations at https://www.tensorflow.org/api_docs/python/tf/lite/Optimize
    #
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # ref. https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter
    converter.experimental_new_converter = False
    converter.experimental_new_quantizer = False
    converter.experimental_enable_resource_variables = False
    # converter.allow_custom_ops = False
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()
    # save
    tflite_model_file = pathlib.Path(
        os.path.join(args.datafolder, "tflite-regression.tflite")
    )
    nbytes = tflite_model_file.write_bytes(tflite_model)
    print(f"Saved tflite model at {tflite_model_file}. Size = {nbytes} bytes")
    print(f"with optimizations = {converter.optimizations}")
