import tensorflow as tf
print(tf.lite.experimental.Analyzer.analyze(model_path="/home/idlab/coralmicro/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"))

print(tf.lite.experimental.Analyzer.analyze(model_path="/home/idlab/coralmicro/apps/coral_unet/tflite-regression.tflite"))

