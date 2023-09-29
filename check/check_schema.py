import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.lite.python import schema_py_generated as schema_fb


def check_model(model_fname):
    print("Checking:", model_fname)
    with open(model_fname, "rb") as file_handle:
        model_buf = bytearray(file_handle.read())

    tflite_model = schema_fb.Model.GetRootAs(model_buf, 0)
    model = schema_fb.ModelT.InitFromObj(tflite_model)

    print("Version:", model.version)
    print("Description:", model.description.decode("utf-8"))

    # Gets metadata from the model file.
    names = []
    for i in range(tflite_model.MetadataLength()):
        meta = tflite_model.Metadata(i)
        names.append(meta.Name().decode("utf-8"))
        if meta.Name().decode("utf-8") == "min_runtime_version":
            buffer_index = meta.Buffer()
            metadata = tflite_model.Buffers(buffer_index)
            min_runtime_version_bytes = metadata.DataAsNumpy().tobytes()
            print("min_runtime_version:", min_runtime_version_bytes.decode())
    _ = [print("*", x) for x in names]


if __name__ == "__main__":
    tflite_input = "apps/coral_unet/tflite-regression.tflite"
    check_model(tflite_input)

    import glob
    for fname in glob.glob("models/*.tflite"):
        print("\n\n", "*" * 40, "\n\n")
        check_model(fname)
