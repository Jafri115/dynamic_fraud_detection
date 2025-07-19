import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configured for inference")
    except RuntimeError as e:
        print("Error configuring GPU:", e)
else:
    print("No GPU available. Inference will run on CPU.")