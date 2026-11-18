import tensorflow as tf

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')

print("TensorFlow working:", True)
print("Available GPUs:", gpus)
print("GPU detected:", len(gpus) > 0)