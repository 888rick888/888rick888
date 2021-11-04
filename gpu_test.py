# import tensorflow as tf
# print(tf.test.is_gpu_available())

# import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name())) ##这是安装好GPU版本的
# else:
#     print("Please install GPU version of TF")##未安装GPU版本的tensorflow

import tensorflow as tf
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(logical_gpus)