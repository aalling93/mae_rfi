import os
import tensorflow as tf

def load_gpu(which: int = 0, memory: int = 60000):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which)
    print("loading gpu")
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(gpus)
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)],
        )
    except RuntimeError as e:
        print("\n error '\n")
        print(e)
    strategy = tf.device(f"/GPU:0")

    return strategy