import cupy as cp


def set_device(rank=0):

    # Set device based on rank
    cp.cuda.runtime.setDevice(rank % cp.cuda.runtime.getDeviceCount())

    # Return current device
    return cp.cuda.runtime.getDevice()


def get_device():
    return cp.cuda.runtime.getDevice()
