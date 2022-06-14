from time import time
import cupy as cp
import numpy as np
from cupyx.time import repeat


def my_func_cp(a):
    # return cp.sqrt(cp.sum(a**2, axis=-1))
    return cp.sum(a**2, axis=0)


def my_func_np(a):
    # return cp.sqrt(cp.sum(a**2, axis=-1))
    return np.sum(a**2, axis=0)


REPEAT = 10
cp.cuda.runtime.setDevice(0)

# Create tensor on GPU
# x_gpu = cp.ones((100000))
x_gpu = cp.random.random((256, 256, 256), dtype=cp.float32)
x_host = cp.asnumpy(x_gpu)

# print(dir(x_gpu))
# print(dir(x_host))
# print(dir(cp))

# By default everything is created on GPU 0
print("Created tensor on device {}".format(x_gpu.device))
# print("Created tensor on device {}".format(x_host.device))


# Create tensor or GPU 0
# with cp.cuda.Device(0):
#     x = cp.ones((100, 100))

# print(my_func_np(x_host))
# print(my_func_cp(x_gpu))

print(repeat(my_func_cp, (x_gpu,), n_repeat=REPEAT))

t = time()
for i in range(REPEAT):
    my_func_np(x_host)
elapsed = (time() - t) * 1e6
print("Elapsed time in function my_func_np: %0.2f us" % (elapsed / REPEAT))


print(repeat(x_gpu.sum, (), n_repeat=REPEAT))

t = time()
for i in range(REPEAT):
    x_host.sum()
elapsed = (time() - t) * 1e6
print("Elapsed time in function sum(): %0.2f us" % (elapsed / REPEAT))

