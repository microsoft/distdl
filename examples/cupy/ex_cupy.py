import cupy as cp

# Create tensor on GPU
x = cp.ones((100, 100))

# By default everything is created on GPU 0
print("Created tensor on device {}".format(x.device))

# Create tensor or GPU 1
with cp.cuda.Device(1):
    x = cp.ones((100, 100))

print("Created tensor on device {}".format(x.device))
