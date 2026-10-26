 #!/usr/bin/env python3
import pycuda
import pycuda.driver as cuda
import pycuda.autoinit
import socket
# Assume just one GPU
dev = cuda.Device(0)
attrs = dev.get_attributes()
major = attrs[pycuda._driver.device_attribute.COMPUTE_CAPABILITY_MAJOR]
minor = attrs[pycuda._driver.device_attribute.COMPUTE_CAPABILITY_MINOR]
print(f"{socket.gethostname()} : CC {major}.{minor}")