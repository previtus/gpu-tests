import tensorflow as tf 

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print(tf.test.gpu_device_name())

print(tf.__version__)



## Troubleshooting?
"""
# check if we have CUDA in path 
echo $LD_LIBRARY_PATH
echo $PATH
# we may have CUDA elsewhere than at the default location
export PATH=/WILD_PLACES/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/WILD_PLACES/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
"""
