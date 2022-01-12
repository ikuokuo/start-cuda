# Start CUDA

Toolkit:

- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  - [Source Code](http://http.download.nvidia.com/cuda-toolkit/)
- [CUDA Toolkit Documentation](http://docs.nvidia.com/cuda/)
  - [CUDA C Programming Guide](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

GPUs:

- [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)
- [Compute Capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)

Books:

- [Professional CUDA C Programming](http://www.hds.bme.hu/~fhegedus/C++/Professional%20CUDA%20C%20Programming.pdf)
  - [Source Code](https://github.com/deeperlearning/professional-cuda-c-programming)
- [CUDA by Example](http://developer.download.nvidia.com/books/cuda-by-example/cuda-by-example-sample.pdf)

## Build

```bash
sudo apt install -y build-essential cmake git

# depends (only for samples/img_proc)
# sudo apt install -y libboost-all-dev libjpeg-dev

# install OpenCV (required)
#  https://github.com/ikuokuo/start-opencv
export OpenCV_DIR=$HOME/opencv-4/lib/cmake

# install TensorRT (optional)
#  https://github.com/NVIDIA/TensorRT
export TensorRT_ROOT=/usr/local/TensorRT

git clone https://github.com/ikuokuo/start-cuda.git
cd start-cuda/
make
```

## Tutorials

- <s>[books](/books)</s>
- [samples](/samples)
  - [quick_start](/samples/quick_start) - [CUDA 快速入门](docs/samples/quick_start.md)
  - [matmul](/samples/matmul)
  - [npp](/samples/npp)
  - [nvjpeg](/samples/nvjpeg)
  - [tensorrt](/samples/tensorrt)
