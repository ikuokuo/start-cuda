# Samples

## Install

OpenCV:

```bash
$ vi conf.mk
OPENCV_PATH ?= $(HOME)/opencv-4
```

<!--
dpkg -s libboost-all-dev
dpkg -L libboost-all-dev
whereis boost
-->

Depends:

```bash
# only for samples/img_proc, if enable build in samples/build.sh or samples/CMakeLists.txt
sudo apt install -y libboost-all-dev libjpeg-dev
```

### Build

```bash
./build.sh
```

Or, `make` in root directory (using cmake).

### Clean

```bash
./clean.sh
```

## References

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- CUDA API References
  - [NPP](https://docs.nvidia.com/cuda/npp/index.html)
