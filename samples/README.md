# Samples

## Install

Depends:

```bash
sudo apt install -y libboost-all-dev libjpeg-dev
```

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
