# Installation for CUDA 8.0

- [CUDA Quick Start Guide](https://developer.nvidia.com/compute/cuda/8.0/Prod/docs/sidebar/CUDA_Quick_Start_Guide-pdf)

## Prerequisites

```bash
xcode-select --install
```

Verify that the toolchain is installed:

```bash
/usr/bin/cc --version
```

## CUDA on macOS

- [Installation Guide for Mac OSX](https://developer.nvidia.com/compute/cuda/8.0/Prod/docs/sidebar/CUDA_Installation_Guide_Mac-pdf)

Set up the required environment variables:

`~/.bash_profile`:

```bash
export CUDA_HOME=/Developer/NVIDIA/CUDA-8.0
export PATH=$CUDA_HOME/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=$CUDA_HOME/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
```

To run CUDA applications in console mode on MacBook Pro with both an integrated GPU and a discrete GPU, use the following settings before dropping to console mode:

1. Uncheck System Preferences > Energy Saver > Automatic Graphic Switch
2. Drag the Computer sleep bar to Never in System Preferences > Energy Saver

## Verification

Driver:

```bash
$ kextstat | grep -i cuda
      381    0 0xffffff7f83982000 0x2000     0x2000     com.nvidia.CUDA (1.1.0) DD792765-CA28-395A-8593-D6837F05C4FF <4 1>
```

Compiler:

```bash
$ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2016 NVIDIA Corporation
    Built on Sun_Oct_30_22:18:43_CDT_2016
    Cuda compilation tools, release 8.0, V8.0.54
```

Samples:

```bash
cd ~/Develop/NVIDIA_CUDA-8.0_Samples/
make
```

```bash
./bin/x86_64/darwin/release/deviceQuery
./bin/x86_64/darwin/release/bandwidthTest
./bin/x86_64/darwin/release/nbody
```

## CUDA on Windows

- [Installation Guide for Microsoft Windows](https://developer.nvidia.com/compute/cuda/8.0/Prod2/docs/sidebar/CUDA_Installation_Guide_Windows-pdf)

```bash
cd C:\ProgramData\NVIDIA Corporation\CUDA Samples\v8.0
explorer .
```

Add `CUDA_PATH_V8_0` environment variable:

```bash
CUDA_PATH_V8_0 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
```

Open `Samples_vs2015.sln` and build smaples.

Right click on the project "1_Utilities/deviceQuery", and select "Set as StartUp Project", then run.

## CUDA on Linux

- [Installation Guide for Linux](https://developer.nvidia.com/compute/cuda/8.0/Prod2/docs/sidebar/CUDA_Installation_Guide_Linux-pdf)
  - Chapter 5. CUDA Cross-Platform Environment

- [Black screen after installing CUDA 8.0rc from NVIDIA and unable to enter tty](http://askubuntu.com/questions/808816/black-screen-after-installing-cuda-8-0rc-from-nvidia-and-unable-to-enter-tty)
  - [Ubuntu 16.04 + Nvidia Driver = Blank screen](http://askubuntu.com/questions/760374/ubuntu-16-04-nvidia-driver-blank-screen)
  - [Ubuntu更新完NVIDIA驱动后，重启电脑进入不了系统，一直处于登录界面](http://blog.csdn.net/autocyz/article/details/51818737)
  - `sudo prime-select intel`

- [Install crossbuild-essential-armhf on amd64](http://askubuntu.com/questions/523226/install-crossbuild-essential-armhf-on-amd64)
  - [apt-get install libssl-dev:arm64 and armhf ...](https://answers.launchpad.net/ubuntu/+question/293624)
  - `sudo dpkg --add-architecture armhf`

## CUDA for Android

- [NVIDIA CUDA for Android](https://docs.nvidia.com/gameworks/index.html#technologies/mobile/cuda_android_main.htm)
  - [OpenCV Tutorial 4: CUDA](https://docs.nvidia.com/gameworks/index.html#technologies/mobile/opencv_tutorial_cuda.htm)
- [NVIDIA CodeWorks for Android](https://developer.nvidia.com/codeworks-android)
  - [Download Center](https://developer.nvidia.com/gameworksdownload#?search=CodeWorks)

Error: on "macOS Sierra"

```bash
Installing Android SDK Base 24.4.1_u1 failed.

Return Code: 2
failed MSpanList_Insert 0x2c9710 0x2cb4bd07083f 0x0
fatal error: MSpanList_Insert
```

- [OpenCV - Introduction to OpenCV](http://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)
  - [Introduction into Android Development](http://docs.opencv.org/master/d9/d3f/tutorial_android_dev_intro.html)
  - [Building OpenCV for Tegra with CUDA](http://docs.opencv.org/master/d6/d15/tutorial_building_tegra_cuda.html)

- [OpenCV for Tegra](https://docs.nvidia.com/gameworks/index.html#technologies/mobile/opencv_main.htm%3FTocPath%3DTechnologies%7CMobile%2520Technologies%7COpenCV%2520for%2520Tegra%7C_____0)
  - [OpenCV on Tegra](https://docs.nvidia.com/gameworks/index.html#technologies/mobile/native_android_opencv.htm%3FTocPath%3DTechnologies%7CMobile%2520Technologies%7CNative%2520Development%2520on%2520NVIDIA%25C2%25A0Android%2520Devices%7C_____4)
  - [Download 2.4.8.2](http://developer.download.nvidia.com/devzone/devcenter/mobile/naw100/001/windows/OpenCV-2.4.8.2-Tegra-sdk-windows.zip)

- [ROS, OpenCV and OpenCV4Tegra on the NVIDIA Jetson TK1](http://www.jetsonhacks.com/2015/06/14/ros-opencv-and-opencv4tegra-on-the-nvidia-jetson-tk1/)
- [opencv4tegra sourcecode. anyone?](https://devtalk.nvidia.com/default/topic/766474/opencv4tegra-sourcecode-anyone-/)

- [JetPack for L4T](https://developer.nvidia.com/embedded/jetpack)
  - JetPack runs on Ubuntu host systems only and can be run without a Jetson Developer Kit.
  - Can get "OpenCV4Tegra 2.4.13"

  - [OpenCV for Tegra](http://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/006/linux-x64/libopencv4tegra-repo_2.4.13-17-g5317135_arm64_l4t-r24.deb)
  - [CUDA Toolkit for L4T](http://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/006/linux-x64/cuda-repo-l4t-8-0-local_8.0.34-1_arm64.deb)
  - [CUDA Toolkit for Ubuntu 14.04](http://developer.download.nvidia.com/devzone/devcenter/mobile/jetpack_l4t/006/linux-x64/cuda-repo-ubuntu1404-8-0-local_8.0.34-1_amd64.deb)
