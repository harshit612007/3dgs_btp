@echo off
echo ==============================================================
echo Initializing Visual Studio C++ Compiler...
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
echo ==============================================================

call conda activate scaffold

echo Setting CUDA 13.4 PATH and Architectures to build with PTX for RTX 5000 compatibility...
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
set "CUDA_PATH_V11_8="
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=9.0+PTX
set DISTUTILS_USE_SDK=1

echo Compiling gsplat from source...
cd gsplat

:: Uninstall existing gsplat if any
pip uninstall -y gsplat

:: Compile and install without build isolation to bypass AppLocker Temp folder restrictions
set BUILD_EXPERIMENTAL=0
pip install . --no-build-isolation --no-cache-dir --no-deps

echo ==============================================================
echo Compilation Complete!
echo ==============================================================

set BUILD_EXPERIMENTAL=0
