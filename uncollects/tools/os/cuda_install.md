rpm -i cuda-repo-rhel7-11-4-local-11.4.4_470.82.01-1.x86_64.rpm
yum --disablerepo="*" --enablerepo="cuda-rhel7-11-4-local" install -y cuda-cudart-11-4.x86_64
yum --disablerepo="*" --enablerepo="cuda-rhel7-11-4-local" install -y libcufft-11-4.x86_64
yum --disablerepo="*" --enablerepo="cuda-rhel7-11-4-local" install -y libcurand-11-4.x86_64
yum --disablerepo="*" --enablerepo="cuda-rhel7-11-4-local" install -y libcublas-11-4.x86_64
rpm -i libcudnn8-8.2.4.15-1.cuda11.4.x86_64.rpm
rpm -i nv-tensorrt-repo-rhel7-cuda11.6-trt8.4.3.1-ga-20220813-1-1.x86_64.rpm
yum --disablerepo="*" --enablerepo="nv-tensorrt-rhel7-cuda11.6-trt8.4.3.1-ga-20220813" install -y libnvinfer8.x86_64
yum --disablerepo="*" --enablerepo="nv-tensorrt-rhel7-cuda11.6-trt8.4.3.1-ga-20220813" install -y libnvonnxparsers8.x86_64q:> 

./build.sh --use_tensorrt --use_cuda --cudnn_home /usr/lib64/ --tensorrt_home /usr/lib64 --cuda_home /usr/local/cuda  --build_wheel --skip_tests --skip_submodule_sync  --config Release --cmake_extra_defines '"CMAKE_CUDA_ARCHITECTURES='37;50;52;60;61;70;75;80'"' &> log