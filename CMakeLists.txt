cmake_minimum_required(VERSION 2.6)
project(pro)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")

find_package(CUDA REQUIRED)
# 如果要支持python则设置python路径
set(HAS_PYTHON OFF)
set(PythonRoot "/datav/software/anaconda3")
set(PythonName "python3.9")

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
#set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")

# 如果你的opencv找不到，可以自己指定目录
# set(OpenCV_DIR   "/usr/local/opencv4/share/opencv4")
# set(OpenCV_DIR   "/home/lxw/anaconda3/envs/trtpy/lib/python3.9/site-packages/trtpy/cpp-packages/opencv4.2")
set(PROTOBUF_DIR "/media/ros/A666B94D66B91F4D/ros/project/protobuf_my")
set(CUDA_TOOLKIT_ROOT_DIR "/usa/local/cuda-11.6")
set(CUDNN_DIR "/media/ros/A666B94D66B91F4D/ros/new_deploy/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive")
set(TENSORRT_DIR "/media/ros/A666B94D66B91F4D/ros/new_deploy/TensorRT-8.4.3.1.Linux.x86_64-gnu.cuda-11.6.cudnn8.4/TensorRT-8.4.3.1")
set(OPENBLAS_DIR "/media/ros/A666B94D66B91F4D/ros/learning/OpenBLAS-0.3.17-Source/install/openblas")
# set(CUDA_TOOLKIT_ROOT_DIR     "/data/sxai/lean/cuda-10.2")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn7.6.5.32-cuda10.2")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.0.0.11")

# set(CUDA_TOOLKIT_ROOT_DIR  "/data/sxai/lean/cuda-11.1")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn8.2.2.26")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.2.1.6")

# 因为protobuf，需要用特定版本，所以这里指定路径


find_package(OpenCV REQUIRED)

include_directories(
        src/tensorRT
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/src/application
        ${PROJECT_SOURCE_DIR}/src/tensorRT
        ${PROJECT_SOURCE_DIR}/src/tensorRT/common
        # /media/ros/A666B94D66B91F4D/ros/learning/deploy/full_yolov5/src/tensorRT/onnxplugin/plugins/common
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${PROTOBUF_DIR}/include
        ${TENSORRT_DIR}/include
        ${CUDNN_DIR}/include
        ${OPENBLAS_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
        ${PROTOBUF_DIR}/lib
        ${TENSORRT_DIR}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDNN_DIR}/lib
        # /media/ros/A666B94D66B91F4D/ros/test/lib  # 加了个插件的so
)
link_directories("/usr/local/cuda-11.6/targets/x86_64-linux/lib")
if ("${HAS_PYTHON}" STREQUAL "ON")
    message("Usage Python ${PythonRoot}")
    include_directories(${PythonRoot}/include/${PythonName})
    link_directories(${PythonRoot}/lib)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_PYTHON")
endif ()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
# message(GLOB_RECURSE)
cuda_add_library(plugin_list SHARED ${cuda_srcs})
target_link_libraries(plugin_list nvinfer nvinfer_plugin )
target_link_libraries(plugin_list cuda cublas cudart cudnn)
target_link_libraries(plugin_list protobuf pthread dl)
target_link_libraries(plugin_list ${OpenCV_LIBS})

add_executable(pro ${cpp_srcs})

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(pro nvinfer nvinfer_plugin dl)
target_link_libraries(pro cuda cublas cudart cudnn)
target_link_libraries(pro protobuf pthread plugin_list)  # 加了个插件的so
target_link_libraries(pro ${OpenCV_LIBS} libopenblas.a)

if ("${HAS_PYTHON}" STREQUAL "ON")
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/example-python/pytrt)
    add_library(pytrtc SHARED ${cpp_srcs})
    target_link_libraries(pytrtc nvinfer nvinfer_plugin)
    target_link_libraries(pytrtc cuda cublas cudart cudnn)
    target_link_libraries(pytrtc protobuf pthread plugin_list)
    target_link_libraries(pytrtc ${OpenCV_LIBS})
    target_link_libraries(pytrtc "${PythonName}")
    target_link_libraries(pro "${PythonName}")
endif ()

add_custom_target(
        yolo
        DEPENDS pro
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
        COMMAND ./pro yolo
)


