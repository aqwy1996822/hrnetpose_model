cmake_minimum_required(VERSION 2.6)
project(hrnetpose_model)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED")

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads)

# setup CUDA
# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/home/leo/LibProjects/TensorRT-7.2.2.3/include/)
link_directories(/home/leo/LibProjects/TensorRT-7.2.2.3//lib/)

find_package(OpenCV 3.4 REQUIRED)
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)

add_executable(test ${PROJECT_SOURCE_DIR}/src/hrnetpose/hrnetpose_model.cpp ${PROJECT_SOURCE_DIR}/src/hrnetpose/calibrator.cpp main.cpp)
#add_executable(test2 ${PROJECT_SOURCE_DIR}/src/hrnetpose/test.cpp)

target_link_libraries(test nvinfer nvinfer_plugin nvonnxparser)
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test ${CUDA_LIBRARIES})

#target_link_libraries(test2 nvinfer nvinfer_plugin nvonnxparser)
#target_link_libraries(test2 ${OpenCV_LIBS})
#target_link_libraries(test2 ${CUDA_LIBRARIES})
