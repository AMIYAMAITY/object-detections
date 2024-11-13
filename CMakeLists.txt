cmake_minimum_required(VERSION 2.6)

# Fancy colorized output messages
string(ASCII 27 Esc)
set(CR "${Esc}[m")
set(Red "${Esc}[1;31m")
set(Green "${Esc}[1;32m")
set(Blue "${Esc}[1;34m")

project(Detector-TRT)
message(STATUS "${Green}Compiling project ${PROJECT_NAME}...${CR}")




add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)

#SpdLog
message(STATUS "${Blue}Checking dependency - SpdLog${CR}")
set(SPDPATH "/usr/")
include_directories(${SPDPATH}/include)


#Nlohmann JSON
message(STATUS "${Blue}Checking dependency - Nlohmann JSON${CR}")
set(NJPATH "/opt/NlohmannJSON/")
include_directories(${NJPATH}/include)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV 4.5.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(MONITOR trtDetection)
set(DSOURCES src/app.cpp)
set(TRTPATH src/yolov7.cpp)
set(UTILSPATH src/utils.cpp)

find_package(Crow)


cuda_add_executable(${MONITOR} ${DSOURCES} ${TRTPATH} ${UTILSPATH})

target_link_libraries (${MONITOR} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} stdc++fs pthread nvinfer nvinfer_plugin cudart  nvparsers nvonnxparser Crow::Crow boost_system)
set_target_properties(${MONITOR} ${TRAINER} PROPERTIES COMPILE_FLAGS "-pthread -std=c++11")

set(CMAKE_INSTALL_RPATH ${SMQPATH}/libs ${MQPATH}/libs)
set(CMAKE_INSTALL_PREFIX "../distribute")

# Install
install(TARGETS ${MONITOR} DESTINATION .)
