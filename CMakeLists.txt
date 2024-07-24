cmake_minimum_required(VERSION 3.26)
project(blur_test LANGUAGES CXX CUDA)
# find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_C_COMPILER "clang") set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCE main.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")

if(CUDAToolkit_VERSION VERSION_GREATER 10.2)
  add_executable(blur_test_cuda blur.cu)
  target_compile_features(blur_test_cuda PRIVATE cxx_std_17)
  set_property(TARGET blur_test_cuda PROPERTY CUDA_ARCHITECTURES native)
  set_target_properties(
    blur_test_cuda
    PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON RUNTIME_OUTPUT_DIRECTORY
                                              "${CMAKE_SOURCE_DIR}/build")
  add_definitions(-DHAVE_CUDA=1)

  target_compile_options(
    blur_test_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --generate-line-info
                           --use_fast_math --relocatable-device-code=true >)
  target_link_libraries(blur_test_cuda ${OpenCV_LIBS} ${CUDA_LIBS})

else()
  message(STATUS "Insufficient CUDA version. Skipping blur_test.cu")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} ${CUDA_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

add_custom_target(
  profile_multithreading
  COMMAND valgrind --tool=callgrind ./blur_test 5 stressm
  DEPENDS blur_test)
add_custom_target(
  profile_single_thread
  COMMAND valgrind --tool=callgrind ./blur_test 5 stress
  DEPENDS blur_test)
