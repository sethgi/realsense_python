cmake_minimum_required(VERSION 3.10)
project(rspy LANGUAGES CXX)

find_package(realsense2 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development NumPy)

add_subdirectory(pybind11)

set(PYBIND11_FINDPYTHON ON)
pybind11_add_module(rs_python rs_python.cpp)

target_include_directories(rs_python PRIVATE
  ${realsense2_INCLUDE_DIR}
)

target_link_libraries(rs_python PUBLIC
    realsense2)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_INSTALL_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS rs_python LIBRARY DESTINATION ${PYTHON_INSTALL_DIR})
