# realsense_python

A simple Python binding for Intel RealSense cameras using pybind11. Use this if the official pip package isn't working for you.

## Requirements

- Intel RealSense SDK
- CMake
- Python 3.x
- C++ compiler

## Installation

```bash
git clone --recursive https://github.com/sethgi/realsense_python.git
cd realsense_python
mkdir build && cd build
cmake ..
make -j
sudo make install # optional, but recommended.
```

The compiled module will be available to import in Python from the build directory.

## Usage

See `demo_gui.py` for an example of how to use the module.

## License

This is a minimal binding implementation. Check Intel's RealSense SDK license for camera usage restrictions.

## Troubleshooting

### No IMU available even though you have a d435i?
Make sure that you are connecting with USB3. The IMU doesn't show up on a USB2 port.
