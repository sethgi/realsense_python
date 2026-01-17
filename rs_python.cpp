#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <librealsense2/rs.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>

namespace py = pybind11;

#define ENABLE_COLOR 1
#define ENABLE_IMU 1

struct IMUData {
    double accel[3] = {0, 0, 0};
    double gyro[3] = {0, 0, 0};
    double accel_timestamp = 0;
    double gyro_timestamp = 0;
};

class RSCam {
public:
    RSCam(bool enable_imu = true) : imu_enabled_(enable_imu) {
        std::cout << "Initializing Camera" << std::endl;
        initialize();
        std::cout << "Done Initializing Camera" << std::endl;
        compute_intrinsics();
    }

    ~RSCam() {
        Stop();
    }

    void Stop() {
        if (imu_running_) {
            imu_running_ = false;
            if (imu_thread_.joinable()) {
                imu_thread_.join();
            }
            if (imu_pipe_started_) {
                imu_pipe_.stop();
                imu_pipe_started_ = false;
            }
        }
        if (pipe_started_) {
            pipe.stop();
            pipe_started_ = false;
        }
    }

    std::tuple<py::array_t<uint8_t>, py::array_t<uint16_t>> GetRGBD() {
        rs2::frameset frameset = pipe.wait_for_frames();
        frameset = align_to_color.process(frameset);

        rs2::frame depth = frameset.get_depth_frame();
        int depth_w = depth.as<rs2::video_frame>().get_width();
        int depth_h = depth.as<rs2::video_frame>().get_height();
        py::array_t<uint16_t> depth_array({depth_h, depth_w}, (uint16_t*)depth.get_data());

#if ENABLE_COLOR
        rs2::frame color = frameset.get_color_frame();
        int color_w = color.as<rs2::video_frame>().get_width();
        int color_h = color.as<rs2::video_frame>().get_height();
        py::array_t<uint8_t> color_array({color_h, color_w, 3}, (uint8_t*)color.get_data());
#else
        py::array_t<uint8_t> color_array({depth_h, depth_w, 3});
        color_array[py::make_tuple(py::ellipsis())] = 0;
#endif
        return std::make_tuple(color_array, depth_array);
    }

    py::array_t<uint8_t> GetRGB() {
        auto [color, _] = GetRGBD();
        return color;
    }

    py::array_t<uint16_t> GetDepth() {
        auto [_, depth] = GetRGBD();
        return depth;
    }

    std::vector<std::vector<double>> GetK(bool depth = false) {
#if ENABLE_COLOR
        return depth ? K_depth : K_rgb;
#else
        return K_depth;
#endif
    }

#if ENABLE_IMU
    // Get the latest IMU readings (accelerometer and gyroscope)
    // Returns: dict with 'accel', 'gyro', 'accel_ts', 'gyro_ts'
    py::dict GetIMU() {
        if (!imu_enabled_) {
            throw std::runtime_error("IMU was not enabled during initialization");
        }

        std::lock_guard<std::mutex> lock(imu_mutex_);
        py::dict result;
        
        py::array_t<double> accel(3);
        py::array_t<double> gyro(3);
        auto accel_buf = accel.mutable_unchecked<1>();
        auto gyro_buf = gyro.mutable_unchecked<1>();
        
        for (int i = 0; i < 3; i++) {
            accel_buf(i) = imu_data_.accel[i];
            gyro_buf(i) = imu_data_.gyro[i];
        }
        
        result["accel"] = accel;  // [x, y, z] in m/s^2
        result["gyro"] = gyro;    // [x, y, z] in rad/s
        result["accel_ts"] = imu_data_.accel_timestamp;
        result["gyro_ts"] = imu_data_.gyro_timestamp;
        
        return result;
    }

    // Get just accelerometer data as numpy array [x, y, z] in m/s^2
    py::array_t<double> GetAccel() {
        if (!imu_enabled_) {
            throw std::runtime_error("IMU was not enabled during initialization");
        }

        std::lock_guard<std::mutex> lock(imu_mutex_);
        py::array_t<double> accel(3);
        auto buf = accel.mutable_unchecked<1>();
        for (int i = 0; i < 3; i++) {
            buf(i) = imu_data_.accel[i];
        }
        return accel;
    }

    // Get just gyroscope data as numpy array [x, y, z] in rad/s
    py::array_t<double> GetGyro() {
        if (!imu_enabled_) {
            throw std::runtime_error("IMU was not enabled during initialization");
        }

        std::lock_guard<std::mutex> lock(imu_mutex_);
        py::array_t<double> gyro(3);
        auto buf = gyro.mutable_unchecked<1>();
        for (int i = 0; i < 3; i++) {
            buf(i) = imu_data_.gyro[i];
        }
        return gyro;
    }

    // Get IMU extrinsics (transformation from IMU to depth camera)
    py::dict GetIMUExtrinsics() {
        if (!imu_enabled_) {
            throw std::runtime_error("IMU was not enabled during initialization");
        }

        py::dict result;
        
        // Rotation matrix (3x3)
        py::array_t<double> rotation({3, 3});
        auto rot_buf = rotation.mutable_unchecked<2>();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rot_buf(i, j) = imu_to_depth_extrinsics_.rotation[i * 3 + j];
            }
        }
        
        // Translation vector (3,)
        py::array_t<double> translation(3);
        auto trans_buf = translation.mutable_unchecked<1>();
        for (int i = 0; i < 3; i++) {
            trans_buf(i) = imu_to_depth_extrinsics_.translation[i];
        }
        
        result["rotation"] = rotation;
        result["translation"] = translation;
        
        return result;
    }

    bool IsIMUEnabled() const {
        return imu_enabled_ && imu_running_;
    }
#endif

private:
    rs2::pipeline pipe;
    rs2::align align_to_color{RS2_STREAM_COLOR};
    std::vector<std::vector<double>> K_depth, K_rgb;
    bool pipe_started_ = false;

#if ENABLE_IMU
    bool imu_enabled_ = true;
    rs2::pipeline imu_pipe_;
    bool imu_pipe_started_ = false;
    std::thread imu_thread_;
    std::atomic<bool> imu_running_{false};
    std::mutex imu_mutex_;
    IMUData imu_data_;
    rs2_extrinsics imu_to_depth_extrinsics_;

    void imu_thread_func() {
        while (imu_running_) {
            try {
                rs2::frameset frames = imu_pipe_.wait_for_frames(1000);
                
                for (auto frame : frames) {
                    auto motion = frame.as<rs2::motion_frame>();
                    if (!motion) continue;
                    
                    auto motion_data = motion.get_motion_data();
                    double timestamp = motion.get_timestamp();
                    
                    std::lock_guard<std::mutex> lock(imu_mutex_);
                    
                    if (motion.get_profile().stream_type() == RS2_STREAM_ACCEL) {
                        imu_data_.accel[0] = motion_data.x;
                        imu_data_.accel[1] = motion_data.y;
                        imu_data_.accel[2] = motion_data.z;
                        imu_data_.accel_timestamp = timestamp;
                    } else if (motion.get_profile().stream_type() == RS2_STREAM_GYRO) {
                        imu_data_.gyro[0] = motion_data.x;
                        imu_data_.gyro[1] = motion_data.y;
                        imu_data_.gyro[2] = motion_data.z;
                        imu_data_.gyro_timestamp = timestamp;
                    }
                }
            } catch (const rs2::error& e) {
                // Timeout or other error, continue
                if (imu_running_) {
                    std::cerr << "IMU error: " << e.what() << std::endl;
                }
            }
        }
    }

    bool check_imu_support(rs2::device& dev) {
        // Method 1: Check device name
        std::string name = dev.get_info(RS2_CAMERA_INFO_NAME);
        // D435i, D455, T265, etc. have IMU
        if (name.find("D435i") != std::string::npos ||
            name.find("D455") != std::string::npos ||
            name.find("D457") != std::string::npos ||
            name.find("T26") != std::string::npos ||
            name.find("L515") != std::string::npos) {
            std::cout << "Device " << name << " is known to have IMU" << std::endl;
            return true;
        }
        
        // Method 2: Check for motion module sensor
        for (auto&& sensor : dev.query_sensors()) {
            std::string sensor_name = sensor.get_info(RS2_CAMERA_INFO_NAME);
            std::cout << "  Checking sensor: " << sensor_name << std::endl;
            
            if (sensor_name.find("Motion") != std::string::npos) {
                std::cout << "  Found Motion Module sensor" << std::endl;
                return true;
            }
            
            // Method 3: Check stream profiles
            for (auto&& profile : sensor.get_stream_profiles()) {
                if (profile.stream_type() == RS2_STREAM_ACCEL ||
                    profile.stream_type() == RS2_STREAM_GYRO) {
                    std::cout << "  Found IMU stream profile" << std::endl;
                    return true;
                }
            }
        }
        return false;
    }

    void start_imu_stream() {
        rs2::config imu_cfg;
        imu_cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
        imu_cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
        
        imu_pipe_.start(imu_cfg);
        imu_pipe_started_ = true;

        // Get extrinsics from accel to depth
        auto imu_profile = imu_pipe_.get_active_profile();
        auto depth_profile = pipe.get_active_profile();
        
        auto accel_stream = imu_profile.get_stream(RS2_STREAM_ACCEL);
        auto depth_stream = depth_profile.get_stream(RS2_STREAM_DEPTH);
        
        imu_to_depth_extrinsics_ = accel_stream.get_extrinsics_to(depth_stream);
        
        imu_running_ = true;
        imu_thread_ = std::thread(&RSCam::imu_thread_func, this);
        
        std::cout << "IMU stream started" << std::endl;
    }
#endif

    void initialize() {
        rs2::context ctx;

#if ENABLE_COLOR
        std::cout << "Both color and depth streams are being used" << std::endl;
#else
        std::cout << "Only depth stream is being used" << std::endl;
#endif

        auto devices = ctx.query_devices();
        if (devices.size() == 0) {
            throw std::runtime_error("No valid devices found");
        }

        // Hardware reset first
        for (auto&& dev : devices) {
            std::cout << "Found device: " << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
            dev.hardware_reset();
        }

        std::this_thread::sleep_for(std::chrono::seconds(3));

        // Re-query devices after reset
        ctx = rs2::context();
        devices = ctx.query_devices();
        if (devices.size() == 0) {
            throw std::runtime_error("No valid devices found after reset");
        }

        // Check for IMU support AFTER reset
        bool has_imu = false;
#if ENABLE_IMU
        if (imu_enabled_) {
            for (auto&& dev : devices) {
                has_imu = check_imu_support(dev);
                if (has_imu) {
                    std::cout << "IMU support detected on " << dev.get_info(RS2_CAMERA_INFO_NAME) << std::endl;
                    break;
                }
            }
            if (!has_imu) {
                std::cout << "No IMU support found on any device" << std::endl;
                imu_enabled_ = false;
            }
        }
#endif

        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_DEPTH);
#if ENABLE_COLOR
        cfg.enable_stream(RS2_STREAM_COLOR);
#endif
        pipe.start(cfg);
        pipe_started_ = true;

#if ENABLE_IMU
        if (imu_enabled_ && has_imu) {
            try {
                start_imu_stream();
            } catch (const rs2::error& e) {
                std::cerr << "Failed to start IMU stream: " << e.what() << std::endl;
                imu_enabled_ = false;
            }
        }
#endif
    }

    void compute_intrinsics() {
        auto profile = pipe.get_active_profile();

#if ENABLE_COLOR
        auto color_stream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        auto color_intrinsics = color_stream.get_intrinsics();
        K_rgb = {
            {color_intrinsics.fx, 0, color_intrinsics.ppx},
            {0, color_intrinsics.fy, color_intrinsics.ppy},
            {0, 0, 1}
        };
#endif
        auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
        auto depth_intrinsics = depth_stream.get_intrinsics();
        K_depth = {
            {depth_intrinsics.fx, 0, depth_intrinsics.ppx},
            {0, depth_intrinsics.fy, depth_intrinsics.ppy},
            {0, 0, 1}
        };
    }
};

PYBIND11_MODULE(rs_python, m) {
    m.doc() = "RealSense Python bindings with RGB-D and IMU support";

    py::class_<RSCam>(m, "RSCam")
        .def(py::init<bool>(), py::arg("enable_imu") = true,
             "Initialize RealSense camera. Set enable_imu=False to disable IMU.")
        .def("GetRGBD", &RSCam::GetRGBD,
             "Retrieve aligned RGB and Depth images as NumPy arrays")
        .def("GetRGB", &RSCam::GetRGB,
             "Retrieve RGB image as NumPy array")
        .def("GetDepth", &RSCam::GetDepth,
             "Retrieve Depth image as NumPy array (uint16, in mm)")
        .def("GetK", &RSCam::GetK, py::arg("depth") = false,
             "Retrieve camera intrinsic matrix (3x3). Use depth=True for depth intrinsics.")
#if ENABLE_IMU
        .def("GetIMU", &RSCam::GetIMU,
             "Get latest IMU data as dict with 'accel' (m/s^2), 'gyro' (rad/s), and timestamps")
        .def("GetAccel", &RSCam::GetAccel,
             "Get accelerometer reading as numpy array [x, y, z] in m/s^2")
        .def("GetGyro", &RSCam::GetGyro,
             "Get gyroscope reading as numpy array [x, y, z] in rad/s")
        .def("GetIMUExtrinsics", &RSCam::GetIMUExtrinsics,
             "Get IMU to depth camera extrinsics (rotation matrix and translation)")
        .def("IsIMUEnabled", &RSCam::IsIMUEnabled,
             "Check if IMU is enabled and running")
#endif
        .def("Stop", &RSCam::Stop,
             "Stop all streams and cleanup");
}
