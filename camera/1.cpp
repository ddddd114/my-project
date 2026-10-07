#include <sdk/GxIAPI.h>
#include <sdk/DxImageProc.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// ================= 传统视觉算法数据结构与参数 =================
namespace {
    struct LightBar {
        cv::RotatedRect rect;
        cv::Point2f top, bottom;
        float length;
        float width;
        float angle;
        int color; // 0: Red, 1: Blue
    };

    struct Armor {
        cv::Point2f leftTop, rightTop, rightBottom, leftBottom;
        cv::Point2f center;
        int color;
    };

    // 红色灯条HSV阈值 (需根据实际环境调整)
    const cv::Scalar RED_LOWER1(0, 100, 100);
    const cv::Scalar RED_UPPER1(20, 255, 255);
    const cv::Scalar RED_LOWER2(160, 100, 100);
    const cv::Scalar RED_UPPER2(180, 255, 255);

    // 蓝色灯条HSV阈值 (需根据实际环境调整)
    const cv::Scalar BLUE_LOWER(80, 100, 100);
    const cv::Scalar BLUE_UPPER(120, 255, 255);

    // 灯条几何过滤参数
    const float MIN_ASPECT_RATIO = 1.5;
    const float MAX_ASPECT_RATIO = 10.0;
    const float MIN_AREA = 20.0;
    const float MAX_AREA = 5000.0;
    const float MIN_FILL_RATIO = 0.35f; // 轮廓面积/旋转矩形面积
    const float MIN_LIGHT_WIDTH = 1.0f;
    const float MIN_LIGHT_LENGTH_RATIO = 0.7f;
    const float MIN_LIGHT_WIDTH_RATIO = 0.5f;
    const float MIN_CENTER_DISTANCE_RATIO = 0.8f;
    const float MAX_CENTER_DISTANCE_RATIO = 5.0f;
    const float MAX_BAR_AXIS_DIFF = 15.0f * static_cast<float>(CV_PI) / 180.0f;
    const float MAX_PERPENDICULAR_ERROR = 35.0f * static_cast<float>(CV_PI) / 180.0f;

    // 辅助计算函数
    float getDistance(cv::Point2f p1, cv::Point2f p2) {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    // 比较无方向直线的夹角，结果范围为 [0, π/2]。
    // 灯条轴线没有正反方向，因此 0° 和 180° 应视为相同方向。
    float axisAngleDifference(float a, float b) {
        float diff = std::fabs(a - b);
        while (diff >= static_cast<float>(CV_PI)) {
            diff -= static_cast<float>(CV_PI);
        }
        return std::min(diff, static_cast<float>(CV_PI) - diff);
    }

    cv::Point2f barCenter(const LightBar& bar) {
        return cv::Point2f((bar.top.x + bar.bottom.x) / 2.0f, (bar.top.y + bar.bottom.y) / 2.0f);
    }

    // 在已经转换好的 HSV 图像上生成颜色掩膜。
    cv::Mat preprocessHSV(const cv::Mat& hsv, int color, const cv::Mat& kernel) {
        cv::Mat binary;
        if (color == 0) { // 红色
            cv::Mat mask1, mask2;
            cv::inRange(hsv, RED_LOWER1, RED_UPPER1, mask1);
            cv::inRange(hsv, RED_LOWER2, RED_UPPER2, mask2);
            binary = mask1 | mask2;
        } else { // 蓝色
            cv::inRange(hsv, BLUE_LOWER, BLUE_UPPER, binary);
        }
        cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
        return binary;
    }

    // 提取灯条
    std::vector<LightBar> findLightBars(const cv::Mat& binaryImg, int color) {
        std::vector<LightBar> lightBars;
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryImg, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            float area = cv::contourArea(contour);
            if (area < MIN_AREA || area > MAX_AREA) continue;

            cv::RotatedRect rect = cv::minAreaRect(contour);
            float width = rect.size.width;
            float height = rect.size.height;
            float rectArea = width * height;
            if (width < MIN_LIGHT_WIDTH || height < MIN_LIGHT_WIDTH || rectArea <= 0.0f) {
                continue;
            }
            float fillRatio = area / rectArea;
            if (fillRatio < MIN_FILL_RATIO) continue;

            float aspectRatio = std::max(width, height) / std::min(width, height);

            if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) continue;

            LightBar bar;
            bar.rect = rect;
            bar.color = color;
            bar.length = std::max(width, height);
            bar.width = std::min(width, height);
            
            float rawAngle = rect.angle;
            if (width < height) rawAngle += 90.0f;
            bar.angle = rawAngle * CV_PI / 180.0f;

            // 使用旋转矩形长轴的中心线端点，而不是任意两个角点。
            // 这样可以避免灯条倾斜时端点偏移导致装甲板四边变形。
            const cv::Point2f center = rect.center;
            const float halfLength = bar.length * 0.5f;
            const float angle = bar.angle;
            const cv::Point2f direction(std::cos(angle), std::sin(angle));
            cv::Point2f endpoint1 = center - direction * halfLength;
            cv::Point2f endpoint2 = center + direction * halfLength;
            if (endpoint1.y <= endpoint2.y) {
                bar.top = endpoint1;
                bar.bottom = endpoint2;
            } else {
                bar.top = endpoint2;
                bar.bottom = endpoint1;
            }
            lightBars.push_back(bar);
        }
        return lightBars;
    }

    // 匹配灯条，组装装甲板
    std::vector<Armor> matchArmors(const std::vector<LightBar>& lightBars) {
        std::vector<Armor> armors;
        if (lightBars.size() < 2) return armors;

        for (size_t i = 0; i < lightBars.size(); i++) {
            for (size_t j = i + 1; j < lightBars.size(); j++) {
                const LightBar& leftBar = lightBars[i];
                const LightBar& rightBar = lightBars[j];

                if (leftBar.color != rightBar.color) continue;

                float lenRatio = std::min(leftBar.length, rightBar.length) /
                                 std::max(leftBar.length, rightBar.length);
                if (lenRatio < MIN_LIGHT_LENGTH_RATIO) continue;

                float widthRatio = std::min(leftBar.width, rightBar.width) /
                                   std::max(leftBar.width, rightBar.width);
                if (widthRatio < MIN_LIGHT_WIDTH_RATIO) continue;

                if (axisAngleDifference(leftBar.angle, rightBar.angle) >
                    MAX_BAR_AXIS_DIFF) continue;

                cv::Point2f leftCenter = barCenter(leftBar);
                cv::Point2f rightCenter = barCenter(rightBar);
                float centerDistance = getDistance(leftCenter, rightCenter);
                float averageLength = (leftBar.length + rightBar.length) * 0.5f;
                if (averageLength <= 0.0f) continue;
                float centerDistanceRatio = centerDistance / averageLength;
                if (centerDistanceRatio < MIN_CENTER_DISTANCE_RATIO ||
                    centerDistanceRatio > MAX_CENTER_DISTANCE_RATIO) continue;

                // 装甲板局部坐标：以两灯条中心连线为 X 轴，
                // 检查两根灯条是否近似平行且处于连线两侧。
                cv::Point2f connection = rightCenter - leftCenter;
                float connectionNorm = cv::norm(connection);
                if (connectionNorm <= 1e-4f) continue;
                cv::Point2f connectionUnit = connection / connectionNorm;
                float leftAxisAngle = leftBar.angle;
                float rightAxisAngle = rightBar.angle;
                float expectedNormalAngle = std::atan2(
                    connectionUnit.y, connectionUnit.x) +
                    static_cast<float>(CV_PI) * 0.5f;
                if (axisAngleDifference(leftAxisAngle, expectedNormalAngle) >
                        MAX_PERPENDICULAR_ERROR ||
                    axisAngleDifference(rightAxisAngle, expectedNormalAngle) >
                        MAX_PERPENDICULAR_ERROR) {
                    continue;
                }

                float leftProjection = leftCenter.dot(connectionUnit);
                float rightProjection = rightCenter.dot(connectionUnit);
                if (rightProjection <= leftProjection) continue;

                Armor armor;
                armor.color = leftBar.color;
                armor.leftTop = leftBar.top;
                armor.leftBottom = leftBar.bottom;
                armor.rightTop = rightBar.top;
                armor.rightBottom = rightBar.bottom;
                
                armor.center.x = (armor.leftTop.x + armor.rightTop.x +
                                  armor.rightBottom.x + armor.leftBottom.x) * 0.25f;
                armor.center.y = (armor.leftTop.y + armor.rightTop.y +
                                  armor.rightBottom.y + armor.leftBottom.y) * 0.25f;

                armors.push_back(armor);
            }
        }
        return armors;
    }

    // 绘制装甲板函数（封装，避免代码重复）
    void drawArmor(cv::Mat& img, const Armor& armor) {
        // OpenCV 使用 BGR：红色为 (0, 0, 255)，蓝色为 (255, 0, 0)。
        const cv::Scalar frameColor = armor.color == 0
            ? cv::Scalar(0, 0, 255)
            : cv::Scalar(255, 0, 0);

        cv::line(img, armor.leftTop, armor.rightTop, frameColor, 2);
        cv::line(img, armor.rightTop, armor.rightBottom, frameColor, 2);
        cv::line(img, armor.rightBottom, armor.leftBottom, frameColor, 2);
        cv::line(img, armor.leftBottom, armor.leftTop, frameColor, 2);

        cv::circle(img, armor.leftTop, 3, frameColor, -1);
        cv::circle(img, armor.rightTop, 3, frameColor, -1);
        cv::circle(img, armor.rightBottom, 3, frameColor, -1);
        cv::circle(img, armor.leftBottom, 3, frameColor, -1);
        cv::circle(img, armor.center, 4, frameColor, -1);
    }

    // --- 原有大恒相机SDK辅助函数 ---
    string GetErrorString(GX_STATUS emErrorStatus) {
        char *error_info = nullptr;
        size_t size = 0;
        GX_STATUS emStatus = GXGetLastError(&emErrorStatus, nullptr, &size);
        if (emStatus != GX_STATUS_SUCCESS) return "<Error when calling GXGetLastError>";
        error_info = new char[size];
        emStatus = GXGetLastError(&emErrorStatus, error_info, &size);
        string error_string = error_info != nullptr ? error_info : "";
        delete[] error_info;
        return emStatus == GX_STATUS_SUCCESS ? error_string : "<Error when calling GXGetLastError>";
    }
    struct DeviceInfo {
        int index;
        string serial;
        string model;
    };
    vector<DeviceInfo> EnumerateDevices() {
        uint32_t device_num = 0;
        GX_STATUS emStatus = GXUpdateAllDeviceList(&device_num, 1000);
        if (emStatus != GX_STATUS_SUCCESS || device_num == 0) {
            cout << "枚举失败或未找到设备: " << GetErrorString(emStatus) << endl;
            return {};
        }
        vector<DeviceInfo> devices;
        cout << "共找到 " << device_num << " 台设备:" << endl;
        for (uint32_t i = 1; i <= device_num; ++i) {
            GX_DEVICE_INFO info;
            memset(&info, 0, sizeof(GX_DEVICE_INFO));
            emStatus = GXGetDeviceInfo(i, &info);
            if (emStatus != GX_STATUS_SUCCESS) continue;
            if (info.emDevType == GX_DEVICE_CLASS_U3V) {
                auto &u3v = info.DevInfo.stU3VDevInfo;
                cout << "  [" << i << "] 型号: " << reinterpret_cast<const char *>(u3v.chModelName) << "  序列号: " << reinterpret_cast<const char *>(u3v.chSerialNumber) << endl;
                devices.push_back({static_cast<int>(i),
                                   reinterpret_cast<const char *>(u3v.chSerialNumber),
                                   reinterpret_cast<const char *>(u3v.chModelName)});
            }
        }
        return devices;
    }
    int FindDeviceIndexBySerial(const vector<DeviceInfo> &devices, const string &serial) {
        for (size_t i = 0; i < devices.size(); ++i) {
            if (devices[i].serial == serial) return devices[i].index;
        }
        return -1;
    }
    bool SetEnumValueByStringChecked(GX_DEV_HANDLE device,
                                     const char* name,
                                     const char* value) {
        GX_STATUS status = GXSetEnumValueByString(device, name, value);
        if (status != GX_STATUS_SUCCESS) {
            cout << name << "=" << value << " 设置失败: "
                 << GetErrorString(status) << endl;
            return false;
        }
        return true;
    }

    double clampNodeValue(const GX_FLOAT_VALUE& node, double value) {
        value = std::max(node.dMin, std::min(node.dMax, value));
        if (node.dInc > 0.0) {
            value = node.dMin + std::round((value - node.dMin) / node.dInc) * node.dInc;
            value = std::max(node.dMin, std::min(node.dMax, value));
        }
        return value;
    }

    bool AddExposureTime(GX_DEV_HANDLE device, double delta_us) {
        GX_FLOAT_VALUE node; memset(&node, 0, sizeof(GX_FLOAT_VALUE));
        GX_STATUS emStatus = GXGetFloatValue(device, "ExposureTime", &node);
        if (emStatus != GX_STATUS_SUCCESS) return false;
        double value = clampNodeValue(node, node.dCurValue + delta_us);
        emStatus = GXSetFloatValue(device, "ExposureTime", value);
        if (emStatus != GX_STATUS_SUCCESS) return false;
        cout << "曝光时间 -> " << value << " us" << endl;
        return true;
    }
    bool AddGain(GX_DEV_HANDLE device, double delta_db) {
        GX_FLOAT_VALUE node; memset(&node, 0, sizeof(GX_FLOAT_VALUE));
        GX_STATUS emStatus = GXGetFloatValue(device, "Gain", &node);
        if (emStatus != GX_STATUS_SUCCESS) return false;
        double value = clampNodeValue(node, node.dCurValue + delta_db);
        emStatus = GXSetFloatValue(device, "Gain", value);
        if (emStatus != GX_STATUS_SUCCESS) return false;
        cout << "增益 -> " << value << " dB" << endl;
        return true;
    }
    bool AddGamma(GX_DEV_HANDLE device, double delta) {
        GX_FLOAT_VALUE node; memset(&node, 0, sizeof(GX_FLOAT_VALUE));
        GX_STATUS emStatus = GXGetFloatValue(device, "Gamma", &node);
        if (emStatus != GX_STATUS_SUCCESS) return false;
        double value = clampNodeValue(node, node.dCurValue + delta);
        emStatus = GXSetFloatValue(device, "Gamma", value);
        if (emStatus != GX_STATUS_SUCCESS) return false;
        cout << "伽马 -> " << value << endl;
        return true;
    }
}

// ================= 主函数 =================
int main(int argc, char *argv[]) {
    GX_STATUS emStatus = GXInitLib();
    if (emStatus != GX_STATUS_SUCCESS) {
        cout << "GXInitLib 失败: " << GetErrorString(emStatus) << endl;
        return -1;
    }
    auto devices = EnumerateDevices();
    if (devices.empty()) {
        GXCloseLib();
        return -1;
    }
    if (argc < 2) {
        cout << "用法: ./daheng_demo <相机序列号>" << endl;
        GXCloseLib();
        return 0;
    }
    const string serial = argv[1];
    int device_index = FindDeviceIndexBySerial(devices, serial);
    if (device_index < 0) {
        cout << "序列号 " << serial << " 不在枚举列表中!" << endl;
        GXCloseLib();
        return -1;
    }
    GX_DEV_HANDLE device = nullptr;
    emStatus = GXOpenDeviceByIndex(device_index, &device);
    if (emStatus != GX_STATUS_SUCCESS) {
        cout << "GXOpenDeviceByIndex: " << GetErrorString(emStatus) << endl;
        GXCloseLib();
        return -1;
    }
    cout << "打开相机 " << serial << " 成功" << endl;
    SetEnumValueByStringChecked(device, "ExposureAuto", "Off");
    SetEnumValueByStringChecked(device, "GainAuto", "Off");
    SetEnumValueByStringChecked(device, "BalanceWhiteAuto", "Continuous");
    SetEnumValueByStringChecked(device, "AcquisitionMode", "Continuous");
    SetEnumValueByStringChecked(device, "TriggerMode", "Off");
    
    emStatus = GXSetEnumValue(device, "PixelFormat", GX_PIXEL_FORMAT_BAYER_RG8);
    if (emStatus != GX_STATUS_SUCCESS) {
        cout << "PixelFormat: " << GetErrorString(emStatus) << endl;
    }
    GX_INT_VALUE limit_node; memset(&limit_node, 0, sizeof(GX_INT_VALUE));
    if (GXGetIntValue(device, "DeviceLinkThroughputLimit", &limit_node) == GX_STATUS_SUCCESS) {
        GXSetIntValue(device, "DeviceLinkThroughputLimit", limit_node.nMax);
    }
    GX_INT_VALUE width_node, height_node;
    memset(&width_node, 0, sizeof(GX_INT_VALUE)); memset(&height_node, 0, sizeof(GX_INT_VALUE));
    GXGetIntValue(device, "Width", &width_node); GXGetIntValue(device, "Height", &height_node);
    int width = static_cast<int>(width_node.nCurValue);
    int height = static_cast<int>(height_node.nCurValue);
    cout << "分辨率 " << width << "x" << height << endl;
    
    uint32_t stream_num = 0;
    GX_DS_HANDLE stream_handle = nullptr;
    uint32_t payload_size = 0;
    if (GXGetDataStreamNumFromDev(device, &stream_num) != GX_STATUS_SUCCESS || stream_num < 1) {
        cout << "获取数据流失败" << endl;
        GXCloseDevice(device); GXCloseLib(); return -1;
    }
    if (GXGetDataStreamHandleFromDev(device, 1, &stream_handle) != GX_STATUS_SUCCESS ||
        stream_handle == nullptr) {
        cout << "获取数据流句柄失败" << endl;
        GXCloseDevice(device); GXCloseLib(); return -1;
    }
    if (GXGetPayLoadSize(stream_handle, &payload_size) != GX_STATUS_SUCCESS) {
        cout << "获取 Payload 大小失败" << endl;
        GXCloseDevice(device); GXCloseLib(); return -1;
    }
    if (GXSetAcqusitionBufferNumber(device, 5) != GX_STATUS_SUCCESS) {
        cout << "设置采集缓冲区数量失败" << endl;
    }
    
    const unsigned int buffer_size = sizeof(unsigned char) * width * height * 3;
    unsigned char *rgb_buffer = static_cast<unsigned char *>(malloc(buffer_size));
    if (rgb_buffer == nullptr) {
        cout << "分配 RGB 图像缓存失败" << endl;
        GXCloseDevice(device);
        GXCloseLib();
        return -1;
    }
    const cv::Mat morph_kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat hsv;
    int64_t color_filter = GX_COLOR_FILTER_NONE;
    GX_ENUM_VALUE filter_value; memset(&filter_value, 0, sizeof(GX_ENUM_VALUE));
    if (GXGetEnumValue(device, "PixelColorFilter", &filter_value) == GX_STATUS_SUCCESS) {
        color_filter = filter_value.stCurValue.nCurValue;
    }
    
    emStatus = GXStreamOn(device);
    if (emStatus != GX_STATUS_SUCCESS) {
        cout << "GXStreamOn: " << GetErrorString(emStatus) << endl;
        free(rgb_buffer); GXCloseDevice(device); GXCloseLib(); return -1;
    }
    
    int frame_count = 0;
    auto fps_window_start = chrono::steady_clock::now();
    double fps = 0.0;
    PGX_FRAME_BUFFER frame_buffer = nullptr;
    
    cout << "取流与装甲板识别开始，按 e/d 调曝光、a/q 调增益、s/w 调伽马、ESC 退出" << endl;
    
    while (true) {
        emStatus = GXDQBuf(device, &frame_buffer, 1000);
        if (emStatus != GX_STATUS_SUCCESS) {
            cout << "GXDQBuf 超时/失败: " << GetErrorString(emStatus) << endl;
            break;
        }
        if (frame_buffer->nStatus != GX_FRAME_STATUS_SUCCESS) {
            GXQBuf(device, frame_buffer); continue;
        }
        
        // 1. 图像格式转换
        VxInt32 dx_status = DxRaw8toRGB24Ex(frame_buffer->pImgBuf, rgb_buffer, 
                                             frame_buffer->nWidth, frame_buffer->nHeight, 
                                             RAW2RGB_NEIGHBOUR, DX_PIXEL_COLOR_FILTER(color_filter), 
                                             false, DX_ORDER_BGR);
        if (dx_status != DX_OK) {
            GXQBuf(device, frame_buffer); continue;
        }
        
        // 2. 包装为OpenCV Mat
        Mat image(frame_buffer->nHeight, frame_buffer->nWidth, CV_8UC3, rgb_buffer);
        Mat display = image.clone(); // 用于绘制的副本
        
        // 3. 传统视觉识别算法
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
        Mat binaryRed = preprocessHSV(hsv, 0, morph_kernel);
        Mat binaryBlue = preprocessHSV(hsv, 1, morph_kernel);
        
        vector<LightBar> redBars = findLightBars(binaryRed, 0);
        vector<LightBar> blueBars = findLightBars(binaryBlue, 1);
        
        vector<Armor> redArmors = matchArmors(redBars);
        vector<Armor> blueArmors = matchArmors(blueBars);
        
        // 4. 绘制识别结果
        for (const auto& armor : redArmors) drawArmor(display, armor);
        for (const auto& armor : blueArmors) drawArmor(display, armor);
        
        // 5. 帧率计算（滑动平均，显示更平滑）
        auto now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(now - fps_window_start).count();
        // 每0.5秒更新一次FPS（也可以用滑动平均）
        if (elapsed >= 0.5) {
            double current_fps = frame_count / elapsed;
            fps = fps * 0.8 + current_fps * 0.2; // 平滑
            frame_count = 0;
            fps_window_start = now;
        }
        ++frame_count;
        
        // 6. 显示FPS和数量
        putText(display, "FPS: " + to_string((int)fps), Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        putText(display, "Red: " + to_string(redArmors.size()) + " Blue: " + to_string(blueArmors.size()), 
                Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
        
        imshow("Daheng Armor Detection", display);
        
        // 7. 按键控制逻辑
        bool quit = false;
        switch (waitKey(1)) {
            case 27: quit = true; break;
            case 'e': AddExposureTime(device, 25.0); break;
            case 'd': AddExposureTime(device, -25.0); break;
            case 'a': AddGain(device, 0.1); break;
            case 'q': AddGain(device, -0.1); break;
            case 's': AddGamma(device, 0.1); break;
            case 'w': AddGamma(device, -0.1); break;
            default: break;
        }
        
        GX_STATUS q_status = GXQBuf(device, frame_buffer);
        if (q_status != GX_STATUS_SUCCESS) {
            cout << "GXQBuf 失败: " << GetErrorString(q_status) << endl;
        }
        if (quit) break;
    }
    
    GXStreamOff(device);
    free(rgb_buffer);
    GXCloseDevice(device);
    GXCloseLib();
    cout << "相机已关闭" << endl;
    return 0;
}