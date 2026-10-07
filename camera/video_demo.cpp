#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;
int main(int argc, char *argv[]) {
    VideoCapture cap;
    bool opened = false;
    if (argc >= 2) {
        string arg = argv[1];
        if (!arg.empty() && arg.find_first_not_of("0123456789") == string::npos) {
            opened = cap.open(stoi(arg));
        } else {
            opened = cap.open(arg);
            double fps = cap.get(CAP_PROP_FPS);
            if (fps <= 0 || fps > 120) fps = 30.0;
            int delay = static_cast<int>(1000.0 / fps);
            Mat frame;
            int key = 0;
            while (cap.read(frame)) {
                putText(frame, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
                imshow("Video", frame);
                key = waitKey(delay); 
                if (key == 27){
                    break;
                }
            }
            cap.release();
            return 0;
        }
    } else {
        opened = cap.open(0);
    }

    if (!opened) {
        cout << "打开失败!用法: ./video_demo [视频文件路径 或 摄像头编号]" << endl;
        return -1;
    }
    int frame_count = 0;
    auto fps_window_start = chrono::steady_clock::now();
    double fps = 0.0;
    Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            cout << "视频结束或取流失败,退出" << endl;
            break;
        }
        ++frame_count;
        auto now = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(now - fps_window_start).count();
        if (elapsed >= 1.0) {
            fps = frame_count / elapsed;
            frame_count = 0;
            fps_window_start = now;
        }
        putText(frame, "FPS: " + to_string(fps), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        imshow("video_demo", frame);
        if (waitKey(1) == 27) {
            break;
        }
    }
    cap.release();
    return 0;
}