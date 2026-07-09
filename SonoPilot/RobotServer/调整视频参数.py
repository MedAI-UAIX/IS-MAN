import cv2
import numpy as np

def find_camera():
    for i in range(100):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"找到摄像头: 索引 {i}")
                return cap
        cap.release()
    raise RuntimeError("未找到可用摄像头")

# 初始化摄像头
cap = find_camera()

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 获取实际分辨率（可能设备不支持1920x1080）
actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"实际分辨率: {int(actual_w)}x{int(actual_h)}")

# 创建窗口
cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

# 创建滑动条
def nothing(x):
    pass

cv2.createTrackbar('Brightness', 'Camera', 50, 100, nothing)  # 0-100，默认50
cv2.createTrackbar('Contrast', 'Camera', 50, 100, nothing)
cv2.createTrackbar('Hue', 'Camera', 50, 100, nothing)
cv2.createTrackbar('Saturation', 'Camera', 50, 100, nothing)

print("按 Q 退出")
print("滑动条范围 0-100，对应设备原始值（可能非线性）")

while True:
    ret, frame = cap.read()
    if not ret:
        print("读取失败")
        break

    # 获取滑动条值
    brightness = cv2.getTrackbarPos('Brightness', 'Camera')
    contrast = cv2.getTrackbarPos('Contrast', 'Camera')
    hue = cv2.getTrackbarPos('Hue', 'Camera')
    saturation = cv2.getTrackbarPos('Saturation', 'Camera')

    # 设置到摄像头（值范围因设备而异，这里用0-100映射）
    # 有些设备是0-255，有些是-64~64，需要根据实际情况调整
    cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    cap.set(cv2.CAP_PROP_HUE, hue)
    cap.set(cv2.CAP_PROP_SATURATION, saturation)

    # 显示当前实际值
    actual_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    actual_contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    actual_hue = cap.get(cv2.CAP_PROP_HUE)
    actual_saturation = cap.get(cv2.CAP_PROP_SATURATION)

    # 在画面上显示信息
    info_text = [
        f"Set: B={brightness} C={contrast} H={hue} S={saturation}",
        f"Actual: B={actual_brightness:.1f} C={actual_contrast:.1f} H={actual_hue:.1f} S={actual_saturation:.1f}",
        f"Resolution: {int(actual_w)}x{int(actual_h)}",
        "Press Q to quit"
    ]
    
    y_offset = 30
    for text in info_text:
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 35

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()