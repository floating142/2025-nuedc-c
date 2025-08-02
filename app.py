import cv2
import numpy as np
import imutils
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import ddddocr
import re
import serial

# 设置为True时，GUI启动后调试窗口默认开启
DEBUG = False

# 串口全局变量
ser = None
serial_lock = threading.Lock()

def open_serial():
    global ser
    serial_port = "/dev/ttyAMA0"
    print(f"使用默认串口设备：{serial_port}")

    while True:
        try:
            with serial_lock:
                ser = serial.Serial(serial_port, 115200, timeout=2)
                time.sleep(2)  # 等待设备准备好
                print("串口连接成功")
                return ser
        except Exception as e:
            print(f"串口打开失败：{e}，重试中...")
            time.sleep(2)

def serial_thread_function(calculator, root):
    global ser
    ser = open_serial()
    while True:
        try:
            if ser.in_waiting:
                line = ser.readline()
                data = line.decode('utf-8', 'ignore').strip()
                match = re.search(r'V:([\d.]+)mVI:([\d.]+)mAP:([\d.]+)mW', data)
                if match:
                    V = float(match.group(1))
                    I = float(match.group(2))
                    P = float(match.group(3))
                    
                    calculator.current_V = V
                    calculator.current_I = I
                    calculator.current_P = P
                    if P > calculator.max_P:
                        calculator.max_P = P
                    
                    root.after(0, lambda: app.update_serial_gui_labels(calculator.current_V, calculator.current_I, calculator.current_P, calculator.max_P))
                time.sleep(0.1)
        except (serial.SerialException, OSError) as e:
            print(f"串口读取异常：{e}，尝试重连...")
            try:
                with serial_lock:
                    ser.close()
            except:
                pass
            ser = open_serial()
            time.sleep(1)
        except KeyboardInterrupt:
            print("用户终止")
            break

class DistanceCalculator:
    def __init__(self, knownWidth_v1, knownWidth_v3, knownDistance):
        self.knownWidth_v1 = knownWidth_v1
        self.knownWidth_v3 = knownWidth_v3
        self.knownDistance = knownDistance
        self.focalLength = 923  # 可通过 calibrate 方法动态标定
        # 初始化 ddddocr 实例
        self.ocr = ddddocr.DdddOcr(det=False, ocr=True)
        # V4 模式的已知尺寸
        self.known_square_side = 5.0  # 假设白色正方形的实际边长为5cm
        
        # 串口数据
        self.current_V = 0.0
        self.current_I = 0.0
        self.current_P = 0.0
        self.max_P = 0.0

    def calibrate(self, calibrationImagePath, mode):
        """通过已知距离和宽度图像进行焦距标定"""
        image = cv2.imread(calibrationImagePath)
        if mode == 1:
            _, marker = self.find_marker_v1(image)
            if marker:
                perWidth = marker[1][0] + marker[1][1]
                self.focalLength = (perWidth * self.knownDistance) / self.knownWidth_v1
        elif mode == 3:
            _, _, height = self.find_marker_v3(image)
            if height:
                self.focalLength = (height * self.knownDistance) / self.knownWidth_v3
    
    def distance_to_camera_v1(self, perWidth):
        if perWidth == 0 or self.focalLength == 0:
            return None
        return (self.knownWidth_v1 * self.focalLength) / perWidth

    def distance_to_camera_v3(self, height_pixels):
        if height_pixels == 0 or self.focalLength == 0:
            return None
        return (self.knownWidth_v3 * self.focalLength) / height_pixels
    
    def find_marker_v1(self, image):
        """V1/V2 模式下寻找主矩形标记（A4，竖直摆放）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        inverted = cv2.bitwise_not(blurred)
        edged = cv2.Canny(inverted, 35, 125)
        cnts = imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
        
        aspect_target = 29.7 / 21.0
        aspect_tol = 1
        h_img, w_img = image.shape[:2]
        img_center = (w_img // 2, h_img // 2)

        candidates = []
        for c in cnts:
            if cv2.contourArea(c) < 5000: continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4: continue
            rect = cv2.minAreaRect(c)
            (x, y), (w, h), angle = rect
            if w == 0 or h == 0: continue
            
            if h > w: # 长边是高度
                if not (0 <= angle <= 10): continue
            else: # 长边是宽度
                if not (80 <= angle < 90): continue

            aspect = max(w, h) / min(w, h)
            if abs(aspect - aspect_target) > aspect_tol: continue
            if aspect < 1.2: continue

            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            if abs(cx - img_center[0]) < w_img * 0.05 and abs(cy - img_center[1]) < h_img * 0.1:
                candidates.append((c, rect))

        if not candidates: return None, None
        candidates.sort(key=lambda x: cv2.contourArea(x[0]))
        return candidates[0]
    
    def process_image_v1(self, image):
        cnt, marker = self.find_marker_v1(image)
        if not marker: return image, None, None, None
        perWidth = marker[1][0] + marker[1][1]
        distance = self.distance_to_camera_v1(perWidth)
        if not distance: return image, None, None, None
        box = cv2.boxPoints(marker).astype(int)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        shape_info = self.detect_shapes_inside_rectangle_v1(image, cnt, distance)
        if shape_info:
            return image, distance, shape_info[0], shape_info[1]
        else:
            return image, distance, None, None

    def detect_shapes_inside_rectangle_v1(self, image, outer_contour, distance):
        x, y, w, h = cv2.boundingRect(outer_contour)
        roi = image[y:y+h, x:x+w]
        if roi.size == 0: return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
        cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE))
        
        for c in cnts:
            if cv2.contourArea(c) < 100: continue
            approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
            shape, size_cm = self.classify_shape_v1(approx, contour=c, distance=distance)
            if shape:
                translated = c + np.array([x, y])
                cv2.drawContours(image, [translated], -1, (255, 0, 0), 2)
                return (shape, size_cm)
        return None

    def classify_shape_v1(self, approx, contour, distance):
        if len(approx) == 3:
            shape = "Triangle"
            x, y, w, h = cv2.boundingRect(approx)
            pixel = max(w, h)
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.9 <= ar <= 1.1:
                shape = "Square"
            else:
                shape = "Rectangle"
                return None, 0
            pixel = max(w, h)
        else:
            area = cv2.contourArea(contour)
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if area / (np.pi * radius**2) > 0.8:
                shape = "Circle"
                pixel = radius * 2
            else:
                return None, 0
        real_size = (pixel * distance) / self.focalLength
        return shape, real_size

    def process_image_v2(self, image):
        cnt, marker = self.find_marker_v1(image)
        if not marker: return image, None, None, None
        perWidth = marker[1][0] + marker[1][1]
        distance = self.distance_to_camera_v1(perWidth)
        if not distance: return image, None, None, None
        box = cv2.boxPoints(marker).astype(int)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        side_length_info = self.detect_smallest_square_v2(image, cnt, distance)
        return image, distance, side_length_info[0] if side_length_info else None, side_length_info[1] if side_length_info else None

    def detect_smallest_square_v2(self, image, outer_contour, distance):
        x_offset, y_offset, w, h = cv2.boundingRect(outer_contour)
        roi = image[y_offset:y_offset+h, x_offset:x_offset+w]
        if roi.size == 0: return None
        roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        translated_contour = outer_contour - np.array([x_offset, y_offset])
        cv2.drawContours(roi_mask, [translated_contour], -1, 255, -1)
        src = cv2.bitwise_and(roi, roi, mask=roi_mask)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 200, 250)
        kernel = np.ones((3, 3), np.uint8)
        edged_closed = cv2.dilate(edged, kernel, iterations=1) 
        cnts = imutils.grab_contours(cv2.findContours(edged_closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        mask = np.zeros_like(gray)
        for c in cnts:
            if cv2.contourArea(c) < 100: continue
            cv2.drawContours(mask, [c], -1, 255, -1)
        binary = mask
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dilated = cv2.dilate(dist, kernel)
        local_max = (dist == dilated)
        local_max = np.logical_and(local_max, dist > 10)
        peaks = [(y, x, dist[y, x]) for y, x in np.column_stack(np.where(local_max))]
        peaks.sort(key=lambda item: item[2], reverse=True)
        final_peaks = []
        suppressed = np.zeros_like(binary, dtype=np.uint8)
        smallest_size = float('inf')

        for y, x, val in peaks:
            if suppressed[y, x] == 255: continue
            if val < 10: continue
            if (val * 2 * distance) / self.focalLength < 5: continue
            
            half_side_pixel = dist[y, x]
            side_length_pixel = max(int(half_side_pixel * 2) - 4, 0)
            real_size_cm = (side_length_pixel * distance) / self.focalLength
            
            if real_size_cm < smallest_size: smallest_size = real_size_cm
            
            final_peaks.append((y, x))
            suppression_radius = int(val)
            cv2.circle(suppressed, (x, y), suppression_radius, 255, -1)
            
            original_x = x + x_offset
            original_y = y + y_offset
            cv2.circle(image, (original_x, original_y), 5, (0, 255, 0), -1)
            text = f"{real_size_cm:.2f}cm"
            cv2.putText(image, text, (original_x - 20, original_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if smallest_size != float('inf'):
            return ("Square", smallest_size)
        else:
            return None

    def find_marker_v3(self, image):
        """V3 模式下寻找主矩形标记（A4，可旋转）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(cv2.bitwise_not(blurred), 35, 125)
        cnts = imutils.grab_contours(cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE))
        h_img, w_img = image.shape[:2]
        img_center = (w_img // 2, h_img // 2)
        candidates = []

        for c in cnts:
            if cv2.contourArea(c) < 5000: continue
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) != 4: continue
            box = self.order_points(approx.reshape(4, 2))
            (tl, tr, br, bl) = box
            w = (np.linalg.norm(br - bl) + np.linalg.norm(tr - tl)) / 2
            h = (np.linalg.norm(tr - br) + np.linalg.norm(tl - bl)) / 2
            aspect = max(w, h) / min(w, h)
            if aspect < 1.2: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            if abs(cx - img_center[0]) < w_img * 0.05 and abs(cy - img_center[1]) < h_img * 0.05:
                candidates.append((c, box, h))

        if not candidates: return None, None, None
        candidates.sort(key=lambda x: cv2.contourArea(x[0]))
        return candidates[0]
    
    def process_image_v3(self, image):
        cnt, box, height = self.find_marker_v3(image)
        output_image = image.copy()
        if box is None: return output_image, None, None, None, None
        cv2.drawContours(output_image, [np.intp(box)], -1, (0, 255, 0), 2)
        distance = self.distance_to_camera_v3(height)
        if distance: cv2.putText(output_image, f"D: {distance:.2f}cm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        warped_image = self.perspective_transform(image, box, height)
        processed_warped, shape, size = self.detect_shapes_inside_rectangle_v3(warped_image, cnt, distance, is_warped=True)
        return output_image, distance, shape, size, processed_warped

    def detect_shapes_inside_rectangle_v3(self, image, outer_contour, distance, is_warped=False):
        if is_warped:
            src, x_offset, y_offset = image.copy(), 0, 0
        else:
            x_offset, y_offset, w, h = cv2.boundingRect(outer_contour)
            src = image[y_offset:y_offset + h, x_offset:x_offset + w]
        if src.size == 0: return image, None, None
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 200, 250)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
        cnts = imutils.grab_contours(cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        mask = np.zeros_like(gray)
        for c in cnts:
            if cv2.contourArea(c) >= 100: cv2.drawContours(mask, [c], -1, 255, -1)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        local_max = np.logical_and(dist == cv2.dilate(dist, np.ones((3, 3), np.uint8)), dist > 10)
        peaks = [(y, x, dist[y, x]) for y, x in np.column_stack(np.where(local_max))]
        peaks.sort(key=lambda item: item[2], reverse=True)
        suppressed = np.zeros_like(mask, dtype=np.uint8)
        smallest_size = float('inf')
        final_peaks = []
        for y, x, val in peaks:
            if suppressed[y, x] == 255: continue
            if (val * 2 * distance) / self.focalLength < 5: continue
            half_side_pixel = dist[y, x]
            side_length_pixel = max(int(half_side_pixel * 2) - 4, 0)
            real_size_cm = (side_length_pixel * distance) / self.focalLength
            if real_size_cm < smallest_size: smallest_size = real_size_cm
            final_peaks.append((y, x))
            cv2.circle(suppressed, (x, y), int(val * 0.95), 255, -1)
        for y, x in final_peaks:
            half_side_pixel = dist[y, x]
            side_length_pixel = max(int(half_side_pixel * 2) - 4, 0)
            real_size_cm = (side_length_pixel * distance) / self.focalLength
            cx, cy = x + x_offset, y + y_offset
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(image, f"{real_size_cm:.2f}cm", (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if smallest_size != float('inf'):
            return image, "Square", smallest_size
        else:
            return image, None, None

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect
        
    def perspective_transform(self, image, box, height):
        rect = self.order_points(box)
        target_height = int(height)
        target_width = int(target_height * (21.0 / 29.7))
        dst = np.array([[0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (target_width, target_height))

    def process_image_v4(self, image, target_number):
        cnt, marker = self.find_marker_v1(image)
        if not marker: return image, None, None, None, False
        
        perWidth = marker[1][0] + marker[1][1]
        distance = self.distance_to_camera_v1(perWidth)
        
        output_image = image.copy()
        box = cv2.boxPoints(marker).astype(int)
        cv2.drawContours(output_image, [box], -1, (0, 255, 0), 2)
        
        found_number, side_length, processed_image = self.detect_shapes_inside_rectangle_v4(output_image, cnt, distance, target_number, is_warped=False)
        
        if found_number is not None:
            cv2.putText(processed_image, f"D: {distance:.2f}cm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            return processed_image, distance, side_length, "Square", True
        
        return processed_image, distance, None, None, False

    def detect_shapes_inside_rectangle_v4(self, image, outer_contour, distance, target_number, is_warped=False):
        if is_warped:
            src, x_offset, y_offset = image.copy(), 0, 0
        else:
            x_offset, y_offset, w, h = cv2.boundingRect(outer_contour)
            src = image[y_offset:y_offset + h, x_offset:x_offset + w]

        if src.size == 0:
            return None, None, image

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 200, 250)
        edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mask = np.zeros_like(gray)
        for c in cnts:
            if cv2.contourArea(c) >= 100:
                cv2.drawContours(mask, [c], -1, 255, -1)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        local_max = np.logical_and(dist == cv2.dilate(dist, np.ones((3, 3), np.uint8)), dist > 10)
        peaks = [(y, x, dist[y, x]) for y, x in np.column_stack(np.where(local_max))]
        peaks.sort(key=lambda item: item[2], reverse=True)

        suppressed = np.zeros_like(mask, dtype=np.uint8)
        final_peaks = []
        found_target_square_side = None

        for y, x, val in peaks:
            if suppressed[y, x] == 255:
                continue
            
            if (val * 2 * distance) / self.focalLength < 5:
                continue
            
            final_peaks.append((y, x))
            cv2.circle(suppressed, (x, y), int(val * 0.95), 255, -1)
        
        for idx, (y, x) in enumerate(final_peaks, 1):
            half_side_pixel = dist[y, x]
            side_length_pixel = max(int(half_side_pixel * 2) - 4, 0)
            
            cx, cy = x + x_offset, y + y_offset
            
            crop_size = 30
            half_crop = crop_size // 2
            x1 = max(cx - half_crop, 0)
            y1 = max(cy - half_crop, 0)
            x2 = min(cx + half_crop, image.shape[1])
            y2 = min(cy + half_crop, image.shape[0])
            crop = image[y1:y2, x1:x2]

            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                crop = cv2.copyMakeBorder(
                    crop,
                    top=max(0, half_crop - cy),
                    bottom=max(0, (cy + half_crop) - image.shape[0]),
                    left=max(0, half_crop - cx),
                    right=max(0, (cx + half_crop) - image.shape[1]),
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)
                )
            try:
                ocr_result = self.ocr.classification(crop)
                cleaned_result = re.sub(r'[^0-9]', '', ocr_result)
                if cleaned_result and int(cleaned_result) == target_number:
                    real_size_cm = (side_length_pixel * distance) / self.focalLength
                    cv2.putText(image, str(target_number), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.putText(image, f"{real_size_cm:.2f}cm", (cx - 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    return target_number, real_size_cm, image
            except Exception as e:
                pass
        
        return None, None, image
    
class App:
    def __init__(self, root, calculator):
        self.root = root
        self.root.title("距离与边长测量")
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.geometry("%dx%d+0+0" % (w, h)) # 启动时最大化窗口
        self.calculator = calculator
        self.cap = cv2.VideoCapture(0)
        
        self.is_debug_on = DEBUG
        self.debug_window = None
        self.debug_label = None
        self.mode = 1
        self.target_ocr_number = -1

        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 14), background='#f0f0f0')
        
        self.create_widgets()
        
        if self.is_debug_on:
            self.show_debug_window()
        
        self.start_serial_thread()

    def create_widgets(self):
        # 使用 grid 布局作为主容器
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        # -------------------- 左侧：主功能控制区 --------------------
        self.control_frame = ttk.Frame(self.main_container, padding="20")
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.label_title = ttk.Label(self.control_frame, text="距离与边长测量", font=("Helvetica", 20, "bold"))
        self.label_title.pack(pady=(0, 20))

        button_frame = ttk.Frame(self.control_frame)
        button_frame.pack(pady=(0, 20))
        self.measure_button_v1 = ttk.Button(button_frame, text="测量模式 V1", command=self.measure_v1)
        self.measure_button_v1.pack(side=tk.LEFT, padx=5)
        self.measure_button_v2 = ttk.Button(button_frame, text="测量模式 V2", command=self.measure_v2)
        self.measure_button_v2.pack(side=tk.LEFT, padx=5)
        self.measure_button_v3 = ttk.Button(button_frame, text="测量模式 V3 (旋转)", command=self.measure_v3)
        self.measure_button_v3.pack(side=tk.LEFT, padx=5)
        self.debug_button = ttk.Button(button_frame, text="显示调试", command=self.toggle_debug)
        self.debug_button.pack(side=tk.LEFT, padx=10)

        ocr_frame = ttk.Frame(self.control_frame)
        ocr_frame.pack(pady=10)
        ttk.Label(ocr_frame, text="OCR目标数字:", font=("Helvetica", 12)).pack(side=tk.LEFT)
        self.ocr_number_label = ttk.Label(ocr_frame, text="--", font=("Helvetica", 16, "bold"), foreground="blue")
        self.ocr_number_label.pack(side=tk.LEFT, padx=10)
        self.measure_button_v4 = ttk.Button(ocr_frame, text="测量模式 V4 (OCR)", command=self.measure_v4)
        self.measure_button_v4.pack(side=tk.LEFT, padx=5)

        keypad_frame = ttk.Frame(self.control_frame)
        keypad_frame.pack(pady=10)
        for i in range(10):
            ttk.Button(keypad_frame, text=str(i), command=lambda i=i: self.set_ocr_number(i)).grid(row=i // 5, column=i % 5, padx=2, pady=2)
        
        self.label_distance = ttk.Label(self.control_frame, text="距离 D: --")
        self.label_distance.pack(pady=10)

        self.label_side_length = ttk.Label(self.control_frame, text="边长 x: --")
        self.label_side_length.pack(pady=10)

        self.label_status = ttk.Label(self.control_frame, text="请点击 '测量模式' 选择模式", font=("Helvetica", 10), foreground="gray")
        self.label_status.pack(pady=(20, 0))

        # -------------------- 右侧：串口数据显示区 --------------------
        self.serial_frame = ttk.Frame(self.main_container, padding="20")
        self.serial_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.label_serial_title = ttk.Label(self.serial_frame, text="串口数据", font=("Helvetica", 16, "bold"))
        self.label_serial_title.pack()
        
        self.label_voltage = ttk.Label(self.serial_frame, text="电压 V: --")
        self.label_voltage.pack(pady=5)

        self.label_current = ttk.Label(self.serial_frame, text="电流 I: --")
        self.label_current.pack(pady=5)
        
        self.label_power = ttk.Label(self.serial_frame, text="功率 P: --")
        self.label_power.pack(pady=5)

        self.label_max_power = ttk.Label(self.serial_frame, text="最大功率 P(max): --", foreground="red")
        self.label_max_power.pack(pady=5)

    def start_serial_thread(self):
        serial_read_thread = threading.Thread(target=serial_thread_function, args=(self.calculator, self.root))
        serial_read_thread.daemon = True
        serial_read_thread.start()

    def update_serial_gui_labels(self, V, I, P, max_P):
        self.label_voltage.config(text=f"电压 V: {V:.2f}mV")
        self.label_current.config(text=f"电流 I: {I:.2f}mA")
        self.label_power.config(text=f"功率 P: {P:.2f}mV")
        self.label_max_power.config(text=f"最大功率 P(max): {max_P:.2f}mV")

    def set_ocr_number(self, number):
        self.target_ocr_number = number
        self.ocr_number_label.config(text=str(number))

    def measure_v1(self):
        self.mode = 1
        self._start_measurement_thread()

    def measure_v2(self):
        self.mode = 2
        self._start_measurement_thread()

    def measure_v3(self):
        self.mode = 3
        self._start_measurement_thread()
    
    def measure_v4(self):
        if self.target_ocr_number == -1:
            self.label_status.config(text="请先选择一个OCR目标数字", foreground="red")
            return
        self.mode = 4
        self._start_measurement_thread()
    
    def _start_measurement_thread(self):
        self.measure_button_v1.config(state=tk.DISABLED)
        self.measure_button_v2.config(state=tk.DISABLED)
        self.measure_button_v3.config(state=tk.DISABLED)
        self.measure_button_v4.config(state=tk.DISABLED)
        self.label_status.config(text="正在测量中...", foreground="blue")
        thread = threading.Thread(target=self._measure_in_thread)
        thread.daemon = True
        thread.start()

    def _measure_in_thread(self):
        MAX_RETRIES = 50
        ERROR_TOLERANCE = 1.0
        MEASUREMENTS_TO_CONFIRM = 3
        
        successful_distances = []
        successful_values = []
        final_type = None

        target_number = self.target_ocr_number

        retries = 0
        while retries < MAX_RETRIES:
            ret, frame = self.cap.read()
            if not ret:
                self.root.after(0, lambda: self.label_status.config(text="错误: 无法获取摄像头画面", foreground="red"))
                self.root.after(0, self._enable_buttons)
                return
            
            processed_frame = frame.copy()
            distance = None
            value = None
            value_type = None
            found_flag = False

            if self.mode == 1:
                processed_frame, distance, value_type, value = self.calculator.process_image_v1(frame)
            elif self.mode == 2:
                processed_frame, distance, value_type, value = self.calculator.process_image_v2(frame)
            elif self.mode == 3:
                processed_frame, distance, value_type, value, _ = self.calculator.process_image_v3(frame)
            elif self.mode == 4:
                processed_frame, distance, value, value_type, found_flag = self.calculator.process_image_v4(frame, target_number)
                if not found_flag:
                    retries += 1
                    time.sleep(0.05)
                    continue

            if distance is not None:
                successful_distances.append(distance)
                if len(successful_distances) >= MEASUREMENTS_TO_CONFIRM and (max(successful_distances) - min(successful_distances)) > ERROR_TOLERANCE:
                    successful_distances = []
            else:
                successful_distances = []
            
            if value is not None:
                successful_values.append(value)
                final_type = value_type
                if len(successful_values) >= MEASUREMENTS_TO_CONFIRM and (max(successful_values) - min(successful_values)) > ERROR_TOLERANCE:
                    successful_values = []
            else:
                successful_values = []

            if len(successful_distances) >= MEASUREMENTS_TO_CONFIRM and len(successful_values) >= MEASUREMENTS_TO_CONFIRM:
                avg_distance = sum(successful_distances) / len(successful_distances)
                avg_value = sum(successful_values) / len(successful_values)
                self.root.after(0, self.update_gui_labels, avg_distance, (final_type, avg_value))
                self.root.after(0, self._enable_buttons)
                return
                
            retries += 1
            time.sleep(0.05)
        
        if self.mode == 4:
            self.root.after(0, lambda: self.label_status.config(text=f"测量超时，未找到数字'{target_number}'", foreground="red"))
        else:
            self.root.after(0, lambda: self.label_status.config(text="测量超时，未获得稳定结果", foreground="red"))
        self.root.after(0, self._enable_buttons)
        self.root.after(0, self.update_gui_labels, None, None)

    def _enable_buttons(self):
        self.measure_button_v1.config(state=tk.NORMAL)
        self.measure_button_v2.config(state=tk.NORMAL)
        self.measure_button_v3.config(state=tk.NORMAL)
        self.measure_button_v4.config(state=tk.NORMAL)

    def toggle_debug(self):
        self.is_debug_on = not self.is_debug_on
        if self.is_debug_on:
            self.show_debug_window()
            self.debug_button.config(text="关闭调试")
        else:
            self.hide_debug_window()
            self.debug_button.config(text="显示调试")

    def show_debug_window(self):
        if self.debug_window is None or not self.debug_window.winfo_exists():
            self.debug_window = tk.Toplevel(self.root)
            self.debug_window.title("调试图像 - 主相机")
            self.debug_window.protocol("WM_DELETE_WINDOW", self.hide_debug_window)
            self.debug_label = tk.Label(self.debug_window)
            self.debug_label.pack()
            self.update_debug_window()

    def hide_debug_window(self):
        if self.debug_window is not None:
            self.debug_window.destroy()
            self.debug_window = None
            self.is_debug_on = False
            self.debug_button.config(text="显示调试")

    def update_gui_labels(self, distance, value_info):
        if distance is not None:
            self.label_distance.config(text=f"距离 D: {distance:.2f} cm")
            self.label_status.config(text="测量成功", foreground="green")
        else:
            self.label_distance.config(text="距离 D: --")
            self.label_side_length.config(text="边长 x: --")

        if value_info is not None:
            value_type, value = value_info
            self.label_side_length.config(text=f"边长 x: {value:.2f} cm ({value_type})")
        else:
            self.label_side_length.config(text="边长 x: --")

    def update_debug_window(self):
        if self.is_debug_on and self.debug_window and self.debug_window.winfo_exists():
            ret, frame = self.cap.read()
            if ret:
                processed_frame = frame.copy()
                if self.mode == 1:
                    processed_frame, _, _, _ = self.calculator.process_image_v1(frame)
                elif self.mode == 2:
                    processed_frame, _, _, _ = self.calculator.process_image_v2(frame)
                elif self.mode == 3:
                    processed_frame, _, _, _, _ = self.calculator.process_image_v3(frame)
                elif self.mode == 4:
                    processed_frame, _, _, _, _ = self.calculator.process_image_v4(frame, -1)

                display_frame = cv2.resize(processed_frame, (640, 480))
                img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.debug_label.imgtk = imgtk
                self.debug_label.config(image=imgtk)
            
            self.debug_window.after(10, self.update_debug_window)

    def on_closing(self):
        self.cap.release()
        try:
            with serial_lock:
                if ser and ser.is_open:
                    ser.close()
        except:
            pass
        self.root.destroy()
        if self.debug_window: self.debug_window.destroy()

if __name__ == "__main__":
    KNOWN_WIDTH_v1 = 42.7  # cm
    KNOWN_WIDTH_v3 = 25.4  # cm
    KNOWN_DISTANCE = 100.0  # cm

    calculator = DistanceCalculator(KNOWN_WIDTH_v1, KNOWN_WIDTH_v3, KNOWN_DISTANCE)

    root = tk.Tk()
    app = App(root, calculator)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()