import os
import cv2

def get_mp4_info(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Can't open file: {file_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    print(f"Doc: {file_path}")
    print(f"  fps: {fps:.2f} FPS")
    print(f"  time: {duration:.2f} 秒")
    print(f"  resolution: {width}x{height}")
    cap.release()

def print_all_mp4_info(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.mp4'):
                file_path = os.path.join(root, file)
                get_mp4_info(file_path)

if __name__ == "__main__":
    folder = "."
    print_all_mp4_info(folder)