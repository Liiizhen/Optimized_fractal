import os
import cv2

src_folder = "/mnt/d/code/nano/fractal_set"
dst_root = src_folder

def extract_frames(video_path, out_dir, num_frames=200):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Can't open video: {video_path}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print(f"Frames of video not enough: {video_path} only{total_frames}frames")
        cap.release()
        return
    os.makedirs(out_dir, exist_ok=True)

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    saved = 0
    for idx, frame_idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Read frame {frame_idx} failed: {video_path}")
            continue
        img_name = f"{idx+1:03d}.jpg"
        cv2.imwrite(os.path.join(out_dir, img_name), frame)
        saved += 1
    cap.release()
    print(f"{video_path} completed {saved} to {out_dir}")

def main():
    for file in os.listdir(src_folder):
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(src_folder, file)
            name, _ = os.path.splitext(file)
            out_dir = os.path.join(dst_root, name)
            extract_frames(video_path, out_dir, num_frames=200)

if __name__ == "__main__":
    main()