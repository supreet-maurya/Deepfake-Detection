import cv2
import os

def video_to_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at the specified frame rate.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where frames will be saved.
        frame_rate (int): Number of frames to save per second.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second
    interval = max(1, int(fps / frame_rate))  # Avoid interval=0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frame_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)

        frame_count += 1

    cap.release()
    print(f"Frames saved to {output_dir}")


def process_datasets(input_base_dir, output_base_dir, frame_rate=1):
    """
    Process train and valid datasets to extract frames while maintaining class structure.

    Args:
        input_base_dir (str): Path to the dataset containing 'train' and 'valid'.
        output_base_dir (str): Path to save extracted frames.
        frame_rate (int): Number of frames to save per second.
    """
    for split in ['train', 'valid']:  # Process both train and valid directories
        input_split_dir = os.path.join(input_base_dir, split)
        output_split_dir = os.path.join(output_base_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)

        for class_name in os.listdir(input_split_dir):  # Iterate over 'fake' and 'real'
            class_dir = os.path.join(input_split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            output_class_dir = os.path.join(output_split_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            for video_file in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_file)
                if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    print(f"Processing {video_path}")
                    video_to_frames(video_path, output_class_dir, frame_rate=frame_rate)


if __name__ == "__main__":
    input_video_dir = "./Datasets_prev"  # Dataset with videos
    output_frame_dir = "./Datasets"     # Target directory for extracted frames
    frame_rate = 1                      # Extract 1 frame per second

    process_datasets(input_video_dir, output_frame_dir, frame_rate)
