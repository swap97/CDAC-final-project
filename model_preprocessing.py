import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Initialize the MTCNN detector globally for reuse
detector = MTCNN()

def process_all_videos(video_dir, output_dir, frames_to_sample=20, img_size=256):
    """
    Extracts face frames from videos and saves them to an output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in os.listdir(video_dir):
        if not video_name.endswith(('.mp4', '.avi', '.mov')):
            continue
            
        video_path = os.path.join(video_dir, video_name)
        video_id = os.path.splitext(video_name)[0]
        video_output_path = os.path.join(output_dir, video_id)
        os.makedirs(video_output_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // frames_to_sample)
        
        frame_count = 0
        saved_count = 0

        while cap.isOpened() and saved_count < frames_to_sample:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(frame_rgb)

                if results:
                    best_face = max(results, key=lambda x: x['confidence'])
                    x, y, width, height = best_face['box']
                    x, y = max(0, x), max(0, y)
                    
                    face_img = frame[y:y+height, x:x+width]
                    if face_img.size > 0:
                        face_resized = cv2.resize(face_img, (img_size, img_size))
                        save_name = f"frame_{saved_count:03d}.jpg"
                        cv2.imwrite(os.path.join(video_output_path, save_name), face_resized)
                        saved_count += 1
            frame_count += 1
        cap.release()
        print(f"Finished processing {video_name}: Saved {saved_count} frames.")

def sequence_generator(video_id_list, face_dir):
    """
    Generator for sequence data suitable for LSTM processing.
    """
    seq_length = 20
    img_size = (128, 128)
    
    for vid_id in video_id_list:
        folder_path = os.path.join(face_dir, vid_id)
        if not os.path.exists(folder_path): continue
            
        img_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])[:seq_length]
        if not img_names: continue
            
        frames = []
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            frames.append(img)
            
        while len(frames) < seq_length:
            frames.append(tf.zeros((img_size[0], img_size[1], 3)))
            
        label = np.random.randint(0, 2) # Dummy labels
        yield tf.stack(frames), label

def get_datasets(train_ids, val_ids, face_dir, batch_size=2):
    """
    Creates tf.data.Dataset objects for training and validation.
    """
    train_ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(train_ids, face_dir),
        output_signature=(
            tf.TensorSpec(shape=(20, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).shuffle(10).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(val_ids, face_dir),
        output_signature=(
            tf.TensorSpec(shape=(20, 128, 128, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
