import os
import cv2
import random
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# Configuration
FACE_DIR = 'Faces_frames'
VIDEO_DIR = 'train_sample_videos'
SEQ_LENGTH = 20
IMG_SIZE = (128, 128)

def process_all_videos(video_dir, output_dir, frames_to_sample=20, img_size=256):
    detector = MTCNN()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_name in os.listdir(video_dir):
        if not video_name.endswith(('.mp4', '.avi', '.mov')): continue
        
        video_path = os.path.join(video_dir, video_name)
        video_output_path = os.path.join(output_dir, os.path.splitext(video_name)[0])
        os.makedirs(video_output_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // frames_to_sample)
        
        frame_count, saved_count = 0, 0
        while cap.isOpened() and saved_count < frames_to_sample:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.detect_faces(frame_rgb)
                if results:
                    best_face = max(results, key=lambda x: x['confidence'])
                    x, y, w, h = best_face['box']
                    x, y = max(0, x), max(0, y)
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size > 0:
                        face_resized = cv2.resize(face_img, (img_size, img_size))
                        cv2.imwrite(os.path.join(video_output_path, f"frame_{saved_count:03d}.jpg"), face_resized)
                        saved_count += 1
            frame_count += 1
        cap.release()

def sequence_generator(video_id_list, face_dir):
    for vid_id in video_id_list:
        folder_path = os.path.join(face_dir, vid_id)
        if not os.path.exists(folder_path): continue
        img_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])[:SEQ_LENGTH]
        if not img_names: continue
        
        frames = []
        for img_name in img_names:
            img = tf.image.decode_jpeg(tf.io.read_file(os.path.join(folder_path, img_name)), channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            frames.append(img)
            
        while len(frames) < SEQ_LENGTH:
            frames.append(tf.zeros((IMG_SIZE[0], IMG_SIZE[1], 3)))
            
        label = np.random.randint(0, 2) # Replace with actual label logic
        yield tf.stack(frames), label

def get_datasets(face_dir):
    all_ids = [d for d in os.listdir(face_dir) if os.path.isdir(os.path.join(face_dir, d))]
    random.seed(42)
    random.shuffle(all_ids)
    split = int(0.8 * len(all_ids))
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(all_ids[:split], face_dir),
        output_signature=(tf.TensorSpec(shape=(20, 128, 128, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
    ).shuffle(10).batch(2).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(all_ids[split:], face_dir),
        output_signature=(tf.TensorSpec(shape=(20, 128, 128, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32))
    ).batch(2).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds