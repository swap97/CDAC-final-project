import os
import random
import tensorflow as tf
from model_preprocessing import process_all_videos, get_datasets
from model_training import build_final_model, train_model
from evaluation import measure_deepfake_reliability

# Configuration based on your local paths
VIDEO_DIR = r'C:\Users\admin\Desktop\Deep_learning\train_sample_videos'
FACE_DIR = r'C:\Users\admin\Desktop\Deep_learning\Faces frames'
MODEL_PATH = "best_deepfake_model.h5"

def main():
    # 1. Run Preprocessing: Extract face frames if folder is empty
    if not os.path.exists(FACE_DIR) or not os.listdir(FACE_DIR):
        print("Preprocessing videos...")
        process_all_videos(VIDEO_DIR, FACE_DIR)

    # 2. Prepare Data: Split and Create Generators
    all_video_ids = [d for d in os.listdir(FACE_DIR) if os.path.isdir(os.path.join(FACE_DIR, d))]
    random.seed(42)
    random.shuffle(all_video_ids)
    
    split = int(0.8 * len(all_video_ids))
    train_ds, val_ds = get_datasets(all_video_ids[:split], all_video_ids[split:], FACE_DIR)

    # 3. Training: Build and run the ResNet+LSTM model
    if not os.path.exists(MODEL_PATH):
        print("Starting Model Training...")
        model, base_cnn = build_final_model()
        model = train_model(model, base_cnn, train_ds, val_ds)
    else:
        print("Loading existing model...")
        model = tf.keras.models.load_model(MODEL_PATH)

    # 4. Reliability Evaluation
    print("\nRunning Reliability Performance Check...")
    measure_deepfake_reliability(model, val_ds)

if __name__ == "__main__":
    main()
