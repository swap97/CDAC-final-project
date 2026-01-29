import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

def predict_video_with_reliability(video_path, model_path='best_deepfake_model.h5', seq_length=20):
    """
    Predicts if a video is REAL or FAKE and provides reliability metrics:
    - Confidence: How certain the model is.
    - Stability: How consistent the prediction remains under visual noise.
    - Audit Flag: Whether the sample is ambiguous and needs human review.
    """
    # 1. Initialization
    model = tf.keras.models.load_model(model_path)
    detector = MTCNN()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // seq_length)
    
    frames = []
    curr_frame = 0
    
    # Extracting and preprocessing Frames
    while cap.isOpened() and len(frames) < seq_length:
        ret, frame = cap.read()
        if not ret: break
        if curr_frame % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)
            if results:
                # Selecting face with highest confidence
                best_face = max(results, key=lambda x: x['confidence'])
                x, y, w, h = best_face['box']
                # Croping and resizing to 128x128 matching model input
                face = frame[max(0,y):y+h, max(0,x):x+w]
                face = cv2.resize(face, (128, 128))
                face = tf.keras.applications.resnet50.preprocess_input(face)
                frames.append(face)
        curr_frame += 1
    cap.release()

    # Padding for short videos
    while len(frames) < seq_length:
        frames.append(np.zeros((128, 128, 3)))

    # Converting to batch: (1, 20, 128, 128, 3)
    input_tensor = np.expand_dims(np.array(frames), axis=0)

    # Base Prediction
    raw_prediction = model.predict(input_tensor, verbose=0)
    class_idx = np.argmax(raw_prediction)
    confidence = raw_prediction[0][class_idx]
    label = "FAKE" if class_idx == 1 else "REAL"

    # Stability Check (Noise Robustness)
    # Adding 5% Gaussian noise to the frames to see if model flips its decision
    noise = np.random.normal(0, 0.05, input_tensor.shape)
    perturbed_input = np.clip(input_tensor + noise, -1, 1) # Keep in ResNet range
    perturbed_prediction = model.predict(perturbed_input, verbose=0)
    
    # Stability score (1.0 = highly stable, lower = prediction "flickers" with noise)
    stability_score = 1 - np.abs(raw_prediction[0][1] - perturbed_prediction[0][1])

    # Audit Logic (Ambiguity)
    # If the probability is in the 'neutral zone' (40% to 60%), flag for audit
    needs_audit = 0.4 < raw_prediction[0][1] < 0.6

    return {
        "prediction": label,
        "confidence": float(confidence),
        "stability": float(stability_score),
        "needs_human_audit": bool(needs_audit),
        "raw_probs": raw_prediction[0].tolist()
    }
