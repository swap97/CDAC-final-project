import tensorflow as tf
import numpy as np

def measure_deepfake_reliability(model_path, dataset, num_batches=5):
    """
    Measures performance using Confidence, Stability (Noise Robustness), 
    and Ambiguity flags.
    """
    model = tf.keras.models.load_model(model_path)
    all_metrics = []
    
    # Process a few batches from your validation dataset
    for x_batch, _ in dataset.take(num_batches):
        # BASELINE PREDICTIONS
        # Model outputs softmax probabilities: [prob_real, prob_fake]
        preds = model.predict(x_batch, verbose=0)
        
        # CONFIDENCE SCORE
        # Confidence is the maximum probability assigned to either class
        batch_conf = np.max(preds, axis=1)
        mean_conf = np.mean(batch_conf)
        
        # AMBIGUITY (Human Audit Flag)
        # Predictions between 0.4 and 0.6 are "uncertain"
        # Since it is softmax, we check if prob_fake is near 0.5
        uncertain_mask = (preds[:, 1] > 0.4) & (preds[:, 1] < 0.6)
        uncertain_indices = np.where(uncertain_mask)[0]

        # STABILITY CHECK (The "Jitter Test")
        # Add random noise to the sequence of frames
        noise = np.random.normal(0, 0.05, x_batch.shape) # 5% noise
        perturbed_x = np.clip(x_batch + noise, -1, 1) # Keep in ResNet range
        
        perturbed_preds = model.predict(perturbed_x, verbose=0)
        
        # Stability is how much the model's opinion changed due to noise
        # 1.0 means no change, lower means the model is "flickering"
        stability = 1 - np.mean(np.abs(preds[:, 1] - perturbed_preds[:, 1]))

        all_metrics.append({
            'conf': mean_conf,
            'stability': stability,
            'audit_count': len(uncertain_indices),
            'uncertain_indices': uncertain_indices,
            'data': x_batch.numpy()
        })

    # Summary Reporting
    avg_c = np.mean([m['conf'] for m in all_metrics])
    avg_s = np.mean([m['stability'] for m in all_metrics])
    total_audit = sum([m['audit_count'] for m in all_metrics])

    print("--- Reliability Performance Report ---")
    print(f"Mean Prediction Confidence: {avg_c:.2%}")
    print(f"Noise Stability Score:      {avg_s:.4f} ")
    print(f"Audit Required:            {total_audit} samples are ambiguous.")
    
    return all_metrics
