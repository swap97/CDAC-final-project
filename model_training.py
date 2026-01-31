import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import datetime

def build_final_model(seq_length=20):
    """
    Builds the ResNet50 + LSTM Hybrid model.
    """
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomWidth(0.2)
    ])

    base_cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_cnn.trainable = False # Initial training with frozen base

    model = models.Sequential([
        layers.Input(shape=(seq_length, 128, 128, 3)),
        layers.TimeDistributed(augmentation),
        layers.TimeDistributed(base_cnn),
        layers.LSTM(128, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    
    return model, base_cnn

def train_model(model, base_cnn, train_ds, val_ds, epochs=15):
    """
    Executes a two-phase training: frozen base followed by fine-tuning.
    """
    # Phase 1: Train Top Layers
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint(filepath="best_deepfake_model.h5", monitor='val_loss', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("Starting Phase 1: Training top layers...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, 
              callbacks=[checkpoint, early_stopping, tensorboard_callback])

    # Phase 2: Fine-tuning
    print("Starting Phase 2: Fine-tuning ResNet layers...")
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-10]: # Unfreeze only the last 10 layers because of less confidence in initial training.
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=val_ds, epochs=10)
    
    return model
