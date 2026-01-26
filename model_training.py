import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import get_datasets

def build_model(seq_length=20):
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1)
    ])

    base_cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_cnn.trainable = False 

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

if __name__ == "__main__":
    train_ds, val_ds = get_datasets('Faces_frames')
    model, base_cnn = build_model()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_deepfake_model.h5", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    # Initial Training
    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks)

    # Fine-tuning
    base_cnn.trainable = True
    for layer in base_cnn.layers[:-10]: layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)