import os
import sys
import numpy as np
import tensorflow as tf
from dataset import get_data_generators
from model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


def compute_class_weights(train_generator):
    """
    Tính class weights để xử lý mất cân bằng dữ liệu.
    AffectNet có số lượng ảnh không đều giữa các class (sad: 3091, disgust: 1229).
    Nếu không cân bằng, AI sẽ thiên vị dự đoán class có nhiều ảnh hơn.
    """
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    return class_weight_dict


def train():
    # Đường dẫn thư mục dữ liệu
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'archive', 'Train')
    test_dir = os.path.join(base_dir, 'archive', 'Test')
    models_dir = os.path.join(base_dir, 'models')

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        sys.exit(1)

    # Config
    batch_size = 32
    img_size = (96, 96)

    # Lấy data generators
    train_gen, val_gen = get_data_generators(train_dir, test_dir, batch_size=batch_size, img_size=img_size)
    num_classes = train_gen.num_classes

    # Tính class weights để cân bằng dữ liệu
    class_weights = compute_class_weights(train_gen)

    # ==============================
    # XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
    # ==============================
    print("\n" + "="*60)
    print("Training EfficientNetB0 - Full Fine-tuning")
    print("="*60)

    model = build_model(input_shape=(*img_size, 3), num_classes=num_classes)

    # Sử dụng learning rate thấp vì train toàn bộ model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # In tóm tắt
    trainable_count = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    print(f"\nTrainable params: {trainable_count:,}")
    print(f"Non-trainable params: {non_trainable_count:,}")

    # Callbacks
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(models_dir, 'emotion_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )

    print("\nStarting training...")
    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights
    )

    # In kết quả
    best_val_acc = max(history.history['val_accuracy'])
    print("\n" + "="*60)
    print(f"Training finished!")
    print(f"Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"Model saved at: {os.path.join(models_dir, 'emotion_model.h5')}")
    print("="*60)


if __name__ == '__main__':
    train()
