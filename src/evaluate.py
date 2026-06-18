"""
Script đánh giá nhanh model hiện tại trên tập Test.
In ra: accuracy tổng, accuracy từng class, confusion matrix.
"""
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models', 'emotion_model.h5')
test_dir = os.path.join(base_dir, 'archive', 'Test')

CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("Loading model...")
model = tf.keras.models.load_model(model_path)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(96, 96),
    color_mode="rgb",
    batch_size=32,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print("\nEvaluating on test set...")
loss, accuracy = model.evaluate(test_gen, verbose=1)
print(f"\n{'='*50}")
print(f"Overall Test Loss:     {loss:.4f}")
print(f"Overall Test Accuracy: {accuracy*100:.2f}%")
print(f"{'='*50}")

# Per-class accuracy
print("\nPredicting all test samples...")
preds = model.predict(test_gen, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_gen.classes

print(f"\n{'='*50}")
print(f"{'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
print(f"{'-'*50}")
for i, name in enumerate(CLASS_NAMES):
    mask = true_classes == i
    total = mask.sum()
    correct = (pred_classes[mask] == i).sum()
    acc = correct / total * 100 if total > 0 else 0
    print(f"{name:<12} {correct:>8} {total:>8} {acc:>9.1f}%")
print(f"{'='*50}")
