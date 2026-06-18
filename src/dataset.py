import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Thứ tự class cố định (khớp với alphabetical sort của thư mục)
CLASS_NAMES = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def get_data_generators(train_dir, test_dir, batch_size=32, img_size=(96, 96)):
    """
    Tạo data generators cho bộ dữ liệu AffectNet 96x96.
    Sử dụng preprocess_input của EfficientNet để chuẩn hóa ảnh đúng cách.
    """

    # Data Augmentation + preprocessing_function của EfficientNet
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        shear_range=0.1,
        fill_mode='nearest'
    )

    # Test chỉ cần preprocess_input, không augmentation
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print("Loading training data from AffectNet...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
        interpolation='bilinear'
    )

    print("Loading validation/test data from AffectNet...")
    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode="rgb",
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
        interpolation='bilinear'
    )

    # In ra mapping của các lớp
    print(f"\nTrain class mapping: {train_generator.class_indices}")
    print(f"Test class mapping:  {validation_generator.class_indices}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Number of classes: {train_generator.num_classes}")

    return train_generator, validation_generator
