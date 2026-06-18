import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization


def build_model(input_shape=(96, 96, 3), num_classes=8):
    """
    Xây dựng mô hình Transfer Learning sử dụng EfficientNetB0.

    Chiến lược: Mở khóa toàn bộ mô hình ngay từ đầu và train với learning rate thấp.
    Lý do: ImageNet features (chó, mèo, xe cộ) rất khác với facial expressions.
    Cần cho phép EfficientNet "quên bớt" kiến thức cũ và học cách nhận diện
    nếp nhăn, khóe miệng, ánh mắt ngay từ đầu.
    """

    # Load EfficientNetB0 (bao gồm preprocessing layer bên trong)
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # MỞ KHÓA TOÀN BỘ để train lại - vì domain quá khác biệt
    base_model.trainable = True

    # Head đơn giản hơn - tránh overfitting
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model
