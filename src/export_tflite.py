import os
import tensorflow as tf


def convert_to_tflite():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    keras_model_path = os.path.join(models_dir, 'emotion_model.h5')
    tflite_model_path = os.path.join(models_dir, 'emotion_model.tflite')

    if not os.path.exists(keras_model_path):
        print(f"Error: Keras model not found at {keras_model_path}")
        print("Please train the model first by running `python src/train.py`.")
        return

    print("Loading Keras model (EfficientNetB0)...")
    model = tf.keras.models.load_model(keras_model_path)

    print("Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Áp dụng lượng tử hóa (Quantization) để giảm dung lượng
    # và tăng tốc độ inference trên mobile
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Đảm bảo tương thích với các op mới của EfficientNet
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    print(f"Saving TFLite model to {tflite_model_path}...")
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # In thông tin file
    size_mb = os.path.getsize(tflite_model_path) / (1024 * 1024)
    print(f"\nConversion completed successfully!")
    print(f"TFLite model size: {size_mb:.2f} MB")
    print(f"Input shape: {model.input_shape}")
    print(f"Output classes: {model.output_shape[-1]}")


if __name__ == '__main__':
    convert_to_tflite()
