import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    AveragePooling1D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Input,
    BatchNormalization,
    Add,
    Concatenate,
    SeparableConv1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision, Model
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import os
import gc

# Configure environment for GPU optimization
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def configure_gpu_and_precision():
    """Configure GPU and mixed precision for optimal performance."""
    print("⚙️ Configuring GPU and precision...")

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"   GPU configured: {[d.name for d in physical_devices]}")

            # Use mixed precision for better performance
            mixed_precision.set_global_policy("mixed_float16")
            print("   Mixed precision: float16 enabled")

        except Exception as e:
            print(f"   GPU configuration warning: {e}")
    else:
        print("   No GPU detected - using CPU")
        mixed_precision.set_global_policy("float32")


def load_and_preprocess_for_cnn_from_loader(
    batch_loader, test_size=0.2, random_state=42
):
    """
    Load and preprocess data from PickleBatchLoader for CNN training.
    This function adapts the CNN preprocessing to work with your existing batch loader.
    """
    print("📂 Loading data from batch loader for CNN...")

    # Collect all data from batch loader
    all_X = []
    all_y = []

    print("   Collecting batches...")
    for i, (X_batch, y_batch) in enumerate(batch_loader.batch_generator()):
        all_X.append(X_batch)
        all_y.append(y_batch)
        print(f"   Loaded batch {i + 1}/{len(batch_loader)}")

    # Combine all batches
    X = np.vstack(all_X)
    y = np.hstack(all_y)

    print(f"   Combined data shape: X={X.shape}, y={y.shape}")

    # For CNN, we need to reshape the input to (samples, timesteps, features)
    # Assuming X is already in the right format from your preprocessing
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"   CNN input shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Class distribution: {np.bincount(y)}")

    # Clean up intermediate variables
    del all_X, all_y, X, X_reshaped
    gc.collect()

    return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test


def create_optimized_cnn_for_batch_training(
    input_shape, num_classes=2, learning_rate=0.001
):
    """
    Create an optimized CNN model suitable for batch training comparison.
    Simplified version for faster training and comparison with other algorithms.
    """
    print("🧠 Creating optimized CNN model for batch training...")

    inputs = Input(shape=input_shape, name="input_layer")

    # Block 1: Multi-scale feature extraction (simplified)
    branch1 = Conv1D(
        32, 3, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Conv1D(
        48, 5, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(branch1)
    branch1 = BatchNormalization()(branch1)

    branch2 = Conv1D(
        32, 7, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv1D(
        48, 9, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(branch2)
    branch2 = BatchNormalization()(branch2)

    # Concatenate branches
    x = Concatenate()([branch1, branch2])
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.25)(x)

    # Block 2: Residual-like block (simplified)
    shortcut = Conv1D(128, 1, padding="same", kernel_regularizer=l2(0.0001))(x)
    shortcut = BatchNormalization()(shortcut)

    conv = Conv1D(
        64, 7, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(x)
    conv = BatchNormalization()(conv)
    conv = Conv1D(
        128, 7, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(conv)
    conv = BatchNormalization()(conv)

    x = Add()([shortcut, conv])
    x = tf.keras.layers.Activation("relu")(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # Block 3: Final convolutions
    x = Conv1D(
        256, 11, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)

    x = Conv1D(
        256, 13, activation="relu", padding="same", kernel_regularizer=l2(0.0001)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Global pooling and dense layers
    x = GlobalMaxPooling1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(256, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.45)(x)

    x = Dense(128, activation="relu", kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.4)(x)

    # Output layer
    outputs = Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax",
        kernel_regularizer=l2(0.0001),
        dtype="float32",
        name="predictions",
    )(x)

    model = Model(inputs=inputs, outputs=outputs, name="Optimized_CNN")

    # Compile model
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1.0)

    # Handle mixed precision
    current_policy = mixed_precision.global_policy().name
    if current_policy == "mixed_float16":
        optimizer = mixed_precision.LossScaleOptimizer(
            optimizer, initial_scale=1024.0, dynamic=True
        )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy"
        if num_classes == 2
        else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    total_params = model.count_params()
    print(f"✅ Optimized CNN Model created!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Estimated memory usage: ~{total_params * 2 / (1024**2):.1f} MB")

    return model


def train_cnn_with_batch_loader(
    batch_loader, epochs=30, batch_size=256, patience=8, learning_rate=0.001
):
    """
    Train CNN using the same batch loader as other algorithms for fair comparison.
    """
    print("🚀 Training CNN with batch loader...")

    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_for_cnn_from_loader(
            batch_loader
        )

        # Get input shape
        input_shape = (X_train.shape[1], X_train.shape[2])
        print(f"   Input shape for CNN: {input_shape}")

        # Create model
        model = create_optimized_cnn_for_batch_training(
            input_shape, learning_rate=learning_rate
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
                mode="max",
                min_delta=0.002,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
        ]

        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = (
            train_dataset.shuffle(min(4096, len(X_train))).batch(batch_size).prefetch(1)
        )

        val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_dataset = val_dataset.batch(batch_size).prefetch(1)

        print(f"   Starting training with batch_size={batch_size}, epochs={epochs}")

        # Train the model
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1,
        )

        # Final evaluation
        test_loss, test_accuracy = model.evaluate(val_dataset, verbose=0)

        print(f"📊 CNN Training completed!")
        print(f"   Final Test Accuracy: {test_accuracy:.4f}")

        return test_accuracy * 100

    except Exception as e:
        print(f"❌ CNN Training failed: {e}")
        return 0.0

    finally:
        # Cleanup
        tf.keras.backend.clear_session()
        gc.collect()


def clear_gpu_memory():
    """Clear GPU memory between training runs."""
    try:
        tf.keras.backend.clear_session()
        if tf.config.list_physical_devices("GPU"):
            physical_devices = tf.config.list_physical_devices("GPU")
            for device in physical_devices:
                try:
                    tf.config.experimental.reset_memory_stats(device)
                except:
                    pass
        gc.collect()
        print("✅ GPU memory cleared")
    except Exception as e:
        print(f"⚠️ GPU cleanup warning: {e}")


# Initialize GPU configuration when module is imported
configure_gpu_and_precision()
