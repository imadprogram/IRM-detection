import os
import sys
import argparse
from pathlib import Path

# add root directory to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import tensorflow as tf
from src.preprocess import load_classification_datasets, img_size, batch_size


def build_mobilenet_model(input_shape=(224, 224, 3), num_classes=4):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def train_classifier(
    data_dir='data',
    models_dir='models',
    epochs_phase1=15,
    epochs_phase2=10,
    batch=batch_size
):
    os.makedirs(models_dir, exist_ok=True)
    models_path = Path(models_dir)

    train_ds, val_ds, test_ds = load_classification_datasets(
        data_dir=data_dir,
        target_size=img_size,
        batch=batch
    )

    model, base_model = build_mobilenet_model(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=4
    )

    # compile feature extraction phase
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_path / 'best_mobilenet_phase1.keras'),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    print('=== starting phase: feature extraction ===')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=callbacks_phase1
    )

    # unfreeze top layers for fine tuning
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # recompile with low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(models_path / 'best_mobilenet_fine_tuned.keras'),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
    ]

    print('=== starting phase: fine tuning ===')
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        callbacks=callbacks_phase2
    )

    print(f'training complete. model saved to {models_path / "best_mobilenet_fine_tuned.keras"}')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train mobilenet classifier')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--epochs-phase1', type=int, default=15)
    parser.add_argument('--epochs-phase2', type=int, default=10)
    args = parser.parse_args()

    train_classifier(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2
    )
