import os
import sys
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers

# dataset configuration parameters
img_size = (224, 224)
batch_size = 32
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


def get_augmentation_layer():
    # sequential augmentation layers for training
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def verify_dataset(data_dir="data"):
    # verify dataset directories and file counts
    base_path = Path(data_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"data directory not found at {base_path}")

    splits = ["Train", "Val", "Test"]
    report = {}

    for split in splits:
        split_path = base_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"split directory missing: {split_path}")

        split_counts = {}
        for cls in classes:
            img_dir = split_path / cls / "images"
            if not img_dir.exists():
                raise FileNotFoundError(f"missing image directory: {img_dir}")

            count = len([f for f in img_dir.iterdir() if f.is_file()])
            if count == 0:
                raise ValueError(f"empty image directory: {img_dir}")
            split_counts[cls] = count

        report[split] = split_counts

    # verify yolo configuration file exists
    yaml_file = base_path / "mri_data.yaml"
    if not yaml_file.exists():
        raise FileNotFoundError(f"missing configuration file: {yaml_file}")

    return report


def generate_yolo_manifests(data_dir="data"):
    # generate text files listing image paths for yolo training
    base_path = Path(data_dir)
    manifest_files = {}

    for split in ["Train", "Val", "Test"]:
        img_paths = []
        for cls in classes:
            img_dir = base_path / split / cls / "images"
            for f in img_dir.glob("*.*"):
                if f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    img_paths.append(str(f.resolve()))

        out_file = base_path / f"{split.lower()}.txt"
        with open(out_file, "w") as f:
            for p in sorted(img_paths):
                f.write(f"{p}\n")
        manifest_files[split] = out_file

    return manifest_files


def load_classification_datasets(data_dir="data", target_size=img_size, batch=batch_size):
    # load stratified splits for classification
    base_path = Path(data_dir)
    autotune = tf.data.AUTOTUNE

    # load training split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        base_path / "Train",
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=target_size,
        batch_size=batch,
        shuffle=True,
        seed=42,
    )

    # load validation split
    val_ds = tf.keras.utils.image_dataset_from_directory(
        base_path / "Val",
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=target_size,
        batch_size=batch,
        shuffle=False,
    )

    # load test split
    test_ds = tf.keras.utils.image_dataset_from_directory(
        base_path / "Test",
        labels="inferred",
        label_mode="categorical",
        class_names=classes,
        image_size=target_size,
        batch_size=batch,
        shuffle=False,
    )

    # apply augmentation to training set
    aug = get_augmentation_layer()
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=autotune)

    # configure prefetching for performance
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds = val_ds.prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # execute verification as standalone script
    print("=== verifying dataset integrity ===")
    try:
        data_directory = sys.argv[1] if len(sys.argv) > 1 else "data"
        counts = verify_dataset(data_directory)
        for split_name, class_stats in counts.items():
            total = sum(class_stats.values())
            print(f"[{split_name}] total images: {total}")
            for c_name, c_cnt in class_stats.items():
                print(f"  - {c_name}: {c_cnt}")

        print("\n=== checking yolo manifests ===")
        manifests = generate_yolo_manifests(data_directory)
        for split_name, path in manifests.items():
            print(f"[{split_name}] manifest ready: {path.name}")

        print("\nall dataset checks passed successfully")
        sys.exit(0)
    except Exception as exc:
        print(f"dataset verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
