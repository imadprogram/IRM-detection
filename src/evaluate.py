import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score
from ultralytics import YOLO

# add root directory to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocess import classes, img_size, batch_size


def evaluate_classifier(model_path='models/best_mobilenet_fine_tuned.keras', data_dir='data'):
    test_dir = Path(data_dir) / 'Test'
    if not Path(model_path).exists():
        print(f'classifier model not found at {model_path}')
        return None

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='categorical',
        class_names=classes,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    model = tf.keras.models.load_model(model_path)
    y_probs = model.predict(test_ds)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in test_ds], axis=0)

    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    return {
        'accuracy': round(acc, 4),
        'macro_f1': round(float(report['macro avg']['f1-score']), 4),
        'weighted_f1': round(float(report['weighted avg']['f1-score']), 4),
        'per_class': {
            cls: {
                'precision': round(float(report[cls]['precision']), 4),
                'recall': round(float(report[cls]['recall']), 4),
                'f1_score': round(float(report[cls]['f1-score']), 4)
            }
            for cls in classes
        }
    }


def evaluate_yolo(model_path, data_yaml='data/mri_data.yaml'):
    path_obj = Path(model_path)
    if not path_obj.exists():
        print(f'yolo model not found at {model_path}')
        return None

    model = YOLO(str(path_obj))
    metrics = model.val(data=data_yaml, split='test')

    return {
        'map50': round(float(metrics.box.map50), 4),
        'map50_95': round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall': round(float(metrics.box.mr), 4)
    }


def evaluate_all(
    data_dir='data',
    data_yaml='data/mri_data.yaml',
    output_path='models/metrics.json'
):
    print('=== evaluating models on test set ===')
    metrics_summary = {}

    # evaluate classification model
    print('\n--- evaluating mobilenet classifier ---')
    clf_res = evaluate_classifier(data_dir=data_dir)
    if clf_res:
        metrics_summary['classification_mobilenet'] = clf_res
        print(f"test accuracy: {clf_res['accuracy'] * 100:.2f}%")
        print(f"macro f1:     {clf_res['macro_f1']:.4f}")

    # evaluate detection models
    print('\n--- evaluating yolov8 detector ---')
    yolo8_res = evaluate_yolo('models/best.pt', data_yaml=data_yaml)
    if yolo8_res:
        metrics_summary['detection_yolov8'] = yolo8_res
        print(f"map50: {yolo8_res['map50']:.4f}")

    print('\n--- evaluating yolo11 detector ---')
    yolo11_res = evaluate_yolo('models/best_yolov11.pt', data_yaml=data_yaml)
    if yolo11_res:
        metrics_summary['detection_yolo11'] = yolo11_res
        print(f"map50: {yolo11_res['map50']:.4f}")

    # save metrics to json file
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print(f'\nall metrics saved to {out_file}')
    return metrics_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate mri models')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--data-yaml', type=str, default='data/mri_data.yaml')
    parser.add_argument('--output', type=str, default='models/metrics.json')
    args = parser.parse_args()

    evaluate_all(
        data_dir=args.data_dir,
        data_yaml=args.data_yaml,
        output_path=args.output
    )
