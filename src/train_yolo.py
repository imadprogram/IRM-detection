import os
import sys
import shutil
import argparse
from pathlib import Path

# add root directory to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from ultralytics import YOLO


def train_yolo(
    data_yaml='data/mri_data.yaml',
    model_type='yolo11n.pt',
    epochs=15,
    imgsz=640,
    batch=16,
    models_dir='models'
):
    models_path = Path(models_dir)
    os.makedirs(models_path, exist_ok=True)

    print(f'loading base model: {model_type}')
    model = YOLO(model_type)

    print(f'starting training on {data_yaml}')
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/detect',
        name='mri_training'
    )

    best_weights = Path(results.save_dir) / 'weights' / 'best.pt'
    target_name = 'best_yolov11.pt' if '11' in model_type else 'best.pt'
    dest_path = models_path / target_name

    # copy best weights to models directory
    if best_weights.exists():
        shutil.copy(best_weights, dest_path)
        print(f'saved trained weights to {dest_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train yolo detector')
    parser.add_argument('--data-yaml', type=str, default='data/mri_data.yaml')
    parser.add_argument('--model-type', type=str, default='yolo11n.pt')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--models-dir', type=str, default='models')
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data_yaml,
        model_type=args.model_type,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        models_dir=args.models_dir
    )
