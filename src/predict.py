import os
import sys
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

# add root directory to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocess import classes, img_size


def load_classifier(model_path='models/best_mobilenet_fine_tuned.keras'):
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f'classifier not found: {path_obj}')
    return tf.keras.models.load_model(str(path_obj))


def load_detector(model_path='models/best_yolov11.pt'):
    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f'detector not found: {path_obj}')
    return YOLO(str(path_obj))


def predict_classification(model, image):
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB') if hasattr(image, 'convert') else Image.fromarray(image).convert('RGB')

    img_resized = img.resize(img_size)
    img_arr = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_arr, axis=0)

    # predict class probabilities
    probs = model.predict(img_batch, verbose=0)[0]
    idx = int(np.argmax(probs))

    prob_dict = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}

    return {
        'class': classes[idx],
        'confidence': round(float(probs[idx]), 4),
        'probabilities': prob_dict
    }


def predict_detection(detector, image, conf_threshold=0.25):
    # detect bounding boxes on image
    results = detector.predict(source=image, conf=conf_threshold, verbose=False)
    r = results[0]

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        score = round(float(box.conf[0]), 4)
        coords = [round(float(c), 1) for c in box.xyxy[0].tolist()]

        detections.append({
            'class': detector.names[cls_id],
            'confidence': score,
            'box': coords
        })

    # convert bgr plot to rgb for display
    annotated_rgb = r.plot()[:, :, ::-1]

    return {
        'detections': detections,
        'count': len(detections),
        'annotated_image': annotated_rgb
    }


def predict_pipeline(
    image_path,
    classifier_path='models/best_mobilenet_fine_tuned.keras',
    detector_path='models/best_yolov11.pt',
    conf=0.25
):
    classifier = load_classifier(classifier_path)
    detector = load_detector(detector_path)

    clf_result = predict_classification(classifier, image_path)
    det_result = predict_detection(detector, image_path, conf_threshold=conf)

    return {
        'classification': clf_result,
        'detection': det_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run dual inference on mri')
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--classifier', type=str, default='models/best_mobilenet_fine_tuned.keras')
    parser.add_argument('--detector', type=str, default='models/best_yolov11.pt')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    out = predict_pipeline(
        image_path=args.image,
        classifier_path=args.classifier,
        detector_path=args.detector,
        conf=args.conf
    )

    print('\n=== classification result ===')
    print(f"diagnosed tumor: {out['classification']['class']} ({out['classification']['confidence'] * 100:.1f}%)")
    print('probabilities:')
    for k, v in out['classification']['probabilities'].items():
        print(f'  - {k}: {v * 100:.1f}%')

    print('\n=== detection result ===')
    print(f"tumors detected: {out['detection']['count']}")
    for d in out['detection']['detections']:
        print(f"  - {d['class']} ({d['confidence'] * 100:.1f}%) at {d['box']}")
