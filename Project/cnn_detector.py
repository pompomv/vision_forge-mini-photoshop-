"""
cnn_detector.py — CNN Object Recognition using MobileNetV2
Pre-trained on ImageNet (1000 classes). No training required.
Auto-downloads model weights on first use.
"""

import cv2
import numpy as np


class CNNDetector:
    """MobileNetV2-based object recognition."""

    def __init__(self):
        self.model = None
        self._decode_predictions = None
        self._preprocess_input = None
        self._loaded = False

    def load_model(self):
        """Lazy-load TensorFlow and MobileNetV2 model."""
        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import (
                decode_predictions, preprocess_input
            )

            print("[CNN] Loading MobileNetV2 model...")
            self.model = MobileNetV2(weights='imagenet')
            self._decode_predictions = decode_predictions
            self._preprocess_input = preprocess_input
            self._loaded = True
            print("[CNN] Model loaded successfully!")
            return True

        except ImportError:
            print("[CNN] TensorFlow not installed. Install with: pip install tensorflow")
            return False
        except Exception as e:
            print(f"[CNN] Error loading model: {e}")
            return False

    def detect(self, img, top_n=5):
        """
        Detect objects in image using MobileNetV2.
        Returns list of {label, description, confidence} dicts.
        """
        if not self._loaded:
            if not self.load_model():
                return None

        # Preprocess: resize to 224x224, convert BGR to RGB
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(rgb.astype(np.float32), axis=0)
        x = self._preprocess_input(x)

        # Predict
        preds = self.model.predict(x, verbose=0)
        results = self._decode_predictions(preds, top=int(top_n))[0]

        return [
            {
                'label': label,
                'description': desc.replace('_', ' ').title(),
                'confidence': round(float(conf) * 100, 2)
            }
            for (label, desc, conf) in results
        ]

    def detect_and_annotate(self, img, top_n=5):
        """
        Detect objects and draw labels on the image.
        Returns (annotated_image, predictions).
        """
        predictions = self.detect(img, top_n)
        if predictions is None:
            return img, []

        annotated = img.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent overlay for labels
        overlay = annotated.copy()
        box_h = 40 * min(len(predictions), 5) + 20
        cv2.rectangle(overlay, (10, 10), (w - 10, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Draw predictions
        y_pos = 40
        for i, pred in enumerate(predictions[:5]):
            text = f"{i+1}. {pred['description']} ({pred['confidence']:.1f}%)"
            color = (0, 255, 0) if pred['confidence'] > 20 else (0, 200, 255)
            cv2.putText(annotated, text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 35

        return annotated, predictions
