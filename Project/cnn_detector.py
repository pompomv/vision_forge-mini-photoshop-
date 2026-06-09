"""
cnn_detector.py — CNN Object Recognition using MobileNetV2
Pre-trained on ImageNet (1000 classes). No training required.
Auto-downloads model weights on first use.
Supports category-based filtering for targeted object recognition.
"""

import cv2
import numpy as np


# =============================================================================
# IMAGENET CATEGORY MAPPING
# Keywords digunakan untuk mencocokkan label ImageNet ke kategori user.
# Setiap kategori berisi keyword yang mewakili kelas-kelas ImageNet terkait.
# =============================================================================

CATEGORY_KEYWORDS = {
    'person': [
        'person', 'man', 'woman', 'boy', 'girl', 'baby', 'face',
        'groom', 'brides', 'scuba', 'mask', 'military', 'soldier',
        'ballplayer', 'gymnast', 'swimmer', 'wrestler',
    ],
    'animal': [
        'dog', 'cat', 'bird', 'fish', 'horse', 'cow', 'sheep', 'elephant',
        'bear', 'zebra', 'giraffe', 'monkey', 'rabbit', 'mouse', 'rat',
        'hamster', 'fox', 'wolf', 'lion', 'tiger', 'leopard', 'cheetah',
        'panda', 'koala', 'penguin', 'owl', 'eagle', 'hawk', 'parrot',
        'turtle', 'snake', 'lizard', 'frog', 'salamander', 'whale',
        'dolphin', 'shark', 'crab', 'lobster', 'snail', 'spider',
        'butterfly', 'bee', 'ant', 'beetle', 'cockroach', 'dragonfly',
        'jellyfish', 'starfish', 'coral', 'goldfish', 'poodle', 'retriever',
        'terrier', 'spaniel', 'collie', 'shepherd', 'bulldog', 'husky',
        'chihuahua', 'beagle', 'rottweiler', 'dachshund', 'hound',
        'tabby', 'persian', 'siamese', 'hen', 'rooster', 'goose', 'duck',
        'flamingo', 'pelican', 'heron', 'crane', 'peacock', 'toucan',
        'macaw', 'cockatoo', 'gorilla', 'chimpanzee', 'orangutan', 'baboon',
        'gazelle', 'impala', 'bison', 'ox', 'ram', 'ibex', 'hog', 'pig',
        'boar', 'warthog', 'hippopotamus', 'rhinoceros', 'armadillo',
        'porcupine', 'hedgehog', 'squirrel', 'chipmunk', 'beaver', 'otter',
        'weasel', 'mink', 'skunk', 'badger', 'mongoose', 'hyena',
        'lynx', 'jaguar', 'cougar', 'alligator', 'crocodile', 'iguana',
        'chameleon', 'gecko', 'komodo', 'newt', 'axolotl', 'toad',
        'stingray', 'eel', 'clownfish', 'puffer', 'hermit_crab', 'mantis',
        'scorpion', 'tick', 'centipede', 'worm', 'slug', 'caterpillar',
    ],
    'vehicle': [
        'car', 'truck', 'bus', 'train', 'bicycle', 'motorcycle', 'boat',
        'ship', 'airplane', 'helicopter', 'jet', 'cab', 'taxi', 'van',
        'ambulance', 'fire_engine', 'police', 'tractor', 'tank',
        'convertible', 'limousine', 'minivan', 'jeep', 'sedan', 'wagon',
        'racer', 'sports_car', 'moped', 'scooter', 'rickshaw',
        'snowplow', 'forklift', 'trailer', 'streetcar', 'trolley',
        'locomotive', 'bullet_train', 'freight', 'canoe', 'kayak',
        'sailboat', 'yacht', 'speedboat', 'gondola', 'catamaran',
        'aircraft', 'airliner', 'warplane', 'glider', 'airship',
    ],
    'food': [
        'food', 'fruit', 'vegetable', 'pizza', 'burger', 'sandwich',
        'salad', 'cake', 'pie', 'bread', 'cheese', 'meat', 'sushi',
        'rice', 'noodle', 'pasta', 'soup', 'ice_cream', 'chocolate',
        'cookie', 'donut', 'muffin', 'banana', 'apple', 'orange',
        'strawberry', 'grape', 'pineapple', 'watermelon', 'lemon',
        'tomato', 'potato', 'carrot', 'broccoli', 'corn', 'mushroom',
        'pepper', 'cucumber', 'eggplant', 'pumpkin', 'pretzel',
        'bagel', 'waffle', 'pancake', 'guacamole', 'burrito', 'taco',
        'hotdog', 'cheeseburger', 'french_loaf', 'espresso', 'cup',
    ],
    'nature': [
        'flower', 'tree', 'forest', 'mountain', 'lake', 'river', 'ocean',
        'beach', 'sky', 'cloud', 'sun', 'moon', 'star', 'garden',
        'valley', 'cliff', 'volcano', 'waterfall', 'island', 'desert',
        'daisy', 'rose', 'tulip', 'sunflower', 'orchid', 'lily',
        'coral_reef', 'reef', 'seashore', 'promontory', 'alp',
        'mushroom', 'hay', 'ear', 'acorn', 'rapeseed', 'hip',
    ],
    'electronics': [
        'computer', 'laptop', 'keyboard', 'mouse', 'monitor', 'screen',
        'phone', 'cell', 'tablet', 'camera', 'television', 'tv',
        'remote', 'printer', 'speaker', 'headphone', 'microphone',
        'router', 'modem', 'hard_disc', 'ipod', 'digital_watch',
        'digital_clock', 'joystick', 'console', 'projector', 'web_site',
        'notebook', 'desktop', 'mac', 'space_bar', 'cd_player',
        'cassette', 'radio', 'hand_held', 'cellular', 'iphone', 'switch',
    ],
    'furniture': [
        'chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'bench',
        'cabinet', 'shelf', 'bookcase', 'wardrobe', 'drawer', 'mirror',
        'lamp', 'chandelier', 'stool', 'rocking_chair', 'throne',
        'dining_table', 'studio_couch', 'four_poster', 'cradle', 'crib',
        'folding_chair', 'barber_chair', 'park_bench', 'window_shade',
        'curtain', 'pillow', 'quilt', 'mattress',
    ],
    'clothing': [
        'shirt', 'pants', 'dress', 'skirt', 'jacket', 'coat', 'hat',
        'cap', 'shoe', 'boot', 'sandal', 'sock', 'tie', 'scarf',
        'glove', 'belt', 'sunglasses', 'watch', 'bag', 'backpack',
        'suit', 'gown', 'jersey', 'jean', 'sweatshirt', 'kimono',
        'bikini', 'swimming_trunks', 'miniskirt', 'hoopskirt', 'sarong',
        'sombrero', 'cowboy_hat', 'bonnet', 'mortarboard', 'helmet',
        'maillot', 'tank_suit', 'trench_coat', 'fur_coat', 'poncho',
        'cardigan', 'pajama', 'brassiere', 'running_shoe', 'clog',
        'sandal', 'loafer', 'sneaker', 'oxford', 'mitten', 'stole',
        'feather_boa', 'wig', 'bow_tie', 'windsor_tie', 'bolo_tie',
    ],
}


class CNNDetector:
    """MobileNetV2-based object recognition with category filtering.
    For 'person' category, uses OpenCV Haar Cascade (face + body) as primary detector
    since MobileNetV2/ImageNet is not optimized for human detection.
    """

    def __init__(self):
        self.model = None
        self._decode_predictions = None
        self._preprocess_input = None
        self._loaded = False
        # Lazy-loaded Haar cascades for person detection
        self._face_cascade = None
        self._body_cascade = None

    def _load_cascades(self):
        """Load OpenCV Haar Cascade classifiers for face and body detection."""
        if self._face_cascade is None:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        if self._body_cascade is None:
            self._body_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_fullbody.xml'
            )

    def detect_persons(self, img, top_n=5):
        """
        Detect persons using OpenCV Haar Cascade (face + body).
        More reliable than ImageNet for human detection.
        Returns list of {label, description, confidence, box, matched_category} dicts.
        """
        self._load_cascades()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        results = []

        # --- Face detection ---
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for i, (x, y, fw, fh) in enumerate(faces[:top_n]):
            area_ratio = round((fw * fh) / (w * h) * 100, 1)
            confidence = min(99.0, 50.0 + area_ratio * 2)
            results.append({
                'label': f'face_{i+1}',
                'description': 'Person (Face Detected)',
                'confidence': round(confidence, 2),
                'box': [int(x), int(y), int(fw), int(fh)],
                'matched_category': 'person',
                'method': 'face'
            })

        # --- Full body detection (fallback jika tidak ada wajah) ---
        if len(results) == 0:
            bodies = self._body_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 100)
            )
            for i, (x, y, bw, bh) in enumerate(bodies[:top_n]):
                area_ratio = round((bw * bh) / (w * h) * 100, 1)
                confidence = min(99.0, 40.0 + area_ratio)
                results.append({
                    'label': f'body_{i+1}',
                    'description': 'Person (Body Detected)',
                    'confidence': round(confidence, 2),
                    'box': [int(x), int(y), int(bw), int(bh)],
                    'matched_category': 'person',
                    'method': 'body'
                })

        return results[:top_n]

    def load_model(self):
        """Lazy-load TensorFlow and MobileNetV2 model."""
        try:
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

            from tensorflow.keras.applications import MobileNetV2  # type: ignore
            from tensorflow.keras.applications.mobilenet_v2 import (  # type: ignore
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

    def detect(self, img, top_n=5, category='all'):
        """
        Detect objects in image.
        - category='person': uses OpenCV Haar Cascade (face + body detection)
        - other categories: uses MobileNetV2 (ImageNet 1000 classes)
        Args:
            img: BGR numpy array
            top_n: number of top predictions to return
            category: filter category ('all', 'person', 'animal', etc.)
        Returns list of {label, description, confidence, matched_category} dicts.
        """
        # === Special case: Person detection via OpenCV Haar Cascade ===
        if category == 'person':
            return self.detect_persons(img, top_n)

        # === CNN detection via MobileNetV2 for all other categories ===
        if not self._loaded:
            if not self.load_model():
                return None

        # Preprocess: resize to 224x224, convert BGR to RGB
        resized = cv2.resize(img, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(rgb.astype(np.float32), axis=0)
        x = self._preprocess_input(x)

        # Predict — get more results for filtering
        fetch_n = 20 if category != 'all' else int(top_n)
        preds = self.model.predict(x, verbose=0)
        results = self._decode_predictions(preds, top=fetch_n)[0]

        predictions = [
            {
                'label': label,
                'description': desc.replace('_', ' ').title(),
                'confidence': round(float(conf) * 100, 2)
            }
            for (label, desc, conf) in results
        ]

        # Apply category filter
        if category != 'all' and category in CATEGORY_KEYWORDS:
            keywords = CATEGORY_KEYWORDS[category]
            filtered = []
            for pred in predictions:
                desc_lower = pred['description'].lower()
                label_lower = pred['label'].lower()
                if any(kw in desc_lower or kw in label_lower for kw in keywords):
                    pred['matched_category'] = category
                    filtered.append(pred)
            predictions = filtered[:int(top_n)]

        return predictions[:int(top_n)]

    def detect_and_annotate(self, img, top_n=5, category='all'):
        """
        Detect objects and draw labels/boxes on the image.
        - Person category: draws bounding boxes around detected faces/bodies
        - Other categories: draws label overlay from CNN predictions
        Returns (annotated_image, predictions).
        """
        predictions = self.detect(img, top_n, category)
        if predictions is None:
            return img, None

        annotated = img.copy()
        h, w = annotated.shape[:2]

        if len(predictions) == 0:
            # No matching predictions — show info overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 70), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            cat_name = category.title() if category != 'all' else 'All'
            cv2.putText(annotated, f"No {cat_name} objects detected in image", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            return annotated, predictions

        # === Person: draw bounding boxes around detected faces/bodies ===
        if category == 'person':
            for pred in predictions:
                box = pred.get('box')
                if box:
                    x, y, bw, bh = box
                    method = pred.get('method', 'face')
                    color = (0, 255, 120) if method == 'face' else (0, 180, 255)
                    label = f"{pred['description']} ({pred['confidence']:.1f}%)"
                    # Draw bounding box
                    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
                    # Draw label background
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    label_y = max(y - 10, lh + 5)
                    cv2.rectangle(annotated, (x, label_y - lh - 5), (x + lw + 8, label_y + 3), color, -1)
                    cv2.putText(annotated, label, (x + 4, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

            # Draw summary header
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, 10), (w - 10, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
            cv2.putText(annotated, f"[Face Detect] {len(predictions)} person(s) found", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
            return annotated, predictions

        # === Other categories: draw label overlay from CNN ===
        overlay = annotated.copy()
        header_h = 40
        box_h = header_h + 35 * min(len(predictions), 5) + 10
        cv2.rectangle(overlay, (10, 10), (w - 10, box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

        # Draw category header
        cat_name = category.title() if category != 'all' else 'All Objects'
        cv2.putText(annotated, f"[CNN] Category: {cat_name}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Draw predictions
        y_pos = header_h + 25
        for i, pred in enumerate(predictions[:5]):
            text = f"{i+1}. {pred['description']} ({pred['confidence']:.1f}%)"
            color = (0, 255, 0) if pred['confidence'] > 20 else (0, 200, 255)
            cv2.putText(annotated, text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 35

        return annotated, predictions
