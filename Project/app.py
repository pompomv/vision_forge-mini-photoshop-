"""
app.py — Flask Server for Mini Photoshop
Serves REST API for image processing and static frontend.
"""

from flask import Flask, request, jsonify, render_template, session, send_file
import os
import uuid
import cv2
import numpy as np
import base64
import io
from image_processor import ImageProcessor
from compressor import (
    compress_huffman, compress_arithmetic, compress_lzw,
    compress_rle, compress_quantize, save_with_quality
)
from cnn_detector import CNNDetector

app = Flask(__name__)
app.secret_key = 'mini-photoshop-pcd-2024'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

processor = ImageProcessor()
detector = CNNDetector()


# =============================================================================
# HELPERS
# =============================================================================

def get_session_id():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())[:8]
    return session['sid']


def get_paths():
    sid = get_session_id()
    return (
        os.path.join(UPLOAD_FOLDER, f'{sid}_original.png'),
        os.path.join(UPLOAD_FOLDER, f'{sid}_current.png')
    )


def img_to_base64(img, fmt='.png'):
    _, buf = cv2.imencode(fmt, img)
    return base64.b64encode(buf).decode('utf-8')


def read_current():
    _, curr_path = get_paths()
    if not os.path.exists(curr_path):
        return None
    return cv2.imread(curr_path)


def save_current(img):
    _, curr_path = get_paths()
    cv2.imwrite(curr_path, img)


def error(msg, code=400):
    return jsonify({'success': False, 'error': msg}), code


def success(img=None, extra=None):
    data = {'success': True}
    if img is not None:
        data['image'] = img_to_base64(img)
        h, w = img.shape[:2]
        data['info'] = {'width': w, 'height': h, 'channels': img.shape[2] if len(img.shape) == 3 else 1}
    if extra:
        data.update(extra)
    return jsonify(data)


# =============================================================================
# ROUTES — Pages
# =============================================================================

@app.route('/')
def index():
    return render_template('index.html')


# =============================================================================
# ROUTES — Image Management
# =============================================================================

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return error('No file provided')
    f = request.files['file']
    if f.filename == '':
        return error('No file selected')

    # Read file into numpy array
    file_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return error('Cannot read image file')

    orig_path, curr_path = get_paths()
    cv2.imwrite(orig_path, img)
    cv2.imwrite(curr_path, img)

    return success(img)


@app.route('/api/reset', methods=['POST'])
def reset():
    orig_path, curr_path = get_paths()
    if not os.path.exists(orig_path):
        return error('No image loaded')
    img = cv2.imread(orig_path)
    cv2.imwrite(curr_path, img)
    return success(img)


@app.route('/api/save', methods=['POST'])
def save():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    filename = data.get('filename', 'output')
    fmt = data.get('format', 'png').lower()
    quality = int(data.get('quality', 95))

    ext = '.' + fmt
    if fmt == 'jpg':
        ext = '.jpg'
    save_path = os.path.join(OUTPUT_FOLDER, filename + ext)

    if fmt in ('jpg', 'jpeg'):
        cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif fmt == 'png':
        cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, min(9, (100 - quality) // 10)])
    else:
        cv2.imwrite(save_path, img)

    return jsonify({
        'success': True,
        'path': save_path,
        'filename': os.path.basename(save_path)
    })


@app.route('/api/download', methods=['GET'])
def download():
    img = read_current()
    if img is None:
        return error('No image loaded')

    fmt = request.args.get('format', 'png')
    ext = '.jpg' if fmt in ('jpg', 'jpeg') else f'.{fmt}'
    _, buf = cv2.imencode(ext, img)

    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype=f'image/{fmt}',
        as_attachment=True,
        download_name=f'mini_photoshop_output{ext}'
    )


# =============================================================================
# ROUTES — Image Enhancement
# =============================================================================

@app.route('/api/enhance', methods=['POST'])
def enhance():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    op = data.get('operation', '')
    apply_change = data.get('apply', True)

    # For preview mode, read from original + accumulated changes
    if op == 'brightness':
        result = processor.adjust_brightness(img, data.get('value', 0))
    elif op == 'contrast':
        result = processor.adjust_contrast(img, data.get('alpha', 1.0))
    elif op == 'histogram_eq':
        result = processor.histogram_equalization(img)
    elif op == 'sharpen':
        result = processor.sharpen(img, data.get('strength', 1.0))
    elif op == 'smooth':
        result = processor.smooth(img, data.get('ksize', 5), data.get('method', 'gaussian'))
    else:
        return error(f'Unknown enhancement operation: {op}')

    if apply_change:
        save_current(result)
    return success(result)


# =============================================================================
# ROUTES — Geometric Transformation
# =============================================================================

@app.route('/api/transform', methods=['POST'])
def transform():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    op = data.get('operation', '')

    if op == 'rotate':
        result = processor.rotate_image(img, data.get('angle', 0), data.get('interpolation', 'bilinear'))
    elif op == 'flip':
        result = processor.flip_image(img, data.get('direction', 'horizontal'))
    elif op == 'crop':
        result = processor.crop_image(img, data.get('x', 0), data.get('y', 0),
                                       data.get('w', img.shape[1]), data.get('h', img.shape[0]))
    elif op == 'resize':
        result = processor.resize_image(img, data.get('width'), data.get('height'),
                                         data.get('interpolation', 'bilinear'))
    elif op == 'translate':
        result = processor.translate_image(img, data.get('tx', 0), data.get('ty', 0))
    else:
        return error(f'Unknown transform operation: {op}')

    save_current(result)
    return success(result)


# =============================================================================
# ROUTES — Image Restoration
# =============================================================================

@app.route('/api/restore', methods=['POST'])
def restore():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    op = data.get('operation', '')

    if op == 'gaussian':
        result = processor.gaussian_blur(img, data.get('ksize', 5))
    elif op == 'median':
        result = processor.median_filter(img, data.get('ksize', 5))
    elif op == 'salt_pepper':
        result = processor.remove_salt_pepper(img, data.get('ksize', 5))
    else:
        return error(f'Unknown restoration operation: {op}')

    save_current(result)
    return success(result)


# =============================================================================
# ROUTES — Binary & Edge Processing
# =============================================================================

@app.route('/api/edge', methods=['POST'])
def edge():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    op = data.get('operation', '')

    if op == 'threshold':
        result = processor.threshold_binary(img, data.get('thresh', 128))
    elif op == 'canny':
        result = processor.edge_canny(img, data.get('t1', 50), data.get('t2', 150))
    elif op == 'sobel':
        result = processor.edge_sobel(img)
    elif op == 'prewitt':
        result = processor.edge_prewitt(img)
    elif op == 'roberts':
        result = processor.edge_roberts(img)
    elif op == 'laplacian':
        result = processor.edge_laplacian(img)
    elif op == 'log':
        result = processor.edge_log(img, data.get('ksize', 5))
    elif op == 'erode':
        result = processor.morphology_erode(img, data.get('ksize', 5), data.get('iterations', 1))
    elif op == 'dilate':
        result = processor.morphology_dilate(img, data.get('ksize', 5), data.get('iterations', 1))
    else:
        return error(f'Unknown edge operation: {op}')

    save_current(result)
    return success(result)


# =============================================================================
# ROUTES — Color Processing
# =============================================================================

@app.route('/api/color', methods=['POST'])
def color():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    op = data.get('operation', '')

    if op == 'grayscale':
        result = processor.to_grayscale(img)
        save_current(result)
        return success(result)
    elif op == 'split':
        channels = processor.split_channels(img)
        channel = data.get('channel', 'red')
        result = channels.get(channel, img)
        save_current(result)
        return success(result)
    elif op == 'hue_saturation':
        result = processor.adjust_hue_saturation(img, data.get('hue', 0), data.get('saturation', 1.0))
        save_current(result)
        return success(result)
    else:
        return error(f'Unknown color operation: {op}')


# =============================================================================
# ROUTES — Segmentation
# =============================================================================

@app.route('/api/segment', methods=['POST'])
def segment():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    op = data.get('operation', '')

    if op == 'threshold':
        result = processor.segment_threshold(img, data.get('thresh', 128))
    elif op == 'edge':
        result = processor.segment_edge(img)
    elif op == 'region':
        result = processor.segment_region(img, data.get('regions', 4))
    else:
        return error(f'Unknown segmentation operation: {op}')

    save_current(result)
    return success(result)


# =============================================================================
# ROUTES — Compression
# =============================================================================

@app.route('/api/compress', methods=['POST'])
def compress():
    img = read_current()
    if img is None:
        return error('No image loaded')

    data = request.get_json(silent=True) or {}
    method = data.get('method', 'huffman')

    compress_funcs = {
        'huffman': lambda: compress_huffman(img),
        'arithmetic': lambda: compress_arithmetic(img, data.get('block_size', 16)),
        'lzw': lambda: compress_lzw(img),
        'rle': lambda: compress_rle(img),
        'quantize': lambda: compress_quantize(img, data.get('levels', 16)),
    }

    if method not in compress_funcs:
        return error(f'Unknown compression method: {method}')

    result, stats = compress_funcs[method]()
    save_current(result)
    return success(result, extra={'stats': stats})


# =============================================================================
# ROUTES — Histogram
# =============================================================================

@app.route('/api/histogram', methods=['GET'])
def histogram():
    orig_path, curr_path = get_paths()

    result = {}
    if os.path.exists(orig_path):
        orig = cv2.imread(orig_path)
        result['before'] = processor.compute_histogram(orig)
    if os.path.exists(curr_path):
        curr = cv2.imread(curr_path)
        result['after'] = processor.compute_histogram(curr)

    return jsonify({'success': True, 'histograms': result})


# =============================================================================
# ROUTES — CNN Object Detection
# =============================================================================

@app.route('/api/detect', methods=['POST'])
def detect():
    img = read_current()
    if img is None:
        return error('No image loaded')

    annotated, predictions = detector.detect_and_annotate(img)
    if predictions is None:
        return error('TensorFlow not installed. Run: pip install tensorflow')

    save_current(annotated)
    return success(annotated, extra={'predictions': predictions})


# =============================================================================
# ROUTES — Get current & original images
# =============================================================================

@app.route('/api/current', methods=['GET'])
def get_current():
    img = read_current()
    if img is None:
        return error('No image loaded')
    return success(img)


@app.route('/api/original', methods=['GET'])
def get_original():
    orig_path, _ = get_paths()
    if not os.path.exists(orig_path):
        return error('No image loaded')
    img = cv2.imread(orig_path)
    return success(img)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("  Mini Photoshop - Digital Image Processing")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
