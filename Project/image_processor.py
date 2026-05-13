"""
image_processor.py — Core Image Processing Engine for Mini Photoshop
All functions accept BGR numpy arrays and return BGR numpy arrays (unless noted).
"""

import cv2
import numpy as np


class ImageProcessor:
    """Stateless image processing utility class."""

    # =========================================================================
    # IMAGE ENHANCEMENT
    # =========================================================================

    @staticmethod
    def adjust_brightness(img, value=0):
        """Adjust brightness by adding value to all pixels. Range: -100 to 100."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(value), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_contrast(img, alpha=1.0):
        """Adjust contrast by scaling pixel values. Range: 0.5 to 3.0."""
        result = cv2.convertScaleAbs(img, alpha=float(alpha), beta=0)
        return result

    @staticmethod
    def histogram_equalization(img):
        """Apply histogram equalization (works on color and grayscale)."""
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def sharpen(img, strength=1.0):
        """Unsharp masking with adjustable strength."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(img, 1.0 + float(strength), blurred, -float(strength), 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def smooth(img, ksize=5, method='gaussian'):
        """Smooth/blur image. Methods: gaussian, median, bilateral."""
        k = int(ksize)
        if k % 2 == 0:
            k += 1
        if method == 'gaussian':
            return cv2.GaussianBlur(img, (k, k), 0)
        elif method == 'median':
            return cv2.medianBlur(img, k)
        elif method == 'bilateral':
            return cv2.bilateralFilter(img, k, 75, 75)
        return img

    # =========================================================================
    # GEOMETRIC TRANSFORMATION
    # =========================================================================

    @staticmethod
    def rotate_image(img, angle=0, interpolation='bilinear'):
        """Rotate image by angle degrees (affine transformation)."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        interp = cv2.INTER_LINEAR if interpolation == 'bilinear' else cv2.INTER_NEAREST
        M = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        # Calculate new bounding box
        cos_a = np.abs(M[0, 0])
        sin_a = np.abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(img, M, (new_w, new_h), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    @staticmethod
    def flip_image(img, direction='horizontal'):
        """Flip image. direction: horizontal, vertical, both."""
        if direction == 'horizontal':
            return cv2.flip(img, 1)
        elif direction == 'vertical':
            return cv2.flip(img, 0)
        elif direction == 'both':
            return cv2.flip(img, -1)
        return img

    @staticmethod
    def crop_image(img, x, y, w, h):
        """Crop image to specified rectangle."""
        ih, iw = img.shape[:2]
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(iw, int(x + w))
        y2 = min(ih, int(y + h))
        return img[y1:y2, x1:x2].copy()

    @staticmethod
    def resize_image(img, width=None, height=None, interpolation='bilinear'):
        """Resize image. If only one dimension given, maintain aspect ratio."""
        h, w = img.shape[:2]
        interp = cv2.INTER_LINEAR if interpolation == 'bilinear' else cv2.INTER_NEAREST
        if width and height:
            return cv2.resize(img, (int(width), int(height)), interpolation=interp)
        elif width:
            ratio = int(width) / w
            return cv2.resize(img, (int(width), int(h * ratio)), interpolation=interp)
        elif height:
            ratio = int(height) / h
            return cv2.resize(img, (int(w * ratio), int(height)), interpolation=interp)
        return img

    @staticmethod
    def translate_image(img, tx=0, ty=0):
        """Translate (shift) image position using affine matrix."""
        h, w = img.shape[:2]
        M = np.float32([[1, 0, float(tx)], [0, 1, float(ty)]])
        return cv2.warpAffine(img, M, (w, h))

    # =========================================================================
    # IMAGE RESTORATION (NOISE REDUCTION)
    # =========================================================================

    @staticmethod
    def gaussian_blur(img, ksize=5):
        """Gaussian blur for noise reduction (spatial filtering)."""
        k = int(ksize)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def median_filter(img, ksize=5):
        """Median filter for salt & pepper noise."""
        k = int(ksize)
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(img, k)

    @staticmethod
    def remove_salt_pepper(img, ksize=5):
        """Remove salt & pepper noise using median filtering."""
        k = int(ksize)
        if k % 2 == 0:
            k += 1
        return cv2.medianBlur(img, k)

    # =========================================================================
    # BINARY & EDGE PROCESSING
    # =========================================================================

    @staticmethod
    def threshold_binary(img, thresh=128):
        """Binary thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
        if len(img.shape) == 3:
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return binary

    @staticmethod
    def edge_canny(img, t1=50, t2=150):
        """Canny edge detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, int(t1), int(t2))
        if len(img.shape) == 3:
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges

    @staticmethod
    def edge_sobel(img):
        """Sobel edge detection (gradient magnitude)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
        if len(img.shape) == 3:
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        return mag

    @staticmethod
    def edge_prewitt(img):
        """Prewitt edge detection using custom kernels."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        px = cv2.filter2D(gray, cv2.CV_64F, kx)
        py = cv2.filter2D(gray, cv2.CV_64F, ky)
        mag = cv2.convertScaleAbs(cv2.magnitude(px, py))
        if len(img.shape) == 3:
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        return mag

    @staticmethod
    def edge_roberts(img):
        """Roberts Cross edge detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kx = np.array([[1, 0], [0, -1]], dtype=np.float64)
        ky = np.array([[0, 1], [-1, 0]], dtype=np.float64)
        rx = cv2.filter2D(gray, cv2.CV_64F, kx)
        ry = cv2.filter2D(gray, cv2.CV_64F, ky)
        mag = cv2.convertScaleAbs(cv2.magnitude(rx, ry))
        if len(img.shape) == 3:
            return cv2.cvtColor(mag, cv2.COLOR_GRAY2BGR)
        return mag

    @staticmethod
    def edge_laplacian(img):
        """Laplacian edge detection (2nd derivative)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        result = cv2.convertScaleAbs(lap)
        if len(img.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result

    @staticmethod
    def edge_log(img, ksize=5):
        """Laplacian of Gaussian edge detection."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        k = int(ksize)
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        lap = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        result = cv2.convertScaleAbs(lap)
        if len(img.shape) == 3:
            return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result

    @staticmethod
    def morphology_erode(img, ksize=5, iterations=1):
        """Morphological erosion with structuring element."""
        k = int(ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.erode(img, kernel, iterations=int(iterations))

    @staticmethod
    def morphology_dilate(img, ksize=5, iterations=1):
        """Morphological dilation with structuring element."""
        k = int(ksize)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return cv2.dilate(img, kernel, iterations=int(iterations))

    # =========================================================================
    # COLOR PROCESSING
    # =========================================================================

    @staticmethod
    def to_grayscale(img):
        """Convert RGB to Grayscale."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def split_channels(img):
        """Split into R, G, B channels (returns dict of 3 BGR images)."""
        b, g, r = cv2.split(img)
        zeros = np.zeros_like(b)
        r_img = cv2.merge([zeros, zeros, r])
        g_img = cv2.merge([zeros, g, zeros])
        b_img = cv2.merge([b, zeros, zeros])
        return {'red': r_img, 'green': g_img, 'blue': b_img}

    @staticmethod
    def adjust_hue_saturation(img, hue_shift=0, sat_scale=1.0):
        """Adjust hue and saturation in HSV space."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
        hsv[:, :, 0] = (hsv[:, :, 0] + float(hue_shift)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(sat_scale), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # =========================================================================
    # IMAGE SEGMENTATION
    # =========================================================================

    @staticmethod
    def segment_threshold(img, thresh=128):
        """Threshold-based segmentation with masking."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, mask = cv2.threshold(gray, int(thresh), 255, cv2.THRESH_BINARY)
        if len(img.shape) == 3:
            result = cv2.bitwise_and(img, img, mask=mask)
            return result
        return mask

    @staticmethod
    def segment_edge(img):
        """Edge-based segmentation using Canny + contour filling."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, -1)
        if len(img.shape) == 3:
            return cv2.bitwise_and(img, img, mask=mask)
        return mask

    @staticmethod
    def segment_region(img, num_regions=4):
        """Region-based segmentation using K-means clustering."""
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = max(2, min(int(num_regions), 16))
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        centers = centers.astype(np.uint8)
        result = centers[labels.flatten()].reshape(img.shape)
        return result

    # =========================================================================
    # HISTOGRAM ANALYSIS
    # =========================================================================

    @staticmethod
    def compute_histogram(img):
        """Compute histogram data for grayscale and RGB channels."""
        result = {}
        if len(img.shape) == 3:
            # RGB histograms
            colors = ('blue', 'green', 'red')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                result[color] = hist.flatten().tolist()
            # Grayscale histogram
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            result['gray'] = hist.flatten().tolist()
        else:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            result['gray'] = hist.flatten().tolist()
        return result
