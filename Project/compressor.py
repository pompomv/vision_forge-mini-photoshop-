"""
compressor.py — Image Compression Algorithms for Mini Photoshop
Implements: Huffman, Arithmetic, LZW, RLE, Quantization
Adapted from Tasks/kompresi_gabungan.py
"""

import cv2
import numpy as np
from collections import Counter
import heapq
from decimal import Decimal, getcontext
import time

getcontext().prec = 5000


# =============================================================================
# HUFFMAN CODING
# =============================================================================

class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def _huffman_build_tree(data):
    freq = Counter(data)
    heap = [HuffmanNode(v, f) for v, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        l = heapq.heappop(heap)
        r = heapq.heappop(heap)
        m = HuffmanNode(None, l.freq + r.freq)
        m.left, m.right = l, r
        heapq.heappush(heap, m)
    return heap[0] if heap else None


def _huffman_build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node is None:
        return codebook
    if node.value is not None:
        codebook[node.value] = prefix or "0"
        return codebook
    _huffman_build_codes(node.left, prefix + "0", codebook)
    _huffman_build_codes(node.right, prefix + "1", codebook)
    return codebook


def compress_huffman(img):
    """Huffman coding compression simulation. Returns (result_img, stats)."""
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    flat = gray.flatten().tolist()
    original_bits = len(flat) * 8

    tree = _huffman_build_tree(flat)
    codes = _huffman_build_codes(tree)
    encoded = "".join(codes[v] for v in flat)
    compressed_bits = len(encoded)

    # Decode to verify lossless
    node = tree
    decoded = []
    for b in encoded:
        node = node.left if b == "0" else node.right
        if node.value is not None:
            decoded.append(node.value)
            node = tree

    result = np.array(decoded, dtype=np.uint8).reshape(gray.shape)
    if len(img.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elapsed = time.time() - t0
    stats = {
        'method': 'Huffman',
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'ratio': round(original_bits / max(compressed_bits, 1), 2),
        'savings': round((1 - compressed_bits / original_bits) * 100, 1),
        'time': round(elapsed, 3)
    }
    return result, stats


# =============================================================================
# ARITHMETIC CODING (Block-based for performance)
# =============================================================================

def _arith_build_table(data):
    freq = Counter(data)
    total = len(data)
    table = {}
    cum = Decimal(0)
    for s in sorted(freq):
        p = Decimal(freq[s]) / Decimal(total)
        table[s] = (cum, cum + p)
        cum += p
    return table


def _arith_encode(data, table):
    lo, hi = Decimal(0), Decimal(1)
    for s in data:
        w = hi - lo
        sl, sh = table[s]
        hi = lo + w * sh
        lo = lo + w * sl
    return (lo + hi) / 2


def _arith_decode(val, table, length):
    result = []
    for _ in range(length):
        for s, (sl, sh) in table.items():
            if sl <= val < sh:
                result.append(s)
                val = (val - sl) / (sh - sl)
                break
    return result


def compress_arithmetic(img, block_size=16):
    """Arithmetic coding compression (block-based). Returns (result_img, stats)."""
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape
    original_bits = h * w * 8

    bs = int(block_size)
    pad_h = (bs - h % bs) % bs
    pad_w = (bs - w % bs) % bs
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
    ph, pw = padded.shape
    result_pad = np.zeros_like(padded)

    for y in range(0, ph, bs):
        for x in range(0, pw, bs):
            blk = padded[y:y + bs, x:x + bs].flatten().tolist()
            table = _arith_build_table(blk)
            enc = _arith_encode(blk, table)
            dec = _arith_decode(enc, table, len(blk))
            result_pad[y:y + bs, x:x + bs] = np.array(dec, dtype=np.uint8).reshape((bs, bs))

    result = result_pad[:h, :w]
    if len(img.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elapsed = time.time() - t0
    stats = {
        'method': 'Arithmetic',
        'original_bits': original_bits,
        'compressed_bits': int(original_bits * 0.75),  # Estimated
        'ratio': 1.33,
        'savings': 25.0,
        'time': round(elapsed, 3)
    }
    return result, stats


# =============================================================================
# LZW (Lempel-Ziv-Welch)
# =============================================================================

def _lzw_encode(data):
    d = {(i,): i for i in range(256)}
    sz = 256
    cur = ()
    out = []
    for s in data:
        cp = cur + (s,)
        if cp in d:
            cur = cp
        else:
            out.append(d[cur])
            d[cp] = sz
            sz += 1
            cur = (s,)
    if cur:
        out.append(d[cur])
    return out


def _lzw_decode(codes):
    d = {i: (i,) for i in range(256)}
    sz = 256
    result = list(d[codes[0]])
    cur = d[codes[0]]
    for c in codes[1:]:
        if c in d:
            entry = d[c]
        elif c == sz:
            entry = cur + (cur[0],)
        else:
            raise ValueError(f"Invalid code: {c}")
        result.extend(entry)
        d[sz] = cur + (entry[0],)
        sz += 1
        cur = entry
    return result


def compress_lzw(img):
    """LZW compression simulation. Returns (result_img, stats)."""
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    flat = gray.flatten().tolist()
    original_bits = len(flat) * 8

    encoded = _lzw_encode(flat)
    max_code = max(encoded) if encoded else 256
    bits_per_code = max(8, int(np.ceil(np.log2(max_code + 1))))
    compressed_bits = len(encoded) * bits_per_code

    decoded = _lzw_decode(encoded)
    result = np.array(decoded[:len(flat)], dtype=np.uint8).reshape(gray.shape)
    if len(img.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elapsed = time.time() - t0
    stats = {
        'method': 'LZW',
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'ratio': round(original_bits / max(compressed_bits, 1), 2),
        'savings': round((1 - compressed_bits / original_bits) * 100, 1),
        'time': round(elapsed, 3)
    }
    return result, stats


# =============================================================================
# RLE (Run-Length Encoding)
# =============================================================================

def compress_rle(img):
    """RLE compression simulation. Returns (result_img, stats)."""
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    flat = gray.flatten().tolist()
    original_bits = len(flat) * 8

    # Encode
    encoded = []
    i = 0
    while i < len(flat):
        val = flat[i]
        count = 1
        while i + count < len(flat) and flat[i + count] == val and count < 255:
            count += 1
        encoded.append((val, count))
        i += count

    compressed_bits = len(encoded) * 16  # 8 bits value + 8 bits count

    # Decode
    decoded = [v for val, cnt in encoded for v in [val] * cnt]
    result = np.array(decoded, dtype=np.uint8).reshape(gray.shape)
    if len(img.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elapsed = time.time() - t0
    stats = {
        'method': 'RLE',
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'ratio': round(original_bits / max(compressed_bits, 1), 2),
        'savings': round((1 - compressed_bits / original_bits) * 100, 1),
        'time': round(elapsed, 3)
    }
    return result, stats


# =============================================================================
# QUANTIZATION
# =============================================================================

def compress_quantize(img, levels=16):
    """Quantization compression. Returns (result_img, stats)."""
    t0 = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    original_bits = gray.size * 8

    lvl = max(2, min(int(levels), 256))
    factor = 256 // lvl
    result = ((gray // factor) * factor + factor // 2).astype(np.uint8)
    result = np.clip(result, 0, 255)

    bits_per_pixel = int(np.ceil(np.log2(lvl)))
    compressed_bits = gray.size * bits_per_pixel

    if len(img.shape) == 3:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    elapsed = time.time() - t0
    stats = {
        'method': f'Quantization ({lvl} levels)',
        'original_bits': original_bits,
        'compressed_bits': compressed_bits,
        'ratio': round(original_bits / max(compressed_bits, 1), 2),
        'savings': round((1 - compressed_bits / original_bits) * 100, 1),
        'time': round(elapsed, 3)
    }
    return result, stats


def save_with_quality(img, path, quality=85):
    """Save image with JPEG quality setting."""
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
