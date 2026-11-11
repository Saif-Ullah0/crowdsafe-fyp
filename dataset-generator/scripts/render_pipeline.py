"""
CrowdSafe Synthetic Simulation — Dual-Mode Render Post-Processing Pipeline
Provides two rendering modes:
- Mode 1: CCTV Mode (Default)
    * Subtle wide-angle barrel lens distortion (k1 ≈ -0.05)
    * Gaussian read noise & Poisson shot noise (sensor grain)
    * DVR H.264 / JPEG compression artifacts (macroblocking)
    * Frame-to-frame exposure and fluorescent color temperature jitter
    * Subtle chromatic aberration at frame edges
- Mode 2: Clean Mode
    * Direct raw 3D rasterized frame without optical degradation
"""

import os
import sys
import math
import random
import argparse
from typing import Optional, Tuple, Union
import numpy as np


def apply_lens_distortion(img: np.ndarray, k1: float = -0.06, k2: float = 0.01) -> np.ndarray:
    """
    Applies subtle radial barrel lens distortion simulating wide-angle CCTV optics.
    Pure NumPy implementation with bilinear interpolation.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Normalized meshgrid [-1, 1]
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    x_norm = (x_idx - cx) / cx
    y_norm = (y_idx - cy) / cy
    r2 = x_norm ** 2 + y_norm ** 2

    # Radial distortion polynomial: r_dist = r * (1 + k1*r^2 + k2*r^4)
    distortion = 1.0 + k1 * r2 + k2 * (r2 ** 2)
    src_x = cx + x_norm * distortion * cx
    src_y = cy + y_norm * distortion * cy

    # Bilinear interpolation
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    x0 = np.floor(src_x).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(src_y).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)

    wx = src_x - x0
    wy = src_y - y0

    output = np.zeros_like(img)
    for c in range(img.shape[2]):
        ia = img[y0, x0, c]
        ib = img[y1, x0, c]
        ic = img[y0, x1, c]
        id_ = img[y1, x1, c]

        top = ia + (ic - ia) * wx
        bottom = ib + (id_ - ib) * wx
        output[:, :, c] = top + (bottom - top) * wy

    return output


def apply_chromatic_aberration(img: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """Applies subtle radial dispersion on R and B channels toward the frame edges."""
    h, w, c = img.shape
    cx, cy = w / 2.0, h / 2.0
    y_idx, x_idx = np.indices((h, w), dtype=np.float32)
    r = np.sqrt(((x_idx - cx) / cx) ** 2 + ((y_idx - cy) / cy) ** 2)

    # Shift Red slightly outward and Blue slightly inward
    shift_r = (strength * (r ** 1.5)).astype(int)
    
    out = img.copy()
    # Red channel outward shift
    out[:, :, 0] = np.roll(out[:, :, 0], shift=1, axis=1)
    # Blue channel inward shift
    out[:, :, 2] = np.roll(out[:, :, 2], shift=-1, axis=1)
    return out


def apply_sensor_noise(img: np.ndarray, sigma: float = 6.0, seed: Optional[int] = None) -> np.ndarray:
    """Adds combined Gaussian read noise and Poisson-like shot noise."""
    if seed is not None:
        np.random.seed(seed)
    
    # Gaussian electronic read noise
    gaussian_noise = np.random.normal(0, sigma, img.shape)
    
    # Signal-dependent photon shot noise (stronger in brighter pixels)
    shot_scale = np.sqrt(np.maximum(img.astype(np.float32), 1.0)) * 0.4
    shot_noise = np.random.normal(0, 1.0, img.shape) * shot_scale

    noisy = img.astype(np.float32) + gaussian_noise + shot_noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_exposure_and_tint_jitter(img: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """Simulates realistic surveillance auto-exposure fluctuation and fluorescent tint."""
    # Low frequency sine wave + subtle random jitter
    jitter_factor = 1.0 + 0.03 * math.sin(frame_idx * 0.15) + np.random.uniform(-0.015, 0.015)
    
    # Slight cool fluorescent green/cyan tint typical of commercial CCTV
    tint = np.array([0.97, 1.01, 0.98], dtype=np.float32)
    
    adjusted = img.astype(np.float32) * jitter_factor * tint
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_dvr_compression_artifacts(img: np.ndarray, quality: int = 68) -> np.ndarray:
    """Simulates security DVR H.264/JPEG macroblocking and quantization artifacts."""
    try:
        import cv2
        # Use OpenCV JPEG encode/decode for authentic 8x8 DCT compression artifacts
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, encoded = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        if success:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass

    try:
        from PIL import Image
        import io
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer))
    except ImportError:
        # Fallback 8x8 block-averaging simulation in pure NumPy
        h, w, c = img.shape
        block_size = 8
        h_pad = (block_size - (h % block_size)) % block_size
        w_pad = (block_size - (w % block_size)) % block_size
        padded = np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='edge')
        
        # Subtle quantization step
        quantized = (padded // 6) * 6
        return quantized[:h, :w, :]


def apply_cctv_pipeline(
    img_rgb: np.ndarray,
    frame_idx: int = 0,
    lens_distort: bool = True,
    sensor_noise: bool = True,
    compression: bool = True,
    lighting_jitter: bool = True
) -> np.ndarray:
    """
    Applies the full composite CCTV optical degradation pipeline to an RGB image.
    """
    frame = img_rgb.copy()

    # 1. Exposure and fluorescent tint jitter
    if lighting_jitter:
        frame = apply_exposure_and_tint_jitter(frame, frame_idx=frame_idx)

    # 2. Barrel lens distortion (6mm wide-angle optics)
    if lens_distort:
        frame = apply_lens_distortion(frame, k1=-0.05, k2=0.008)

    # 3. Chromatic dispersion
    frame = apply_chromatic_aberration(frame, strength=1.2)

    # 4. Sensor noise (read noise + shot noise)
    if sensor_noise:
        frame = apply_sensor_noise(frame, sigma=4.5)

    # 5. DVR video compression artifacts
    if compression:
        frame = apply_dvr_compression_artifacts(frame, quality=68)

    return frame


def process_rendered_frame(
    input_image_path: str,
    output_image_path: str,
    mode: str = "cctv",
    frame_idx: int = 0
) -> str:
    """
    Reads an image from disk, applies the specified mode ('cctv' or 'clean'),
    and saves to output_image_path.
    Supports Pillow, OpenCV, and Blender's bpy.data.images transparently.
    """
    img_arr = None
    loaded_via_bpy = False
    bpy_img = None

    try:
        from PIL import Image
        img = Image.open(input_image_path).convert("RGB")
        img_arr = np.array(img)
    except ImportError:
        try:
            import cv2
            bgr = cv2.imread(input_image_path)
            if bgr is not None:
                img_arr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except ImportError:
            pass

    if img_arr is None:
        try:
            import bpy
            bpy_img = bpy.data.images.load(os.path.abspath(input_image_path))
            w, h = bpy_img.size[0], bpy_img.size[1]
            rgba = np.array(bpy_img.pixels[:], dtype=np.float32).reshape((h, w, 4))
            img_arr = (rgba[:, :, :3] * 255.0).astype(np.uint8)
            loaded_via_bpy = True
        except Exception as e:
            raise RuntimeError(f"Unable to load image {input_image_path} (Pillow/OpenCV/bpy missing or failed: {e})")

    if mode.lower() == "cctv":
        processed = apply_cctv_pipeline(img_arr, frame_idx=frame_idx)
    else:  # 'clean' mode
        processed = img_arr

    os.makedirs(os.path.dirname(os.path.abspath(output_image_path)), exist_ok=True)

    if loaded_via_bpy and bpy_img is not None:
        import bpy
        h, w = processed.shape[:2]
        rgba_out = np.dstack([processed.astype(np.float32) / 255.0, np.ones((h, w, 1), dtype=np.float32)])
        bpy_img.pixels = rgba_out.ravel()
        bpy_img.filepath_raw = os.path.abspath(output_image_path)
        bpy_img.file_format = 'PNG'
        bpy_img.save()
        bpy.data.images.remove(bpy_img)
    else:
        try:
            from PIL import Image
            out_pil = Image.fromarray(processed)
            out_pil.save(output_image_path)
        except ImportError:
            import cv2
            cv2.imwrite(output_image_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))

    return output_image_path


def main():
    argv = sys.argv
    if "--" in argv:
        script_args = argv[argv.index("--") + 1:]
    else:
        script_args = argv[1:]

    parser = argparse.ArgumentParser(description="Dual-Mode CCTV / Clean Image Post-Processor.")
    parser.add_argument("--input", required=True, help="Input raw image path")
    parser.add_argument("--output", required=True, help="Output processed image path")
    parser.add_argument("--mode", default="cctv", choices=["cctv", "clean"], help="Render mode: cctv (default) or clean")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame sequence index")
    args = parser.parse_args(script_args)

    process_rendered_frame(args.input, args.output, mode=args.mode, frame_idx=args.frame_idx)
    print(f">>> Processed [{args.mode.upper()}] frame saved to: {args.output}")


if __name__ == "__main__":
    main()
