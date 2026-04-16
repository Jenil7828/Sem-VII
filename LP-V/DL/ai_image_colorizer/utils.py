from pathlib import Path
from typing import Dict, Optional

import urllib.request
import urllib.error
import cv2
import numpy as np
import torch

from model import UNetColorizer


def load_opencv_dnn_colorization_model(model_dir: str) -> cv2.dnn_Net:
    """Load OpenCV colorization DNN from local model files."""
    model_root = Path(model_dir)
    model_path = model_root / "colorization_release_v2.caffemodel"
    proto_path = model_root / "colorization_deploy_v2.prototxt"
    pts_path = model_root / "pts_in_hull.npy"

    # Return None instead of raising so the caller can handle missing files gracefully.
    if not model_path.exists() or not proto_path.exists() or not pts_path.exists():
        return None

    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    pts_in_hull = np.load(str(pts_path)).transpose().reshape(2, 313, 1, 1)

    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

    return net


def models_exist(model_dir: str) -> bool:
    """Check whether the required OpenCV DNN model files exist in model_dir."""
    model_root = Path(model_dir)
    return (
        (model_root / "colorization_release_v2.caffemodel").exists()
        and (model_root / "colorization_deploy_v2.prototxt").exists()
        and (model_root / "pts_in_hull.npy").exists()
    )


def save_uploaded_model_files(
    prototxt_file, caffemodel_file, pts_file, model_dir: str
) -> None:
    """Save uploaded model files (Streamlit UploadedFile) into model_dir."""
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    if prototxt_file is not None:
        prototxt_path = model_root / "colorization_deploy_v2.prototxt"
        with open(prototxt_path, "wb") as f:
            f.write(prototxt_file.getbuffer())

    if caffemodel_file is not None:
        caffemodel_path = model_root / "colorization_release_v2.caffemodel"
        with open(caffemodel_path, "wb") as f:
            f.write(caffemodel_file.getbuffer())

    if pts_file is not None:
        pts_path = model_root / "pts_in_hull.npy"
        with open(pts_path, "wb") as f:
            f.write(pts_file.getbuffer())

# Candidate download URLs. prototxt and pts are generally available; caffemodel may require mirrors.
DEFAULT_MODEL_URLS = {
    "prototxt": [
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    ],
    "pts": [
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy",
    ],
    "caffemodel": [
        "https://github.com/dath1s/colorizor/blob/main/colorization_release_v2.caffemodel",
        "https://github.com/richzhang/colorization/raw/caffe/colorization/models/colorization_release_v2.caffemodel",
        "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_release_v2.caffemodel",
        "http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
    ],
}


def _download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Download a single file to dest. Raises RuntimeError on failure."""
    try:
        urllib.request.urlretrieve(url, str(dest))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code} when downloading {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error {e.reason} when downloading {url}")
    except Exception as e:
        raise RuntimeError(f"Failed to download {url}: {e}")


def auto_download_models(model_dir: str) -> Dict[str, Dict[str, Optional[str]]]:
    """Attempt to automatically download missing model files into model_dir.

    Returns a dict with keys 'prototxt','caffemodel','pts' holding {'ok':bool,'url' or 'error':...}
    """
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Optional[str]]] = {}

    targets = {
        "prototxt": model_root / "colorization_deploy_v2.prototxt",
        "pts": model_root / "pts_in_hull.npy",
        "caffemodel": model_root / "colorization_release_v2.caffemodel",
    }

    for key, dest in targets.items():
        if dest.exists():
            results[key] = {"ok": True, "url": str(dest)}
            continue

        urls = DEFAULT_MODEL_URLS.get(key, [])
        ok = False
        last_err = None
        for u in urls:
            try:
                _download_file(u, dest)
                results[key] = {"ok": True, "url": u}
                ok = True
                break
            except Exception as e:
                last_err = str(e)

        if not ok:
            results[key] = {"ok": False, "error": last_err}

    return results


def download_from_url(url: str, dest_path: str) -> None:
    """Download a single file from user-supplied URL to dest_path.

    Raises RuntimeError on failure.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _download_file(url, dest)


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode the uploaded image.")
    return image




def preprocess_for_unet(image_bgr: np.ndarray, input_size: int = 256) -> Dict[str, np.ndarray]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    rgb_float = rgb.astype(np.float32) / 255.0
    lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]

    l_resized = cv2.resize(l_channel, (input_size, input_size), interpolation=cv2.INTER_AREA)
    l_norm = (l_resized / 50.0) - 1.0

    return {
        "rgb": rgb,
        "gray": gray,
        "l_original": l_channel,
        "l_norm": l_norm,
        "original_size": np.array([rgb.shape[1], rgb.shape[0]], dtype=np.int32),
    }


def postprocess_from_ab(l_original: np.ndarray, ab_pred: np.ndarray) -> np.ndarray:
    h, w = l_original.shape
    ab_resized = cv2.resize(ab_pred, (w, h), interpolation=cv2.INTER_CUBIC)

    lab_out = np.zeros((h, w, 3), dtype=np.float32)
    lab_out[:, :, 0] = l_original
    lab_out[:, :, 1:] = ab_resized

    rgb_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)
    rgb_out = np.clip(rgb_out, 0, 1)
    return (rgb_out * 255).astype(np.uint8)


def infer_with_unet(
    model: UNetColorizer, image_bgr: np.ndarray, device: torch.device, return_features: bool = False
):
    data = preprocess_for_unet(image_bgr)
    l_tensor = torch.from_numpy(data["l_norm"]).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if return_features:
            ab_pred_norm, features = model(l_tensor, return_features=True)
        else:
            ab_pred_norm = model(l_tensor)
            features = None

    ab_pred = ab_pred_norm.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ab_pred = ab_pred * 110.0

    colorized_rgb = postprocess_from_ab(data["l_original"], ab_pred)

    return data["gray"], colorized_rgb, features


def colorize_with_opencv_dnn(
    image_bgr: np.ndarray,
    net: cv2.dnn_Net,
    ab_scale: float = 1.0,
    enhance_contrast: bool = False,
) -> Dict[str, np.ndarray]:
    """OpenCV DNN colorization with corrected LAB pipeline and diagnostics."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Keep float32 in [0, 1] before LAB conversion.
    img_float = gray_bgr.astype(np.float32) / 255.0
    img_lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)

    l_original = img_lab[:, :, 0]
    h, w = l_original.shape

    # Model expects 224x224 L channel centered by mean 50.
    l_resized = cv2.resize(l_original, (224, 224), interpolation=cv2.INTER_CUBIC)
    l_centered = l_resized - 50.0
    blob = cv2.dnn.blobFromImage(l_centered)

    net.setInput(blob)
    ab_pred = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Scale chroma channels for user-controlled color intensity.
    ab_pred = ab_pred * float(ab_scale)
    ab_up = cv2.resize(ab_pred, (w, h), interpolation=cv2.INTER_CUBIC)

    lab_out = np.zeros((h, w, 3), dtype=np.float32)
    lab_out[:, :, 0] = l_original
    lab_out[:, :, 1:] = ab_up

    bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
    bgr_out = np.clip(bgr_out, 0.0, 1.0)

    if enhance_contrast:
        # Mild contrast boost to improve punch without oversmoothing detail.
        out_lab = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        out_lab[:, :, 0] = clahe.apply((out_lab[:, :, 0] * 255).astype(np.uint8)).astype(np.float32) / 255.0
        bgr_out = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
        bgr_out = np.clip(bgr_out, 0.0, 1.0)

    colorized_bgr = (bgr_out * 255).astype(np.uint8)
    colorized_rgb = cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB)

    # Diagnostics for sepia/low-diversity debugging.
    a_vals = ab_pred[:, :, 0]
    b_vals = ab_pred[:, :, 1]
    stats = {
        "a_min": float(np.min(a_vals)),
        "a_max": float(np.max(a_vals)),
        "a_mean": float(np.mean(a_vals)),
        "a_std": float(np.std(a_vals)),
        "b_min": float(np.min(b_vals)),
        "b_max": float(np.max(b_vals)),
        "b_mean": float(np.mean(b_vals)),
        "b_std": float(np.std(b_vals)),
    }

    gray_lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
    color_lab = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2LAB)

    comparison = {
        "gray_a_mean": float(np.mean(gray_lab[:, :, 1])),
        "gray_b_mean": float(np.mean(gray_lab[:, :, 2])),
        "color_a_mean": float(np.mean(color_lab[:, :, 1])),
        "color_b_mean": float(np.mean(color_lab[:, :, 2])),
        "color_a_std": float(np.std(color_lab[:, :, 1])),
        "color_b_std": float(np.std(color_lab[:, :, 2])),
    }

    return {
        "gray": gray,
        "colorized_rgb": colorized_rgb,
        "ab_pred_small": ab_pred,
        "ab_stats": stats,
        "lab_compare": comparison,
    }
