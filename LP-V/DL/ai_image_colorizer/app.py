from pathlib import Path

import cv2
import pandas as pd
import numpy as np
import streamlit as st
import torch

from model import load_unet_model, summarize_model
from utils import (
    colorize_with_opencv_dnn,
    decode_uploaded_image,
    infer_with_unet,
    load_opencv_dnn_colorization_model,
    models_exist,
    save_uploaded_model_files,
    auto_download_models,
    download_from_url,
)


st.set_page_config(page_title="Deep Learning Image Colorizer", layout="wide")


@st.cache_resource(show_spinner=False)
def get_unet_model(weights_path: str, device_name: str):
    device = torch.device(device_name)
    model, loaded_pretrained = load_unet_model(Path(weights_path), device)
    return model, loaded_pretrained


def get_opencv_net(model_dir: str):
    # No caching here so users can upload files and reload model during runtime
    return load_opencv_dnn_colorization_model(model_dir)


def render_feature_maps(features) -> None:
    if not features:
        return

    st.markdown("#### Intermediate Feature Maps")
    cols = st.columns(len(features))

    for i, (name, fmap) in enumerate(features.items()):
        fmap_2d = fmap[0, 0].detach().cpu().numpy()
        fmap_2d = (fmap_2d - fmap_2d.min()) / (fmap_2d.max() - fmap_2d.min() + 1e-8)
        fmap_u8 = (fmap_2d * 255).astype(np.uint8)

        with cols[i]:
            st.image(fmap_u8, caption=name, width="stretch", clamp=True)


def main() -> None:
    st.title("Deep Learning Image Colorizer")
    st.write(
        "This app colorizes grayscale images in LAB space. "
        "For OpenCV DNN, the L channel is centered by 50 before inference and predicted a,b channels "
        "are merged back with original L for reconstruction."
    )

    st.markdown("### Model Used")
    model_name = st.radio("Select model", ["OpenCV DNN (Fixed Pipeline)", "U-Net (PyTorch)"], horizontal=True)

    st.info(
        "DL approach: We convert image RGB->LAB, keep the L channel as input, then estimate a,b channels "
        "with a CNN model. Finally, L + predicted (a,b) is converted back to RGB."
    )

    with st.sidebar:
        st.markdown("### Color Controls")
        color_intensity = st.slider("Color Intensity (a,b scale)", min_value=0.5, max_value=2.0, value=1.1, step=0.1)
        enhance_contrast = st.checkbox("Enhance Contrast (CLAHE)", value=False)
        show_hist = st.checkbox("Show a,b histograms", value=True)
        show_lab_compare = st.checkbox("Show LAB comparison", value=True)

    uploaded_file = st.file_uploader("Upload grayscale image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.warning("Please upload an image to start.")
        return

    try:
        image_bgr = decode_uploaded_image(uploaded_file)
        preview_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        preview_gray_rgb = cv2.cvtColor(preview_gray, cv2.COLOR_GRAY2RGB)

        st.markdown("### Input Preview")
        st.image(preview_gray_rgb, caption="Input Grayscale", width="stretch")

        if st.button("Colorize Image", type="primary"):
            with st.spinner("Running deep learning inference..."):
                if model_name == "OpenCV DNN (Fixed Pipeline)":
                    model_dir = Path(__file__).resolve().parent / "models"
                    net = get_opencv_net(str(model_dir))

                    if net is None:
                        st.warning(
                            "OpenCV DNN model files not found in the project's `models/` folder."
                        )

                        # Offer automatic download attempt
                        attempt_auto = st.checkbox("Attempt automatic download of missing model files", value=True)
                        if attempt_auto:
                            with st.spinner("Attempting to download model files..."):
                                try:
                                    results = auto_download_models(str(model_dir))
                                    st.json(results)
                                    all_ok = all(results[k]["ok"] for k in ("prototxt", "pts", "caffemodel"))
                                    if all_ok:
                                        st.success("All model files downloaded — loading model now...")
                                        net = get_opencv_net(str(model_dir))
                                        if net is None:
                                            st.error("Downloaded files but failed to load the model. The files may be corrupted.")
                                        else:
                                            st.success("Model loaded successfully — continuing inference.")
                                    else:
                                        st.error("Auto-download incomplete. See results and either provide a direct URL or upload missing files.")
                                except Exception as e:
                                    st.error(f"Auto-download failed: {e}")

                        # If automatic download did not yield a model, offer manual upload and direct-URL download
                        if net is None:
                            with st.expander("Provide or upload model files (recommended)", expanded=True):
                                st.markdown(
                                    "Upload the three model files: `colorization_deploy_v2.prototxt`, `colorization_release_v2.caffemodel`, and `pts_in_hull.npy`."
                                )
                                proto_up = st.file_uploader("Upload prototxt (colorization_deploy_v2.prototxt)")
                                caff_up = st.file_uploader("Upload caffemodel (colorization_release_v2.caffemodel)")
                                pts_up = st.file_uploader("Upload pts_in_hull.npy")

                                if st.button("Save model files to models/ and load"):
                                    try:
                                        save_uploaded_model_files(proto_up, caff_up, pts_up, str(model_dir))
                                        st.success("Saved files to models/. Attempting to load...")
                                        net = get_opencv_net(str(model_dir))
                                        if net is not None:
                                            st.success("Model loaded successfully — you can now press Colorize again.")
                                        else:
                                            st.error("Saved files but failed to load the model. Ensure files are correct and try again.")
                                    except Exception as e:
                                        st.error(f"Failed to save files: {e}")

                                st.markdown("---")
                                st.markdown("Or provide a direct URL to the caffemodel file (if you have one):")
                                caff_url = st.text_input("Direct URL to caffemodel (http/https)")
                                if st.button("Download caffemodel from URL") and caff_url:
                                    with st.spinner("Downloading caffemodel from provided URL..."):
                                        try:
                                            download_from_url(caff_url, str(model_dir / "colorization_release_v2.caffemodel"))
                                            st.success("Downloaded caffemodel — attempting to load model...")
                                            net = get_opencv_net(str(model_dir))
                                            if net is not None:
                                                st.success("Model loaded successfully — you can now press Colorize again.")
                                            else:
                                                st.error("Downloaded caffemodel but model failed to load. File may be invalid.")
                                        except Exception as e:
                                            st.error(f"Failed to download caffemodel: {e}")

                            st.info(
                                "After placing or uploading the files, press Colorize again to load and run the model.\n"
                                "If you don't have the files, use the U-Net model option instead."
                            )
                            # Abort inference for this run; user must re-run after adding files
                            st.stop()

                    # At this point net is available
                    result = colorize_with_opencv_dnn(
                        image_bgr=image_bgr,
                        net=net,
                        ab_scale=float(color_intensity),
                        enhance_contrast=enhance_contrast,
                    )
                    gray = result["gray"]
                    colorized_rgb = result["colorized_rgb"]

                    st.markdown("### a,b Prediction Debug")
                    st.json(result["ab_stats"])

                    if show_hist:
                        ab_small = result["ab_pred_small"]
                        a_flat = ab_small[:, :, 0].reshape(-1)
                        b_flat = ab_small[:, :, 1].reshape(-1)

                        hist_a, bins_a = np.histogram(a_flat, bins=40)
                        hist_b, bins_b = np.histogram(b_flat, bins=40)

                        hist_df = pd.DataFrame(
                            {
                                "bin_center": (bins_a[:-1] + bins_a[1:]) / 2.0,
                                "a_channel": hist_a,
                                "b_channel": hist_b,
                            }
                        )
                        st.line_chart(hist_df.set_index("bin_center")[ ["a_channel", "b_channel"] ])

                    if show_lab_compare:
                        st.markdown("### LAB Comparison (Input Gray vs Colorized)")
                        st.json(result["lab_compare"])

                else:
                    device_name = "cuda" if torch.cuda.is_available() else "cpu"
                    weights_path = Path(__file__).resolve().parent / "models" / "unet_colorizer.pth"
                    model, loaded_pretrained = get_unet_model(str(weights_path), device_name)

                    gray, colorized_rgb, features = infer_with_unet(
                        model=model,
                        image_bgr=image_bgr,
                        device=torch.device(device_name),
                        return_features=True,
                    )

                    status_msg = (
                        "Loaded pre-trained U-Net weights."
                        if loaded_pretrained
                        else "No U-Net checkpoint found. Running architecture with default initialization."
                    )
                    st.caption(status_msg)

                    with st.expander("Model Summary", expanded=False):
                        st.code(summarize_model(model), language="text")

                    with st.expander("Feature Maps", expanded=False):
                        render_feature_maps(features)

            st.markdown("### Before vs After")
            col1, col2 = st.columns(2)

            with col1:
                gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                st.image(gray_rgb, caption="Original Grayscale", width="stretch")

            with col2:
                st.image(colorized_rgb, caption="Colorized Output", width="stretch")

            success, encoded = cv2.imencode(".png", cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2BGR))
            if success:
                st.download_button(
                    label="Download Colorized Image",
                    data=encoded.tobytes(),
                    file_name="colorized_output.png",
                    mime="image/png",
                )
            else:
                st.error("Could not prepare image for download.")

    except Exception as exc:
        st.error(f"Error: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
