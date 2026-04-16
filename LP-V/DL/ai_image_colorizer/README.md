# Deep Learning Image Colorizer (Streamlit)

This project provides an end-to-end image colorization web app with:

- Custom U-Net (PyTorch): L channel -> predicted a,b channels in LAB color space

## Project Structure

- model.py: U-Net architecture and model loading
- utils.py: preprocessing, postprocessing, and inference utilities
- app.py: Streamlit user interface

## Features

- Upload JPG/PNG image
- LAB workflow: RGB -> LAB, infer a,b from L, LAB -> RGB
- Before/After side-by-side display
- Model section and DL explanation in UI
- Loading spinner during inference
- Download colorized image
- Optional feature-map visualization
- Optional model summary in UI

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Place trained checkpoint at models/unet_colorizer.pth to use pre-trained weights.
- If the checkpoint is missing, the app still runs with default model initialization for architecture demonstration.
 - Place trained checkpoint at models/unet_colorizer.pth to use pre-trained weights.
 - OpenCV DNN model files (optional) — if you want to use the OpenCV pretrained baseline, place these files in the `models/` directory:
	 - `colorization_deploy_v2.prototxt`
	 - `colorization_release_v2.caffemodel`
	 - `pts_in_hull.npy`

	The app provides an upload option (in the UI) to save these files into `models/` if they are not present. Automatic downloads are intentionally disabled because public hosting for the caffemodel is unreliable.

## Obtaining the OpenCV DNN files

- `colorization_deploy_v2.prototxt` and `pts_in_hull.npy` are typically available in the original project repositories (search for `colorization_deploy_v2.prototxt` and `pts_in_hull.npy`).
- `colorization_release_v2.caffemodel` may not be consistently hosted; if you have a copy from the original author or a mirror, place it in `models/`.

If you cannot obtain the OpenCV files, use the built-in `U-Net (PyTorch)` option which runs the U-Net architecture (place a checkpoint at `models/unet_colorizer.pth` to use trained weights).
