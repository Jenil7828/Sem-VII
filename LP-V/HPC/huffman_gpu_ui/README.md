# Huffman Encoding on GPU (Mini Project)

This mini project implements Huffman text compression with GPU acceleration and a simple web UI.

## Stack
- Python
- Streamlit (UI)
- Numba CUDA kernel (GPU symbol-to-code mapping)
- NumPy (tables and bit-packing support)

## Project Structure
- `app.py`: Streamlit web app
- `huffman_gpu.py`: Huffman logic + GPU kernel + packing/decompression utilities
- `requirements.txt`: Python dependencies

## How It Works
1. Input text is converted to UTF-8 bytes.
2. A Huffman codebook is built from character frequencies on CPU.
3. GPU kernel maps each byte to code bits and code length in parallel.
4. CPU packs variable-length codes into compressed byte stream.
5. App verifies decoding to ensure lossless output.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes
- If CUDA is not available, the app automatically uses CPU fallback for symbol encoding.
- The project is intentionally educational and focuses on a clear end-to-end pipeline.
