import streamlit as st

from huffman_gpu import build_export_payload, codebook_rows, compress_text, decompress_text

st.set_page_config(page_title="GPU Huffman Encoder", page_icon="H", layout="wide")

st.title("Mini Project: Huffman Encoding on GPU")
st.caption("Built with Streamlit + CUDA (Numba). If CUDA is unavailable, it automatically falls back to CPU.")

left, right = st.columns([3, 2])

with left:
    sample = """GPU-based Huffman compression demo text. Change this input and click Encode."""
    text = st.text_area("Input text", value=sample, height=220)
    encode_clicked = st.button("Encode", type="primary", use_container_width=True)

with right:
    st.subheader("How it works")
    st.markdown(
        "\n".join(
            [
                "1. Build Huffman tree and variable-length codebook on CPU.",
                "2. Send input bytes and lookup tables to GPU.",
                "3. Parallel kernel maps each symbol to (code bits, code length).",
                "4. CPU packs variable-length bitstream into final compressed bytes.",
            ]
        )
    )

if encode_clicked:
    result = compress_text(text)
    st.session_state["huffman_result"] = result

if "huffman_result" in st.session_state:
    result = st.session_state["huffman_result"]

    st.subheader("Compression Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original (bytes)", f"{len(result.original_bytes)}")
    m2.metric("Compressed (bytes)", f"{len(result.encoded_bytes)}")
    m3.metric("Total Bits", f"{result.total_encoded_bits}")
    m4.metric("Ratio", f"{result.compression_ratio:.3f}")

    st.info(f"Execution backend: {'GPU (CUDA)' if result.used_gpu else 'CPU fallback'}")

    st.subheader("Codebook")
    rows = codebook_rows(result)
    if rows:
        st.table(rows)
    else:
        st.write("No symbols to display for empty input.")

    st.subheader("Encoded Output Preview")
    st.code(result.encoded_bytes.hex()[:800] or "(empty)", language="text")

    decoded = decompress_text(result)
    st.subheader("Decoded Text")
    st.text_area("Decoded result", value=decoded, height=180)

    if decoded == result.original_text:
        st.success("Decode verification passed. Original text restored correctly.")
    else:
        st.error("Decode verification failed. Decoded text differs from original.")

    export_blob = build_export_payload(result)
    st.download_button(
        label="Download compressed report",
        data=export_blob,
        file_name="huffman_gpu_export.txt",
        mime="text/plain",
        use_container_width=True,
    )
