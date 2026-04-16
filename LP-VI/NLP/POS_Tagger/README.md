# POS Tagger for Indian Languages

Objective
---------
A simple POS tagging demo for Indian languages with a Streamlit UI using Stanza.

Setup
-----
1. Create and activate a Python virtual environment.

   Windows:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. (Optional) Download Stanza models for a language (first run may do this automatically):

   ```python
   import stanza
   stanza.download('hi')  # Hindi
   stanza.download('mr')  # Marathi
   stanza.download('ta')  # Tamil
   ```

Run the UI
----------

```bash
streamlit run app.py
```

Files
-----
- `pos_tagger.py`: wrapper for Stanza that returns tags as a DataFrame.
- `app.py`: Streamlit UI to enter text and see POS tags.
- `sample_sentences.txt`: example sentences to try.

Notes
-----
- The first run will download language models via Stanza and may take time.
- For large-scale usage, install the appropriate PyTorch wheel for your platform before installing stanza.
