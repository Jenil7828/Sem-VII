# Lightweight POS tagger wrapper using Stanza
import stanza
import pandas as pd
from typing import Tuple

_PIPELINES = {}


def _init_pipeline(lang: str = 'hi', use_gpu: bool = False):
    try:
        nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos', use_gpu=use_gpu, verbose=False)
    except Exception:
        # download model and retry
        stanza.download(lang)
        nlp = stanza.Pipeline(lang=lang, processors='tokenize,pos', use_gpu=use_gpu, verbose=False)
    return nlp


def get_pipeline(lang: str = 'hi', use_gpu: bool = False):
    key = (lang, use_gpu)
    if key not in _PIPELINES:
        _PIPELINES[key] = _init_pipeline(lang, use_gpu)
    return _PIPELINES[key]


def tag_text(text: str, lang: str = 'hi', use_gpu: bool = False) -> pd.DataFrame:
    """Return a pandas DataFrame with columns: word, upos, xpos"""
    nlp = get_pipeline(lang, use_gpu)
    doc = nlp(text)
    rows = []
    for sent in doc.sentences:
        for w in sent.words:
            rows.append({'word': w.text, 'upos': getattr(w, 'upos', None), 'xpos': getattr(w, 'xpos', None)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    # quick test (will download model if missing)
    print(tag_text('यह एक परीक्षण वाक्य है।', lang='hi'))
