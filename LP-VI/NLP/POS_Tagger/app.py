import streamlit as st
import pandas as pd
from pos_tagger import tag_text

st.set_page_config(page_title='POS Tagger (Indian Languages)', layout='centered')
st.title('POS Tagger — Indian Languages (Demo)')

lang_map = {'Hindi': 'hi', 'Marathi': 'mr', 'Tamil': 'ta'}
lang_choice = st.selectbox('Language', list(lang_map.keys()), index=0)
lang = lang_map[lang_choice]

text = st.text_area('Enter text to tag', value='भारत में भाषा विविधता है।', height=150)

# Improved color palette for UPOS tags
TAG_COLORS = {
    'NOUN': '#FFD166',
    'VERB': '#06D6A0',
    'ADJ' : '#118AB2',
    'ADV' : '#EF476F',
    'PRON': '#8ECAE6',
    'NUM' : '#F4D35E',
    'DET' : '#FDE68A',
    'ADP' : '#C7F9CC',
    'CONJ': '#E9D5FF',
    'PART': '#FEC89A',
    'INTJ': '#F0EFEB',
    'X'   : '#E2E8F0',
    'PUNCT': '#F8F9FA'
}

def _hex_to_rgb(hex_color: str):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _text_color_for_bg(hex_color: str) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    # perceived luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return '#000000' if lum > 186 else '#ffffff'

def _escape_html(s: str) -> str:
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

if st.button('Tag Text'):
    with st.spinner('Tagging...'):
        try:
            df = tag_text(text, lang=lang)
        except Exception as e:
            st.error(f'Error: {e}')
            df = pd.DataFrame()
    if df.empty:
        st.info('No tokens found or an error occurred.')
    else:
        st.dataframe(df)

        # Legend (show known tags with colors)
        st.markdown('### Tag legend')
        legend_tags = list(TAG_COLORS.keys())
        legend_html = ' '.join(
            f"<span style='background:{TAG_COLORS[t]};color:{_text_color_for_bg(TAG_COLORS[t])};padding:6px 10px;border-radius:6px;margin:4px;display:inline-block'>{t}</span>"
            for t in legend_tags
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        # Inline colorized tokens
        def colorize(row):
            upos = row.get('upos') if isinstance(row, dict) else row['upos']
            if upos is None:
                upos = 'X'
            bg = TAG_COLORS.get(upos, TAG_COLORS['X'])
            fg = _text_color_for_bg(bg)
            word = _escape_html(row['word'])
            return f"<span style='background:{bg};color:{fg};padding:4px 8px;border-radius:6px;margin:2px;display:inline-block'>{word} <small style='opacity:.9'>/{_escape_html(upos)}</small></span>"

        html = ' '.join(colorize(r) for _, r in df.iterrows())
        st.markdown(html, unsafe_allow_html=True)

st.info('Model download: first run may download language models via Stanza.')
