# ===================================================================
# Universidad Autónoma de Chihuahua | Facultad de Ingeniería
# Data Science 
# Proyecto Final
# Olanda Prieto Ordaz
# Andres Gonzalez Alonso 
# 25 de mayo del  2026
# Modelos: SVM · Stacking · Random Forest
# ===================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import tempfile
import os
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(
    page_title="MurmurScan",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    box-sizing: border-box;
}

.stApp { background: #F5F2ED; color: #111; }

section[data-testid="stSidebar"] {
    background: #111111;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #F5F2ED !important; }
section[data-testid="stSidebar"] .stRadio label { color: #F5F2ED !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #F5F2ED !important; }
section[data-testid="stSidebar"] .stNumberInput label { color: #F5F2ED !important; }
section[data-testid="stSidebar"] .stMultiSelect label { color: #F5F2ED !important; }
section[data-testid="stSidebar"] .stFileUploader label { color: #F5F2ED !important; }
section[data-testid="stSidebar"] input { 
    background: #222 !important; 
    color: #F5F2ED !important;
    border: 1px solid #444 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #222 !important;
    color: #F5F2ED !important;
    border: 1px solid #444 !important;
}

/* Fix multiselect in dark sidebar */
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #222 !important;
    border: 1px solid #444 !important;
    color: #F5F2ED !important;
}
section[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {
    background: #333 !important;
    color: #F5F2ED !important;
}
section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
    background: #222 !important;
}
section[data-testid="stSidebar"] .stMultiSelect input {
    color: #F5F2ED !important;
    background: transparent !important;
}

.hero-block {
    display: flex;
    align-items: flex-end;
    gap: 0;
    margin-bottom: 32px;
    border-bottom: 3px solid #111;
    padding-bottom: 20px;
}
.hero-accent {
    width: 14px;
    height: 80px;
    background: #D62828;
    margin-right: 20px;
    flex-shrink: 0;
}
.hero-text {}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #111;
    line-height: 1;
    margin: 0;
}
.hero-sub {
    font-size: 0.78rem;
    color: #888;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 8px;
    font-weight: 400;
}

.label-tag {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #888;
    border-top: 2px solid #111;
    padding-top: 6px;
    margin-bottom: 14px;
}

.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    border: 2px solid #111;
    margin-bottom: 24px;
}
.kpi-cell {
    padding: 20px 16px;
    border-right: 2px solid #111;
    background: #fff;
}
.kpi-cell:last-child { border-right: none; }
.kpi-num {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #111;
    line-height: 1;
}
.kpi-lbl {
    font-size: 0.7rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

.result-box {
    border: 3px solid #111;
    padding: 36px 28px;
    text-align: center;
    background: #fff;
    position: relative;
}
.result-accent-n {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 6px;
    background: #2A9D2A;
}
.result-accent-a {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 6px;
    background: #D62828;
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1;
    margin-top: 12px;
}

.prob-wrap { margin: 20px 0; }
.prob-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    margin-bottom: 5px;
}
.prob-cls { font-family: 'Space Mono', monospace; color: #333; font-weight: 700; }
.prob-val { font-family: 'Space Mono', monospace; color: #666; }
.prob-bar-bg {
    height: 6px;
    background: #E0DDD8;
    margin-bottom: 14px;
}
.prob-bar-n  { height: 100%; background: #2A9D2A; }
.prob-bar-a  { height: 100%; background: #D62828; }

.strip-info {
    border-left: 4px solid #003087;
    background: #F0F4FF;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #003087;
    margin: 10px 0;
}
.strip-warn {
    border-left: 4px solid #E8A000;
    background: #FFFBF0;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #7a5200;
    margin: 10px 0;
}
.strip-err {
    border-left: 4px solid #D62828;
    background: #FFF0F0;
    padding: 12px 16px;
    font-size: 0.84rem;
    color: #8B0000;
    margin: 10px 0;
}

.sb-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #666 !important;
    border-top: 1px solid #333;
    padding-top: 14px;
    margin: 18px 0 10px;
}

.sdot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }
.sdot-on  { background:#2A9D2A; }
.sdot-off { background:#D62828; }

.stButton > button {
    background: #D62828;
    color: #fff;
    border: none;
    border-radius: 0;
    padding: 14px 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    width: 100%;
    transition: background 0.15s;
}
.stButton > button:hover { background: #aa1f1f; }

hr { border-color: #ddd; margin: 28px 0; }

.sb-brand {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #F5F2ED !important;
    letter-spacing: -0.5px;
    padding: 8px 0 4px;
}
.sb-brand-sub {
    font-size: 0.62rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #555 !important;
}

.detail-row {
    display: flex;
    border-bottom: 1px solid #E0DDD8;
    padding: 9px 0;
    font-size: 0.83rem;
}
.detail-key {
    font-family: 'Space Mono', monospace;
    color: #888;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 140px;
    flex-shrink: 0;
}
.detail-val { color: #111; font-weight: 500; }

.footer {
    border-top: 2px solid #111;
    margin-top: 40px;
    padding-top: 16px;
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: #aaa;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  PIPELINES
# ══════════════════════════════════════════════════════════════════════
class FeatureEngineer(BaseEstimator, TransformerMixin):
    AGE_ORDER = ['Neonate', 'Infant', 'Child', 'Adolescent', 'Young Adult']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['bmi'] = df['Weight'] / ((df['Height'] / 100) ** 2)
        df['recording_count'] = (
            df['Recording locations:']
            .fillna('')
            .apply(lambda x: len(x.split('+')) if x else 0)
        )
        df['age_group'] = pd.Categorical(
            df['Age'].fillna('Child'),
            categories=self.AGE_ORDER,
            ordered=True
        ).codes
        return df


class AudioFeatureExtractor(BaseEstimator, TransformerMixin):
    N_MFCC = 13

    def __init__(self, audio_dir="training_data", sr=4000, duration=10):
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration

    def fit(self, X, y=None):
        return self

    def extract_from_array(self, audio: np.ndarray) -> np.ndarray:
        sr = self.sr
        mfcc       = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.N_MFCC)
        delta_mfcc = librosa.feature.delta(mfcc)
        chroma     = librosa.feature.chroma_stft(y=audio, sr=sr)
        centroid   = librosa.feature.spectral_centroid(y=audio, sr=sr)
        rolloff    = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zcr        = librosa.feature.zero_crossing_rate(y=audio)
        rms        = librosa.feature.rms(y=audio)
        return np.concatenate([
            mfcc.mean(axis=1),       mfcc.std(axis=1),
            delta_mfcc.mean(axis=1), delta_mfcc.std(axis=1),
            chroma.mean(axis=1),     chroma.std(axis=1),
            centroid.mean(axis=1),   centroid.std(axis=1),
            rolloff.mean(axis=1),    rolloff.std(axis=1),
            zcr.mean(axis=1),        zcr.std(axis=1),
            rms.mean(axis=1),        rms.std(axis=1),
        ])

    def transform(self, X):
        return np.zeros((len(X), 84))


# ══════════════════════════════════════════════════════════════════════
#  CARGAR MODELOS
# ══════════════════════════════════════════════════════════════════════
MODEL_DIR = "models"

@st.cache_resource(show_spinner=False)
def load_pipelines():
    keys = ["tabular_pipeline", "audio_scaler", "label_encoder"]
    arts, missing = {}, []
    for k in keys:
        path = os.path.join(MODEL_DIR, f"{k}.pkl")
        if os.path.exists(path):
            try:
                arts[k] = joblib.load(path)
            except Exception as e:
                missing.append(f"{path} ({e})")
        else:
            missing.append(path)
    return arts, missing

@st.cache_resource(show_spinner=False)
def load_model_safe(filename):
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        try:
            return joblib.load(path), None
        except Exception as e:
            return None, str(e)
    return None, f"Archivo no encontrado: {path}"

pipelines,      missing_pipes   = load_pipelines()
svm_model,      svm_err         = load_model_safe("svm_optimizado.pkl")
stacking_model, stacking_err    = load_model_safe("stacking_opt.pkl")
rf_model,       rf_err          = load_model_safe("rf_optimizado.pkl")

MODEL_CFG = {
    "SVM": {
        "num": "01", "color": "#D62828", "css": "model-svm",
        "tag": "SVM · RBF Kernel",
        "desc": "Support Vector Machine, kernel RBF, optimizado con GridSearchCV",
        "model": svm_model, "err": svm_err,
    },
    "Stacking": {
        "num": "02", "color": "#003087", "css": "model-stack",
        "tag": "Ensemble · Stacking",
        "desc": "Meta-modelo de ensamble con múltiples clasificadores base",
        "model": stacking_model, "err": stacking_err,
    },
    "Random Forest": {
        "num": "03", "color": "#E8A000", "css": "model-rf",
        "tag": "Random Forest · Ensemble",
        "desc": "Bosque aleatorio optimizado, robusto ante ruido y outliers",
        "model": rf_model, "err": rf_err,
    },
}


# ══════════════════════════════════════════════════════════════════════
#  PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════
def build_features(tab_row: pd.DataFrame, audio: np.ndarray) -> np.ndarray:
    X_tab     = pipelines["tabular_pipeline"].transform(tab_row)
    extractor = AudioFeatureExtractor()
    X_aud_raw = extractor.extract_from_array(audio).reshape(1, -1)
    X_aud     = pipelines["audio_scaler"].transform(X_aud_raw)
    return np.hstack([X_tab, X_aud])

def predict_sklearn(model_key: str, X: np.ndarray):
    le    = pipelines["label_encoder"]
    model = MODEL_CFG[model_key]["model"]
    y_enc = model.predict(X)[0]
    label = le.inverse_transform([y_enc])[0]
    probs = None
    if hasattr(model, "predict_proba"):
        raw   = model.predict_proba(X)[0]
        probs = {c: float(p) for c, p in zip(le.classes_, raw)}
    return label, probs

def load_audio_bytes(data: bytes, sr=4000, duration=10) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        audio, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    finally:
        os.unlink(path)
    return audio


# ══════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN 
# ══════════════════════════════════════════════════════════════════════
def plot_audio_bauhaus(audio: np.ndarray, sr: int = 4000, model_color: str = "#D62828") -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    fig.patch.set_facecolor("#FFFFFF")
    fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.07, right=0.97, top=0.88, bottom=0.12)

    CREAM = "#F5F2ED"
    BLACK = "#111111"
    GRAY  = "#AAAAAA"

    def style(ax, title):
        ax.set_facecolor(CREAM)
        for sp in ax.spines.values():
            sp.set_color(BLACK)
            sp.set_linewidth(1.2)
        ax.tick_params(colors=GRAY, labelsize=7, length=3)
        ax.set_title(title, color=BLACK, fontsize=8, fontweight='bold',
                     fontfamily='monospace', pad=6, loc='left')
        ax.grid(False)

    t = np.linspace(0, len(audio)/sr, len(audio))
    axes[0,0].fill_between(t, audio, color=model_color, alpha=0.85, linewidth=0)
    axes[0,0].axhline(0, color=BLACK, linewidth=0.6)
    style(axes[0,0], "WAVEFORM")
    axes[0,0].set_xlabel("s", color=GRAY, fontsize=7)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    axes[0,1].imshow(mfcc, aspect='auto', origin='lower',
                     cmap='Greys', extent=[0, len(audio)/sr, 0, 13])
    style(axes[0,1], "MFCC")
    axes[0,1].set_xlabel("s", color=GRAY, fontsize=7)
    axes[0,1].set_ylabel("coef", color=GRAY, fontsize=7)

    rms  = librosa.feature.rms(y=audio)[0]
    t_r  = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    axes[1,0].fill_between(t_r, rms, color=BLACK, alpha=0.8, linewidth=0)
    style(axes[1,0], "ENERGÍA RMS")
    axes[1,0].set_xlabel("s", color=GRAY, fontsize=7)

    zcr  = librosa.feature.zero_crossing_rate(audio)[0]
    t_z  = librosa.frames_to_time(np.arange(len(zcr)), sr=sr)
    axes[1,1].fill_between(t_z, zcr, color=model_color, alpha=0.6, linewidth=0)
    axes[1,1].plot(t_z, zcr, color=model_color, linewidth=0.8)
    style(axes[1,1], "ZERO-CROSSING RATE")
    axes[1,1].set_xlabel("s", color=GRAY, fontsize=7)

    fig.suptitle("ANÁLISIS ESPECTRAL", color=BLACK, fontsize=9,
                 fontfamily='monospace', fontweight='bold', x=0.07, ha='left', y=0.97)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-brand">MURMURSCAN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-brand-sub">Proyecto Final </div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="sb-label">Modelo</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "", options=["SVM", "Stacking", "Random Forest"],
        index=0, label_visibility="collapsed"
    )
    cfg = MODEL_CFG[model_choice]
    if cfg["model"] is not None:
        st.markdown(
            f'<div style="border-left:4px solid {cfg["color"]}; background:#1e1e1e; '
            f'padding:12px 14px; margin:8px 0;">'
            f'<div style="font-family:\'Space Mono\',monospace; font-size:0.72rem; '
            f'font-weight:700; color:{cfg["color"]}; letter-spacing:0.5px; margin-bottom:6px;">'
            f'{cfg["tag"]}</div>'
            f'<div style="font-size:0.73rem; color:#aaaaaa; line-height:1.55;">'
            f'{cfg["desc"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f'<div class="strip-err">❌ {cfg["err"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Paciente</div>', unsafe_allow_html=True)
    age_group = st.selectbox("Edad", ['Neonate','Infant','Child','Adolescent','Young Adult'], index=2)
    c1, c2 = st.columns(2)
    with c1: height = st.number_input("Altura cm", 30.0, 220.0, 150.0, 0.5)
    with c2: weight = st.number_input("Peso kg",   1.0,  200.0,  50.0, 0.5)
    bmi = weight / ((height/100)**2)
    st.markdown(
        f'<div style="background:#1e1e1e; border-left:4px solid #555; padding:9px 14px; '
        f'margin:8px 0; font-size:0.78rem; color:#cccccc;">'
        f'IMC — <b style="color:#ffffff;">{bmi:.1f}</b></div>',
        unsafe_allow_html=True
    )
    pregnancy = st.selectbox("Embarazo", ['nan','False','True'], index=0)

    st.markdown('<div class="sb-label" style="border-top:1px solid #333; padding-top:14px; margin-top:18px; margin-bottom:8px; font-family:\'Space Mono\',monospace; font-size:0.62rem; letter-spacing:2.5px; text-transform:uppercase; color:#666;">Ubicaciones</div>', unsafe_allow_html=True)
    locations = st.multiselect(
        "Ubicaciones de grabación",
        options=['AV','MV','PV','TV','Phc'],
        default=['AV','MV'],
        label_visibility="collapsed"
    )
    loc_str = "+".join(locations) if locations else "AV"

    st.markdown('<div class="sb-label">Audio</div>', unsafe_allow_html=True)
    audio_file = st.file_uploader("Archivo .wav", type=["wav"], label_visibility="collapsed")

    st.markdown("---")
    predict_btn = st.button("ANALIZAR", use_container_width=True)

    st.markdown('<div class="sb-label">Estado</div>', unsafe_allow_html=True)
    for mname, mcfg in MODEL_CFG.items():
        dot = "sdot-on" if mcfg["model"] else "sdot-off"
        st.markdown(
            f'<div style="font-size:0.75rem;color:#aaa;margin:5px 0;">'
            f'<span class="sdot {dot}"></span><b style="color:#ddd;">{mname}</b></div>',
            unsafe_allow_html=True
        )


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero-block">
    <div class="hero-accent" style="background:{cfg['color']};"></div>
    <div class="hero-text">
        <div class="hero-title">MurmurScan</div>
        <div class="hero-sub">Clasificación de Sonidos Cardíacos · UACH · Facultad de Ingeniería · 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)

if missing_pipes:
    st.markdown(
        '<div class="strip-err">⚠ Pipelines no encontrados: '
        + " · ".join(f"<code>{f}</code>" for f in missing_pipes)
        + "</div>",
        unsafe_allow_html=True
    )

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-cell">
        <div class="kpi-num" style="color:{cfg['color']};">{height:.0f}</div>
        <div class="kpi-lbl">Altura cm</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-num">{weight:.0f}</div>
        <div class="kpi-lbl">Peso kg</div>
    </div>
    <div class="kpi-cell">
        <div class="kpi-num">{bmi:.1f}</div>
        <div class="kpi-lbl">IMC</div>
    </div>
    <div class="kpi-cell" style="border-right:none;">
        <div class="kpi-num">{len(locations)}</div>
        <div class="kpi-lbl">Ubicaciones</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_L, col_R = st.columns([3, 2], gap="large")

with col_L:
    st.markdown('<div class="label-tag">Señal de Audio</div>', unsafe_allow_html=True)

    if audio_file:
        audio_bytes = audio_file.read()
        audio_arr   = load_audio_bytes(audio_bytes)
        fig = plot_audio_bauhaus(audio_arr, model_color=cfg["color"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.audio(io.BytesIO(audio_bytes), format="audio/wav")
    else:
        st.markdown("""
        <div style="border:2px dashed #ccc; background:#fff; padding:60px 20px;
                    text-align:center; color:#bbb; font-family:'Space Mono',monospace;
                    font-size:0.8rem; letter-spacing:1px;">
            SUBE UN ARCHIVO .WAV<br>PARA VISUALIZAR LA SEÑAL
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="label-tag" style="margin-top:28px;">Modelos Disponibles</div>', unsafe_allow_html=True)
    rows = ""
    for mname, mcfg in MODEL_CFG.items():
        active  = "font-weight:700;" if mname == model_choice else ""
        status  = "✓" if mcfg["model"] else "✗"
        sc      = mcfg["color"]
        rows += f"""
        <div style="display:flex; border-bottom:1px solid #E0DDD8; padding:10px 0;
                    align-items:center; font-size:0.82rem; {active}">
            <div style="width:8px; height:8px; background:{sc};
                        border-radius:50%; margin-right:12px; flex-shrink:0;"></div>
            <div style="width:120px; font-family:'Space Mono',monospace;
                        font-size:0.78rem;">{mname}</div>
            <div style="flex:1; color:#666; font-size:0.75rem;">{mcfg['tag']}</div>
            <div style="color:{sc}; font-family:'Space Mono',monospace;
                        font-size:0.78rem;">{status}</div>
        </div>"""
    st.markdown(f'<div style="background:#fff; padding:4px 16px 8px;">{rows}</div>',
                unsafe_allow_html=True)


with col_R:
    st.markdown('<div class="label-tag">Resultado</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown(f"""
        <div style="border:2px solid #111; padding:50px 24px; text-align:center;
                    background:#fff;">
            <div style="width:60px; height:60px; border:3px solid {cfg['color']};
                        margin:0 auto 20px; display:flex; align-items:center;
                        justify-content:center; font-family:'Space Mono',monospace;
                        font-size:1.2rem; font-weight:700; color:{cfg['color']};">
                {cfg['num']}
            </div>
            <div style="font-family:'Space Mono',monospace; font-size:0.9rem;
                        font-weight:700; color:#111;">{model_choice}</div>
            <div style="font-size:0.72rem; color:#aaa; letter-spacing:2px;
                        text-transform:uppercase; margin-top:6px;">{cfg['tag']}</div>
            <div style="font-size:0.8rem; color:#888; margin-top:20px; line-height:1.6;">
                Sube el audio y presiona<br><b>ANALIZAR</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        errs = []
        if not audio_file:           errs.append("Falta el archivo de audio .wav")
        if not pipelines:            errs.append("Pipelines no cargados")
        if cfg["model"] is None:     errs.append(f"Modelo no disponible: {cfg['err']}")

        if errs:
            for e in errs:
                st.markdown(f'<div class="strip-err">✗ {e}</div>', unsafe_allow_html=True)
        else:
            tab_row = pd.DataFrame([{
                "Patient ID": "pred_online", "Height": height, "Weight": weight,
                "Age": age_group, "Pregnancy status": pregnancy,
                "Recording locations:": loc_str, "Murmur": "Unknown",
            }])

            with st.spinner("Procesando..."):
                X     = build_features(tab_row, audio_arr)
                label, probs = predict_sklearn(model_choice, X)

            accent_cls = "result-accent-n" if label == "Normal" else "result-accent-a"
            res_color  = "#2A9D2A" if label == "Normal" else "#D62828"
            st.markdown(f"""
            <div class="result-box">
                <div class="{accent_cls}"></div>
                <div style="font-size:0.68rem; color:#aaa; letter-spacing:3px;
                            text-transform:uppercase; margin-top:8px;">{model_choice} · {cfg['tag']}</div>
                <div class="result-label" style="color:{res_color};">{label.upper()}</div>
                <div style="font-size:0.78rem; color:#888; margin-top:12px; line-height:1.6;">
                    {"Sin anomalías detectadas en el sonido cardíaco."
                     if label == "Normal"
                     else "Posible anomalía detectada. Consulte a un especialista."}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if probs:
                st.markdown('<div class="label-tag" style="margin-top:20px;">Probabilidades</div>',
                            unsafe_allow_html=True)
                for cls in sorted(probs):
                    pct  = probs[cls] * 100
                    bfill = "prob-bar-n" if cls == "Normal" else "prob-bar-a"
                    st.markdown(f"""
                    <div class="prob-wrap">
                        <div class="prob-head">
                            <span class="prob-cls">{cls.upper()}</span>
                            <span class="prob-val">{pct:.1f}%</span>
                        </div>
                        <div class="prob-bar-bg">
                            <div class="{bfill}" style="width:{pct:.1f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="label-tag" style="margin-top:20px;">Detalle</div>',
                        unsafe_allow_html=True)
            details = [
                ("MODELO",      model_choice),
                ("ARQUITECTURA",cfg["tag"]),
                ("ALTURA",      f"{height:.1f} cm"),
                ("PESO",        f"{weight:.1f} kg"),
                ("IMC",         f"{bmi:.2f}"),
                ("EDAD",        age_group),
                ("UBICACIONES", loc_str),
                ("RESULTADO",   label.upper()),
            ]
            rows_html = "".join(
                f'<div class="detail-row"><div class="detail-key">{k}</div>'
                f'<div class="detail-val">{v}</div></div>'
                for k, v in details
            )
            st.markdown(f'<div style="background:#fff; padding:0 12px;">{rows_html}</div>',
                        unsafe_allow_html=True)


st.markdown(f"""
<div class="footer">
    <span>UACH · FACULTAD DE INGENIERÍA · DATA SCIENCE</span>
    <span>ANDRES GONZALEZ ALONSO · 2026</span>
    <span>SVM · STACKING · RANDOM FOREST</span>
</div>
""", unsafe_allow_html=True)