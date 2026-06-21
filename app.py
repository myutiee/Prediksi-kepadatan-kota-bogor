import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Volume Lalu Lintas Kota Bogor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.main { background: linear-gradient(135deg, #f5f7fa 0%, #e8f4fd 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
    padding: 28px 32px; border-radius: 16px; margin-bottom: 24px;
    color: white; text-align: center;
    box-shadow: 0 8px 32px rgba(102,126,234,0.4);
}
.main-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 6px 0; }
.main-header p  { font-size: 0.95rem; opacity: 0.9; margin: 0; }
.kpi-card {
    background: white; border-radius: 14px; padding: 20px 18px;
    text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-top: 4px solid; height: 100%;
}
.kpi-val { font-size: 1.8rem; font-weight: 700; line-height: 1.1; margin: 6px 0 4px; }
.kpi-lbl { font-size: 0.78rem; color: #888; font-weight: 500; text-transform: uppercase; }
.kpi-sub { font-size: 0.72rem; color: #aaa; margin-top: 2px; }
.section-title {
    font-size: 1.1rem; font-weight: 600; color: #2d3748;
    margin: 20px 0 12px; padding-left: 10px; border-left: 4px solid #667eea;
}
.result-box {
    border-radius: 14px; padding: 24px; color: white;
    text-align: center; box-shadow: 0 8px 24px rgba(102,126,234,0.35);
}
.result-box .rv { font-size: 2.4rem; font-weight: 700; margin: 8px 0; }
.result-box .rl { font-size: 0.85rem; opacity: 0.85; text-transform: uppercase; }
.result-box .rk { font-size: 1rem; font-weight: 600; margin-top: 6px; }
.info-box {
    background: linear-gradient(135deg, #e0f2fe, #e8f5e9);
    border-radius: 10px; padding: 14px 16px;
    border-left: 4px solid #0ea5e9; font-size: 13px; color: #334155; margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Load & Train Model (dari CSV di GitHub) ──────────────────
CSV_URL = "https://raw.githubusercontent.com/myutiee/Prediksi-kepadatan-kota-bogor/main/Semua%20kondisi_volume_ruas_jalan%20-%20Data%20mentah%20.csv"

@st.cache_resource(show_spinner="⏳ Memuat data dan melatih model...")
def load_and_train():
    # Load CSV
    df = pd.read_csv(CSV_URL)

    # Preprocessing
    df_clean = df.copy()
    df_clean['hari'] = (df_clean['hari'].astype(str)
        .str.replace('\n','',regex=False).str.replace('  ',' ',regex=False)
        .str.strip().str.upper())
    df_clean['segmen'] = (df_clean['segmen'].astype(str)
        .str.replace('\n','',regex=False).str.replace('  ',' ',regex=False)
        .str.strip().str.upper())
    df_clean['hari'] = df_clean['hari'].replace('NAN','HARI KERJA').fillna('HARI KERJA')
    df_clean = df_clean[df_clean['volume_lalu_lintas'] > 0]
    Q1 = df_clean['volume_lalu_lintas'].quantile(0.25)
    Q3 = df_clean['volume_lalu_lintas'].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[~((df_clean['volume_lalu_lintas'] < Q1-1.5*IQR)|
                          (df_clean['volume_lalu_lintas'] > Q3+1.5*IQR))].reset_index(drop=True)

    le_hari   = LabelEncoder()
    le_segmen = LabelEncoder()
    df_clean['hari_encoded']   = le_hari.fit_transform(df_clean['hari'])
    df_clean['segmen_encoded'] = le_segmen.fit_transform(df_clean['segmen'])

    # Feature Engineering
    df_clean = df_clean.sort_values(['segmen','hari','tahun']).reset_index(drop=True)
    df_clean['vol_lag1']        = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].shift(1)
    df_clean['vol_lag2']        = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].shift(2)
    df_clean['vol_rolling2']    = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean())
    df_clean['vol_trend']       = df_clean['volume_lalu_lintas'] - df_clean['vol_lag1']
    df_clean['tahun_idx']       = df_clean['tahun'] - 2015
    df_clean['segmen_hari']     = df_clean['segmen_encoded'] * 10 + df_clean['hari_encoded']
    df_clean['segmen_mean_vol'] = df_clean.groupby('segmen_encoded')['volume_lalu_lintas'].transform('mean')
    df_clean['hari_mean_vol']   = df_clean.groupby('hari_encoded')['volume_lalu_lintas'].transform('mean')

    FEATURES = ['tahun','no_segmen','segmen_encoded','hari_encoded','segmen_hari',
                'tahun_idx','segmen_mean_vol','hari_mean_vol',
                'vol_lag1','vol_lag2','vol_rolling2','vol_trend']
    TARGET = 'volume_lalu_lintas'

    X = df_clean[FEATURES]
    y = df_clean[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train 4 model
    models = {
        'Linear Regression'    : Pipeline([('imp',SimpleImputer(strategy='median')),('lr',LinearRegression())]),
        'Random Forest'        : Pipeline([('imp',SimpleImputer(strategy='median')),('rf',RandomForestRegressor(n_estimators=100,random_state=42))]),
        'Gradient Boosting'    : Pipeline([('imp',SimpleImputer(strategy='median')),('gb',GradientBoostingRegressor(n_estimators=100,random_state=42))]),
        'HistGradient Boosting': HistGradientBoostingRegressor(max_iter=300,random_state=42),
    }

    results_metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        yp = model.predict(X_test)
        results_metrics[name] = {
            'MAE' : round(mean_absolute_error(y_test,yp),4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test,yp)),4),
            'R2'  : round(r2_score(y_test,yp),4),
        }

    best_model = models['Gradient Boosting']

    # Prediksi 2026
    last_vol    = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].last().reset_index()
    last_vol.columns = ['segmen','hari','vol_lag1']
    prev_vol    = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].apply(
        lambda x: x.iloc[-2] if len(x)>=2 else x.iloc[-1]).reset_index()
    prev_vol.columns = ['segmen','hari','vol_lag2']
    rolling_vol = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].apply(
        lambda x: x.tail(2).mean()).reset_index()
    rolling_vol.columns = ['segmen','hari','vol_rolling2']
    trend_vol   = df_clean.groupby(['segmen','hari'])['volume_lalu_lintas'].apply(
        lambda x: x.iloc[-1]-x.iloc[-2] if len(x)>=2 else 0).reset_index()
    trend_vol.columns = ['segmen','hari','vol_trend']

    seg_info = df_clean[['segmen','hari','no_segmen','segmen_encoded','hari_encoded',
                          'segmen_hari','segmen_mean_vol','hari_mean_vol']].drop_duplicates()
    df_2026  = seg_info.merge(last_vol).merge(prev_vol).merge(rolling_vol).merge(trend_vol)
    df_2026['tahun']         = 2026
    df_2026['tahun_idx']     = 11
    df_2026['prediksi_2026'] = best_model.predict(df_2026[FEATURES]).round(2)
    df_2026 = df_2026.sort_values('prediksi_2026', ascending=False).reset_index(drop=True)

    return df_clean, df_2026, results_metrics, best_model, FEATURES, le_hari, le_segmen

df_clean, df_2026, metrics, model, FEATURES, le_hari, le_segmen = load_and_train()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px;'>
        <div style='font-size:2.5rem;'>🚦</div>
        <div style='font-size:1rem;font-weight:700;color:white;margin-top:6px;'>Lalu Lintas Bogor</div>
        <div style='font-size:0.72rem;color:#aaa;margin-top:2px;'>Sistem Prediksi ML</div>
    </div>
    """, unsafe_allow_html=True)
    menu = st.radio("", [
        "📊  Dashboard Prediksi 2026",
        "🔮  Prediksi Manual",
        "📈  Evaluasi Model",
        "🗂️  Data Historis",
    ], label_visibility="collapsed")
    st.markdown("<hr style='border-color:#ffffff22;margin:20px 0 16px;'>", unsafe_allow_html=True)
    best_r2 = max(v['R2'] for v in metrics.values())
    best_mae = min(v['MAE'] for v in metrics.values())
    st.markdown(f"""
    <div style='font-size:11px;color:#aaa;padding:0 4px;'>
        <b style='color:#ccc;'>Info Model</b><br>
        🤖 GradientBoosting<br>
        📐 R² = {best_r2}<br>
        📉 MAE = {best_mae} SMP/Jam<br>
        📅 Data: 2015 – 2025<br>
        📍 {df_clean['segmen'].nunique()} Segmen Ruas Jalan
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🚦 Prediksi Volume Lalu Lintas Kota Bogor</h1>
    <p>Machine Learning · Gradient Boosting · R² = {best_r2} · Data 2015–2025</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if "Dashboard" in menu:
    filter_hari = st.selectbox("🗓️ Filter Tipe Hari", ["Semua","HARI KERJA","HARI LIBUR"])
    if filter_hari != "Semua":
        df_show = df_2026[df_2026['hari']==filter_hari].copy()
    else:
        df_show = df_2026.groupby('segmen')['prediksi_2026'].mean().reset_index()
        df_show.columns = ['segmen','prediksi_2026']
        df_show = df_show.sort_values('prediksi_2026',ascending=False)

    c1,c2,c3,c4 = st.columns(4)
    cards = [
        (c1,"🔴","#ff4757",f"{df_show['prediksi_2026'].max():,.0f}","Volume Tertinggi","SMP/Jam"),
        (c2,"🟢","#2ed573",f"{df_show['prediksi_2026'].min():,.0f}","Volume Terendah","SMP/Jam"),
        (c3,"📊","#667eea",f"{df_show['prediksi_2026'].mean():,.0f}","Rata-rata","SMP/Jam"),
        (c4,"📍","#ffa502",f"{len(df_show)}","Jumlah Segmen","ruas jalan"),
    ]
    for col,icon,color,val,lbl,sub in cards:
        with col:
            st.markdown(f"""<div class="kpi-card" style="border-top-color:{color}">
                <div style="font-size:1.6rem">{icon}</div>
                <div class="kpi-val" style="color:{color}">{val}</div>
                <div class="kpi-lbl">{lbl}</div><div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart, col_tbl = st.columns([1.2,1])
    with col_chart:
        st.markdown('<div class="section-title">📊 Top 15 Ruas Jalan Terpadat</div>', unsafe_allow_html=True)
        top15 = df_show.head(15).sort_values('prediksi_2026')
        colors_bar = ['#ff4757' if v>2500 else '#ffa502' if v>1800 else '#2ed573' for v in top15['prediksi_2026']]
        fig,ax = plt.subplots(figsize=(8,6.5))
        fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
        bars = ax.barh(range(len(top15)), top15['prediksi_2026'], color=colors_bar, height=0.65, edgecolor='white')
        ax.set_yticks(range(len(top15)))
        ax.set_yticklabels([s[:42] for s in top15['segmen']], fontsize=8)
        ax.set_xlabel('Volume (SMP/Jam)', fontsize=9)
        ax.set_title('Prediksi Volume 2026 — Top 15', fontsize=11, fontweight='bold')
        ax.spines[['top','right','left']].set_visible(False)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        for bar,val in zip(bars, top15['prediksi_2026']):
            ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2, f'{val:,.0f}', va='center', fontsize=7.5, fontweight='bold')
        p1=mpatches.Patch(color='#ff4757',label='Tinggi >2500')
        p2=mpatches.Patch(color='#ffa502',label='Sedang 1800–2500')
        p3=mpatches.Patch(color='#2ed573',label='Normal <1800')
        ax.legend(handles=[p1,p2,p3], fontsize=8, loc='lower right', framealpha=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_tbl:
        st.markdown('<div class="section-title">📋 Tabel Prediksi Lengkap</div>', unsafe_allow_html=True)
        df_display = df_show[['segmen','prediksi_2026']].copy()
        df_display['prediksi_2026'] = df_display['prediksi_2026'].apply(lambda x: f"{x:,.2f}")
        df_display.columns = ['Segmen Ruas Jalan','Volume (SMP/Jam)']
        st.dataframe(df_display, use_container_width=True, height=430)

# ════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI MANUAL
# ════════════════════════════════════════════════════════════
elif "Prediksi Manual" in menu:
    st.markdown('<div class="section-title">🔮 Prediksi Volume untuk Input Kustom</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Pilih segmen ruas jalan, tipe hari, dan tahun untuk prediksi volume lalu lintas.</div>', unsafe_allow_html=True)

    col_inp, col_res = st.columns([1,1.1])
    with col_inp:
        st.markdown('<div class="section-title">⚙️ Parameter Input</div>', unsafe_allow_html=True)
        segmen_input = st.selectbox("📍 Pilih Segmen", sorted(df_clean['segmen'].unique()))
        hari_input   = st.selectbox("🗓️ Tipe Hari", ["HARI KERJA","HARI LIBUR"])
        tahun_input  = st.slider("📅 Tahun Prediksi", 2024, 2030, 2026)
        predict_btn  = st.button("🚀 Prediksi Sekarang!", use_container_width=True)

    with col_res:
        if predict_btn:
            seg_hari_data = df_clean[
                (df_clean['segmen']==segmen_input)&(df_clean['hari']==hari_input)
            ].sort_values('tahun')
            seg_enc  = le_segmen.transform([segmen_input])[0]
            hari_enc = le_hari.transform([hari_input])[0]
            no_seg   = df_clean[df_clean['segmen']==segmen_input]['no_segmen'].iloc[0]
            vol_last = seg_hari_data['volume_lalu_lintas'].iloc[-1] if len(seg_hari_data)>=1 else df_clean['volume_lalu_lintas'].mean()
            vol_prev = seg_hari_data['volume_lalu_lintas'].iloc[-2] if len(seg_hari_data)>=2 else vol_last
            vol_roll = seg_hari_data['volume_lalu_lintas'].tail(2).mean() if len(seg_hari_data)>=1 else vol_last
            vol_trend= vol_last - vol_prev
            seg_mean = df_clean[df_clean['segmen_encoded']==seg_enc]['volume_lalu_lintas'].mean()
            hari_mean= df_clean[df_clean['hari_encoded']==hari_enc]['volume_lalu_lintas'].mean()
            input_df = pd.DataFrame([{
                'tahun':tahun_input,'no_segmen':no_seg,
                'segmen_encoded':seg_enc,'hari_encoded':hari_enc,
                'segmen_hari':seg_enc*10+hari_enc,'tahun_idx':tahun_input-2015,
                'segmen_mean_vol':seg_mean,'hari_mean_vol':hari_mean,
                'vol_lag1':vol_last,'vol_lag2':vol_prev,
                'vol_rolling2':vol_roll,'vol_trend':vol_trend,
            }])
            pred_vol = model.predict(input_df[FEATURES])[0]
            if pred_vol > 2500:
                kategori,warna = "TINGGI","#ff4757"
            elif pred_vol > 1800:
                kategori,warna = "SEDANG","#ffa502"
            else:
                kategori,warna = "NORMAL","#2ed573"
            st.markdown(f"""
            <div class="result-box" style="background:linear-gradient(135deg,{warna}cc,{warna});">
                <div class="rl">Prediksi Volume Lalu Lintas</div>
                <div class="rv">{pred_vol:,.2f}</div>
                <div class="rl">SMP/Jam · {tahun_input} · {hari_input}</div>
                <div class="rk">Kategori: {kategori}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f1f5f9;border-radius:14px;padding:40px;text-align:center;color:#94a3b8;margin-top:16px;">
                <div style="font-size:3rem">🔮</div>
                <div style="font-size:1rem;font-weight:600;margin-top:8px;">Hasil prediksi muncul di sini</div>
                <div style="font-size:0.8rem;margin-top:4px;">Isi parameter di kiri lalu klik Prediksi</div>
            </div>""", unsafe_allow_html=True)

    if predict_btn and len(seg_hari_data) > 0:
        st.markdown('<div class="section-title">📈 Tren Historis + Prediksi</div>', unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(11,4))
        fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
        ax.plot(seg_hari_data['tahun'], seg_hari_data['volume_lalu_lintas'],
                'o-', color='#667eea', lw=2.5, markersize=7, label='Data Historis')
        ax.fill_between(seg_hari_data['tahun'], seg_hari_data['volume_lalu_lintas'], alpha=0.15, color='#667eea')
        ax.scatter([tahun_input],[pred_vol], color=warna, s=180, zorder=5,
                   label=f'Prediksi {tahun_input}: {pred_vol:,.0f} SMP/Jam', edgecolors='white', linewidth=2)
        ax.set_title(f'{segmen_input[:55]} — {hari_input}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Tahun', fontsize=9); ax.set_ylabel('Volume (SMP/Jam)', fontsize=9)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════
# HALAMAN 3 — EVALUASI MODEL
# ════════════════════════════════════════════════════════════
elif "Evaluasi" in menu:
    st.markdown('<div class="section-title">📈 Perbandingan Performa 4 Model</div>', unsafe_allow_html=True)
    df_eval = pd.DataFrame(metrics).T.round(4)
    df_eval.index.name = 'Model'

    model_colors = ['#667eea','#ffa502','#2ed573','#ff4757']
    model_icons  = ['📐','🌲','⚡','🏆']
    cols = st.columns(len(df_eval))
    for i,(idx,row) in enumerate(df_eval.iterrows()):
        with cols[i]:
            is_best = row['R2'] == df_eval['R2'].max()
            border  = '#ffd700' if is_best else model_colors[i%len(model_colors)]
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{border}">
                <div style="font-size:1.4rem">{model_icons[i]}</div>
                <div style="font-size:0.7rem;font-weight:700;color:#555;margin:6px 0 10px;text-transform:uppercase">{idx}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{border}">{row['R2']:.4f}</div>
                <div style="font-size:0.7rem;color:#aaa">R² Score</div>
                <hr style="border-color:#f0f0f0;margin:8px 0">
                <div style="font-size:0.8rem;color:#666">MAE: <b>{row['MAE']:.2f}</b></div>
                <div style="font-size:0.8rem;color:#666">RMSE: <b>{row['RMSE']:.2f}</b></div>
                {'<div style="margin-top:8px"><span style="background:#ffd700;color:#333;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700">⭐ TERBAIK</span></div>' if is_best else ''}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    names  = list(df_eval.index)
    short  = ['Lin.Reg','Rand.Forest','Grad.Boost','HistGB'][:len(names)]
    colors = model_colors[:len(names)]
    col_r2, col_mae = st.columns(2)
    with col_r2:
        st.markdown('<div class="section-title">R² Score</div>', unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
        bars = ax.bar(short, df_eval['R2'], color=colors, width=0.55, edgecolor='white')
        ax.axhline(0.8, color='#ff4757', linestyle='--', lw=1.5, label='Target 0.8', alpha=0.7)
        ax.set_ylim(0,1.1); ax.set_ylabel('R²', fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.25, linestyle='--'); ax.legend(fontsize=8)
        for bar,val in zip(bars, df_eval['R2']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with col_mae:
        st.markdown('<div class="section-title">MAE (SMP/Jam)</div>', unsafe_allow_html=True)
        fig,ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
        bars2 = ax.bar(short, df_eval['MAE'], color=colors, width=0.55, edgecolor='white')
        ax.set_ylabel('MAE (SMP/Jam)', fontsize=9)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.25, linestyle='--')
        for bar,val in zip(bars2, df_eval['MAE']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val:.1f}', ha='center', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="info-box">
        <b>📌 Interpretasi:</b><br>
        • R² mendekati 1 → model mampu menjelaskan sebagian besar variasi volume lalu lintas<br>
        • MAE kecil → rata-rata kesalahan prediksi rendah<br>
        • Peningkatan R² dicapai dengan <b>Feature Engineering</b>: lag, rolling mean, dan tren historis
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HALAMAN 4 — DATA HISTORIS
# ════════════════════════════════════════════════════════════
elif "Historis" in menu:
    st.markdown('<div class="section-title">🗂️ Data Historis (Setelah Preprocessing)</div>', unsafe_allow_html=True)
    col1,col2 = st.columns(2)
    with col1:
        seg_filter = st.multiselect("📍 Filter Segmen", sorted(df_clean['segmen'].unique()),
                                    default=sorted(df_clean['segmen'].unique())[:3])
    with col2:
        hari_filter = st.multiselect("🗓️ Filter Hari", ['HARI KERJA','HARI LIBUR'],
                                     default=['HARI KERJA','HARI LIBUR'])

    df_filtered = df_clean[
        (df_clean['segmen'].isin(seg_filter))&(df_clean['hari'].isin(hari_filter))
    ][['tahun','segmen','hari','volume_lalu_lintas']].sort_values(['segmen','hari','tahun'])

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Baris", f"{len(df_filtered)}")
    c2.metric("Rata-rata Volume", f"{df_filtered['volume_lalu_lintas'].mean():,.0f} SMP/Jam" if len(df_filtered)>0 else "-")
    c3.metric("Periode","2015 – 2025")
    st.dataframe(df_filtered, use_container_width=True, height=300)

    if len(seg_filter)>0 and len(df_filtered)>0:
        st.markdown('<div class="section-title">📈 Tren Volume per Segmen</div>', unsafe_allow_html=True)
        palette = ['#667eea','#ff4757','#2ed573','#ffa502','#a55eea','#00d2d3','#ff6b81','#48dbfb']
        fig,ax = plt.subplots(figsize=(11,5))
        fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
        color_idx = 0
        for seg in seg_filter:
            for hari in hari_filter:
                subset = df_clean[(df_clean['segmen']==seg)&(df_clean['hari']==hari)].sort_values('tahun')
                if len(subset)>0:
                    color = palette[color_idx%len(palette)]
                    ls = '-' if hari=='HARI KERJA' else '--'
                    ax.plot(subset['tahun'], subset['volume_lalu_lintas'], marker='o', markersize=5,
                            color=color, linestyle=ls,
                            label=f"{seg[:20]}–{'Kerja' if hari=='HARI KERJA' else 'Libur'}", lw=2, alpha=0.85)
                    color_idx += 1
        ax.set_title('Tren Volume Lalu Lintas Historis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Tahun', fontsize=9); ax.set_ylabel('Volume (SMP/Jam)', fontsize=9)
        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.85, bbox_to_anchor=(1.01,1), borderaxespad=0)
        ax.grid(True, alpha=0.25, linestyle='--'); ax.spines[['top','right']].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;color:#aaa;font-size:12px;">
    🚦 <b>Sistem Prediksi Volume Lalu Lintas Kota Bogor</b> · Gradient Boosting · Data BPS 2015–2025
</div>""", unsafe_allow_html=True)
