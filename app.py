import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# ======================================================================
# PAGE SETUP
# ======================================================================
st.set_page_config(
    page_title="Manufacturing Defect Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================
# DESIGN TOKENS
# Palette dan tipografi dirancang mengikuti bahasa visual "andon" pada
# lini produksi manufaktur: hijau/kuning/merah untuk status kualitas,
# biru dongker & abu-baja untuk elemen struktural/instrumen.
# ======================================================================
NAVY_900 = "#0F2942"
NAVY_700 = "#1B4C7E"
STEEL_500 = "#5B7A99"
STEEL_100 = "#E7EEF5"
PAPER = "#F4F6F8"
INK = "#1A2733"
INK_SOFT = "#5C6B7A"
ANDON_GREEN = "#2F855A"
ANDON_AMBER = "#D9900D"
ANDON_RED = "#C0392B"

CLUSTER_PALETTE = ["#1B4C7E", "#2F8F82", "#8A5FBF", "#B5533C", "#5B7A99", "#C9A227"]
SEVERITY_COLORS = {"Minor": ANDON_GREEN, "Moderate": ANDON_AMBER, "Critical": ANDON_RED}
SEVERITY_ORDER = ["Minor", "Moderate", "Critical"]
SEVERITY_MAP = {"Minor": 1, "Moderate": 2, "Critical": 3}

# Kolom yang wajib ada pada CSV (baik file default maupun yang diunggah pengguna)
REQUIRED_COLUMNS = ["severity", "defect_type", "defect_location", "inspection_method", "repair_cost"]

# Variabel nominal (tanpa urutan alami) yang bisa ditambahkan sebagai fitur clustering opsional
CATEGORICAL_FEATURE_OPTIONS = ["defect_type", "defect_location", "inspection_method"]

TEMPLATE_CSV = (
    "defect_id,product_id,defect_type,defect_date,defect_location,severity,inspection_method,repair_cost\n"
    "1,101,Structural,1/1/2024,Component,Minor,Visual Inspection,150.00\n"
    "2,102,Functional,1/2/2024,Internal,Critical,Automated Testing,980.50\n"
)

# ======================================================================
# GLOBAL CSS
# ======================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=block');

html, body, [class*="st-"], .stMarkdown, .stApp {{
    font-family: 'IBM Plex Sans', sans-serif;
}}

/* Streamlit's built-in icons (sidebar collapse arrow, expander chevrons,
   file uploader icon, etc.) are plain text ligatures (e.g. "arrow_right")
   that only turn into glyphs when the "Material Symbols Rounded" font is
   loaded. Two things were breaking that: (1) the global font-family rule
   above matches every Streamlit class — they all contain "st-" — including
   these icon spans, so their font got silently swapped out for IBM Plex
   Sans, which has no ligature for "arrow_right"; and (2) Streamlit's own
   loading of that font is unreliable after deployment (a known upstream
   bug: github.com/streamlit/streamlit/issues/9945). The @import above
   fetches the font ourselves so we don't depend on Streamlit's copy, and
   this block is Google's official required CSS for the font to render
   ligatures as icons instead of raw text. */
[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    display: inline-block !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    overflow: hidden !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
    text-rendering: optimizeLegibility !important;
}}

.stApp {{
    background-color: {PAPER};
    background-image:
      linear-gradient(rgba(27,76,126,0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(27,76,126,0.035) 1px, transparent 1px);
    background-size: 26px 26px;
}}

/* ---------- Hero header ---------- */
.dq-hero {{
    background: linear-gradient(120deg, {NAVY_900} 0%, {NAVY_700} 100%);
    border-radius: 12px;
    padding: 30px 34px;
    margin-bottom: 20px;
    box-shadow: 0 8px 22px rgba(15,41,66,0.20);
}}
.dq-hero-eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 2.5px;
    color: #9FC1E0;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.dq-hero h1 {{
    color: #FFFFFF;
    font-size: 27px;
    font-weight: 700;
    margin: 0 0 8px 0;
    line-height: 1.25;
}}
.dq-hero p {{
    color: #C7D9E8;
    font-size: 14px;
    margin: 0;
    max-width: 760px;
    line-height: 1.5;
}}

/* ---------- KPI cards ---------- */
.dq-kpi {{
    background: #FFFFFF;
    border: 1px solid #E1E7EC;
    border-left: 4px solid {NAVY_700};
    border-radius: 8px;
    padding: 14px 16px;
    height: 100%;
}}
.dq-kpi-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {INK_SOFT};
    margin-bottom: 6px;
}}
.dq-kpi-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 24px;
    font-weight: 600;
    color: {INK};
}}
.dq-kpi-sub {{
    font-size: 12px;
    color: {INK_SOFT};
    margin-top: 4px;
}}

/* ---------- Andon severity strip ---------- */
.dq-andon-wrap {{
    background: #FFFFFF;
    border: 1px solid #E1E7EC;
    border-radius: 8px;
    padding: 12px 16px 14px 16px;
    margin-top: 14px;
}}
.dq-andon-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: {INK_SOFT};
    margin-bottom: 8px;
}}
.dq-andon-bar {{
    display: flex;
    width: 100%;
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
}}
.dq-andon-legend {{
    display: flex;
    gap: 22px;
    margin-top: 10px;
    flex-wrap: wrap;
}}
.dq-andon-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 12.5px;
    color: {INK};
}}
.dq-dot {{
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
}}

/* ---------- Section tag ---------- */
.dq-section-tag {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: {NAVY_700};
    background: {STEEL_100};
    border-radius: 4px;
    padding: 3px 10px;
    margin-bottom: 8px;
}}

/* ---------- Sidebar ---------- */
.dq-side-header {{
    background: {NAVY_900};
    border-radius: 8px;
    padding: 14px 14px;
    margin-bottom: 14px;
}}
.dq-side-header .eyebrow {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    letter-spacing: 1.5px;
    color: #9FC1E0;
    text-transform: uppercase;
}}
.dq-side-header .title {{
    color: #FFFFFF;
    font-size: 15px;
    font-weight: 600;
    margin-top: 3px;
}}
.dq-side-tag {{
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: {NAVY_700};
    background: {STEEL_100};
    border-radius: 4px;
    padding: 3px 10px;
    margin: 14px 0 6px 0;
}}

/* ---------- Tabs ---------- */
/* Inactive tab labels were nearly invisible against the page background
   (only readable on hover, since Streamlit's default inactive-tab color is
   a very light gray). Give the tab strip its own background card and force
   a legible, non-hover-dependent text color on every tab. */
[data-baseweb="tab-list"] {{
    background: {STEEL_100};
    border-radius: 8px;
    padding: 6px 8px 0 8px;
    gap: 4px;
}}
[data-baseweb="tab-list"] button[data-baseweb="tab"] {{
    background: transparent;
}}
[data-baseweb="tab-list"] button[data-baseweb="tab"] p {{
    color: {INK_SOFT} !important;
    opacity: 1 !important;
    font-weight: 500;
}}
[data-baseweb="tab-list"] button[data-baseweb="tab"][aria-selected="true"] p {{
    color: {NAVY_700} !important;
    font-weight: 700;
}}

/* misc */
footer {{ visibility: hidden; }}
[data-testid="stMetricDelta"] svg {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# HELPERS
# ======================================================================
def kpi_card(label: str, value: str, sub: str = "", accent: str = NAVY_700):
    st.markdown(f"""
    <div class="dq-kpi" style="border-left-color:{accent};">
        <div class="dq-kpi-label">{label}</div>
        <div class="dq-kpi-value">{value}</div>
        <div class="dq-kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def section_tag(text: str):
    st.markdown(f'<div class="dq-section-tag">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_manufacturing_data(file) -> pd.DataFrame:
    """Memuat data mentah dari CSV. `file` bisa berupa path lokal (str) atau
    objek UploadedFile dari st.file_uploader — keduanya diterima pd.read_csv."""
    return pd.read_csv(file)


def validate_schema(df: pd.DataFrame) -> list:
    """Mengembalikan daftar kolom wajib yang tidak ditemukan pada DataFrame."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


@st.cache_data(show_spinner="Menghitung kurva elbow (WCSS)...")
def compute_wcss(X_scaled: np.ndarray, k_max: int = 9):
    return [
        KMeans(n_clusters=i, init="k-means++", random_state=42, n_init=10).fit(X_scaled).inertia_
        for i in range(1, k_max + 1)
    ]


def plotly_scatter(data: pd.DataFrame, cluster_col: str, title: str):
    fig = go.Figure()
    for c in sorted(data[cluster_col].unique()):
        sub = data[data[cluster_col] == c]
        fig.add_trace(go.Scatter(
            x=sub["repair_cost"], y=sub["severity_score"],
            mode="markers",
            name=f"Cluster {c}",
            marker=dict(
                size=9,
                color=CLUSTER_PALETTE[c % len(CLUSTER_PALETTE)],
                line=dict(width=0.7, color="white"),
                opacity=0.85,
            ),
            customdata=sub[["defect_type", "defect_location", "inspection_method", "severity"]],
            hovertemplate=(
                "<b>%{customdata[3]} severity</b><br>"
                "Repair cost: $%{x:,.2f}<br>"
                "Type: %{customdata[0]}<br>"
                "Location: %{customdata[1]}<br>"
                "Inspection: %{customdata[2]}<extra></extra>"
            ),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(family="IBM Plex Sans", size=14, color=INK), x=0),
        xaxis_title="Repair Cost ($)",
        yaxis=dict(
            title="Severity Spectrum",
            tickmode="array", tickvals=[1, 2, 3],
            ticktext=["1 · Minor", "2 · Moderate", "3 · Critical"],
        ),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="IBM Plex Sans", color=INK_SOFT, size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1),
        margin=dict(t=60, l=10, r=10, b=10),
        height=380,
    )
    fig.update_xaxes(gridcolor="#EEF2F5", zeroline=False)
    fig.update_yaxes(gridcolor="#EEF2F5", zeroline=False)
    return fig


# ======================================================================
# APP BODY
# ======================================================================
try:
    # ------------------------------------------------------------------
    # SIDEBAR — HEADER
    # ------------------------------------------------------------------
    st.sidebar.markdown("""
    <div class="dq-side-header">
        <div class="eyebrow">Teknik Industri · UDINUS</div>
        <div class="title">Analytics Console</div>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # SIDEBAR — SUMBER DATA (upload opsional + validasi skema)
    # ------------------------------------------------------------------
    st.sidebar.markdown('<div class="dq-side-tag">Sumber Data</div>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Unggah CSV kustom (opsional)", type=["csv"],
        help=f"Kolom wajib: {', '.join(REQUIRED_COLUMNS)}. Tanpa unggahan, aplikasi memakai defects_data.csv bawaan.",
    )
    st.sidebar.download_button(
        "⬇  Unduh Template CSV",
        data=TEMPLATE_CSV.encode("utf-8"),
        file_name="template_defects_data.csv",
        mime="text/csv",
    )

    data_source = uploaded_file if uploaded_file is not None else "defects_data.csv"
    df_raw = load_manufacturing_data(data_source)

    missing_cols = validate_schema(df_raw)
    if missing_cols:
        st.error(
            "CSV tidak dapat diproses karena kolom wajib berikut tidak ditemukan: "
            f"**{', '.join(missing_cols)}**.\n\n"
            f"Kolom wajib pada file: {', '.join(REQUIRED_COLUMNS)}. Pastikan nama kolom sama persis "
            "(huruf kecil, underscore) seperti pada template — unduh template di atas sebagai acuan."
        )
        st.stop()

    df_full = df_raw.copy()

    # Guard: repair_cost harus numerik
    df_full["repair_cost"] = pd.to_numeric(df_full["repair_cost"], errors="coerce")
    invalid_cost = int(df_full["repair_cost"].isna().sum())
    df_full = df_full.dropna(subset=["repair_cost"]).copy()

    # Guard: severity labels di luar skala Minor/Moderate/Critical
    df_full["severity_score"] = df_full["severity"].map(SEVERITY_MAP)
    unmapped_labels = sorted(
        df_full.loc[df_full["severity_score"].isna(), "severity"].dropna().unique().tolist()
    )
    unmapped = int(df_full["severity_score"].isna().sum())
    df_full = df_full.dropna(subset=["severity_score"]).copy()
    df_full["severity_score"] = df_full["severity_score"].astype(int)

    if df_full.empty:
        st.error(
            "Tidak ada baris valid tersisa setelah validasi kolom `severity` dan `repair_cost`. "
            "Periksa kembali isi file CSV Anda dibandingkan dengan template."
        )
        st.stop()

    # ------------------------------------------------------------------
    # SIDEBAR — FILTER
    # ------------------------------------------------------------------
    st.sidebar.markdown('<div class="dq-side-tag">Filter Data</div>', unsafe_allow_html=True)
    with st.sidebar.expander("Persempit dataset", expanded=False):
        sel_severity = st.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
        sel_type = st.multiselect(
            "Jenis Cacat", sorted(df_full["defect_type"].unique()),
            default=sorted(df_full["defect_type"].unique()),
        )
        sel_method = st.multiselect(
            "Metode Inspeksi", sorted(df_full["inspection_method"].unique()),
            default=sorted(df_full["inspection_method"].unique()),
        )

    df = df_full[
        df_full["severity"].isin(sel_severity)
        & df_full["defect_type"].isin(sel_type)
        & df_full["inspection_method"].isin(sel_method)
    ].copy()

    if df.empty:
        st.warning("Tidak ada data yang cocok dengan filter saat ini. Ubah filter di sidebar.")
        st.stop()

    # ------------------------------------------------------------------
    # SIDEBAR — REKAYASA FITUR
    # ------------------------------------------------------------------
    st.sidebar.markdown('<div class="dq-side-tag">Rekayasa Fitur</div>', unsafe_allow_html=True)
    with st.sidebar.expander("Encoding & fitur tambahan", expanded=False):
        severity_encoding = st.radio(
            "Encoding Severity",
            ["Ordinal (1 → 3)", "One-Hot (kategorikal)"],
            index=0,
            help=(
                "Severity punya urutan alami Minor < Moderate < Critical, sehingga encoding ordinal "
                "1/2/3 valid secara statistik dan mempertahankan informasi jarak/urutan antar tingkat "
                "keparahan — ini alasan ordinal dipakai sebagai default, berbeda dari variabel nominal "
                "murni seperti defect_type. One-Hot disediakan untuk eksperimen/perbandingan, tapi bila "
                "diterapkan pada variabel berurutan seperti ini, informasi urutannya akan hilang."
            ),
        )
        extra_cat_features = st.multiselect(
            "Fitur kategorikal tambahan (one-hot)",
            CATEGORICAL_FEATURE_OPTIONS,
            default=[],
            help="Menambahkan defect_type / defect_location / inspection_method ke ruang fitur clustering, di luar repair_cost dan severity.",
        )

    st.sidebar.markdown('<div class="dq-side-tag">Clustering Engine</div>', unsafe_allow_html=True)
    n_clusters = st.sidebar.slider(
        label="Target Clusters (K)",
        min_value=2, max_value=6, value=3,
        help="Optimal K menurut Elbow Method & Dendrogram umumnya berada di sekitar K = 3.",
    )
    st.sidebar.caption(f"Menganalisis **{len(df):,}** dari {len(df_full):,} total catatan cacat.")
    if unmapped:
        label_str = ", ".join(f"'{l}'" for l in unmapped_labels) if unmapped_labels else "tidak dikenali"
        st.sidebar.caption(f"⚠️ {unmapped} baris diabaikan — label severity {label_str} di luar skala Minor/Moderate/Critical.")
    if invalid_cost:
        st.sidebar.caption(f"⚠️ {invalid_cost} baris diabaikan karena repair_cost bukan angka valid.")

    # ------------------------------------------------------------------
    # HERO
    # ------------------------------------------------------------------
    st.markdown("""
    <div class="dq-hero">
        <div class="dq-hero-eyebrow">Quality Engineering · Unsupervised Learning</div>
        <h1>Manufacturing Quality &amp; Defect Segmentation</h1>
        <p>Platform analitik untuk quality assurance produksi. Mengelompokkan cacat operasional
        berdasarkan tingkat keparahan fisik dan dampak biaya perbaikan, menggunakan K-Means dan
        Hierarchical Clustering.</p>
    </div>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # KPI ROW
    # ------------------------------------------------------------------
    total_records = len(df)
    avg_cost = df["repair_cost"].mean()
    max_cost = df["repair_cost"].max()

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Total Defect Records", f"{total_records:,}", "Setelah filter aktif", NAVY_700)
    with k2:
        kpi_card("Average Repair Cost", f"${avg_cost:,.2f}", "Per kejadian cacat", STEEL_500)
    with k3:
        kpi_card("Maximum Loss Instance", f"${max_cost:,.2f}", "Kerugian tunggal tertinggi", ANDON_RED)

    # ------------------------------------------------------------------
    # ANDON SEVERITY STRIP
    # ------------------------------------------------------------------
    sev_counts = df["severity"].value_counts().reindex(SEVERITY_ORDER).fillna(0).astype(int)
    sev_pct = (sev_counts / total_records * 100).round(1)

    bar_segments = "".join(
        f'<div style="width:{max(sev_pct[s], 0.5)}%; background:{SEVERITY_COLORS[s]};"></div>'
        for s in SEVERITY_ORDER
    )
    legend_items = "".join(
        f'<div class="dq-andon-item"><span class="dq-dot" style="background:{SEVERITY_COLORS[s]};"></span>'
        f'{s} — {sev_counts[s]:,} ({sev_pct[s]}%)</div>'
        for s in SEVERITY_ORDER
    )
    st.markdown(f"""
    <div class="dq-andon-wrap">
        <div class="dq-andon-title">Andon Status Strip — Distribusi Severity</div>
        <div class="dq-andon-bar">{bar_segments}</div>
        <div class="dq-andon-legend">{legend_items}</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ------------------------------------------------------------------
    # FEATURE ENGINEERING (on filtered data)
    # ------------------------------------------------------------------
    is_ordinal = severity_encoding.startswith("Ordinal")

    feature_frames = [df[["repair_cost"]]]
    feature_frames.append(
        df[["severity_score"]] if is_ordinal
        else pd.get_dummies(df["severity"], prefix="severity").astype(int)
    )
    if extra_cat_features:
        feature_frames.append(pd.get_dummies(df[extra_cat_features], prefix=extra_cat_features).astype(int))

    X = pd.concat(feature_frames, axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_summary = ["repair_cost", f"severity ({'ordinal' if is_ordinal else 'one-hot'})"] + extra_cat_features
    st.sidebar.caption(f"Fitur clustering aktif: {', '.join(feature_summary)} → {X.shape[1]} dimensi setelah encoding.")

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------
    tab_visualization, tab_data_profile, tab_math_validation = st.tabs([
        "🗺️  Spatial Segmentation Maps",
        "📋  Cluster Profiling & Data",
        "📐  Model Validation Mechanics",
    ])

    # ---- TAB 1 ----------------------------------------------------------
    with tab_visualization:
        section_tag("Model Comparison Matrix")
        st.caption("Perbandingan spasial antara model Partition-based (K-Means) dan Hierarchical (Agglomerative). Arahkan kursor ke titik untuk detail catatan cacat.")
        if not is_ordinal or extra_cat_features:
            st.caption("ℹ️ Klaster dihitung dari seluruh dimensi fitur yang dipilih di sidebar, namun sumbu 'Severity Spectrum' pada scatter plot tetap memakai skala ordinal (1–3) agar mudah dibaca secara visual.")

        km = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init=10)
        df["cluster_kmeans"] = km.fit_predict(X_scaled)

        hc = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
        df["cluster_hierarchy"] = hc.fit_predict(X_scaled)

        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            with st.container(border=True):
                st.plotly_chart(
                    plotly_scatter(df, "cluster_kmeans", "K-Means Partitioning Profile"),
                    width="stretch",
                )
        with vis_col2:
            with st.container(border=True):
                st.plotly_chart(
                    plotly_scatter(df, "cluster_hierarchy", "Agglomerative Hierarchy Profile (Ward)"),
                    width="stretch",
                )

    # ---- TAB 2 ----------------------------------------------------------
    with tab_data_profile:
        section_tag("Statistical Profile per Segment")
        st.caption("Rata-rata metrik pada tiap klaster, digunakan untuk memprioritaskan intervensi rekayasa.")

        prof_col1, prof_col2 = st.columns(2)
        with prof_col1:
            st.markdown("**K-Means Cluster Center Averages**")
            km_profile = df.groupby("cluster_kmeans")[["repair_cost", "severity_score"]].mean().round(2)
            km_profile.index.name = "Cluster"
            with st.container(border=True):
                st.dataframe(km_profile, width="stretch")
        with prof_col2:
            st.markdown("**Hierarchical Cluster Center Averages**")
            hc_profile = df.groupby("cluster_hierarchy")[["repair_cost", "severity_score"]].mean().round(2)
            hc_profile.index.name = "Cluster"
            with st.container(border=True):
                st.dataframe(hc_profile, width="stretch")

        st.write("")
        section_tag("Master Log Repository")
        with st.container(border=True):
            st.dataframe(df, width="stretch", height=360)

        st.download_button(
            "⬇  Unduh data hasil klaster (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="defects_clustered.csv",
            mime="text/csv",
        )

    # ---- TAB 3 ----------------------------------------------------------
    with tab_math_validation:
        section_tag("Mathematical Optimization Reference")
        st.caption("Diagnostik ground-truth untuk memilih jumlah klaster (K) secara objektif, independen dari slider di sidebar.")

        sil_km = silhouette_score(X_scaled, df["cluster_kmeans"]) if n_clusters > 1 else np.nan
        sil_hc = silhouette_score(X_scaled, df["cluster_hierarchy"]) if n_clusters > 1 else np.nan
        s1, s2 = st.columns(2)
        with s1:
            kpi_card("Silhouette Score — K-Means", f"{sil_km:.3f}", f"Pada K = {n_clusters} (rentang -1 s.d. 1, makin tinggi makin baik)", ANDON_GREEN if sil_km > 0.4 else ANDON_AMBER)
        with s2:
            kpi_card("Silhouette Score — Hierarchical", f"{sil_hc:.3f}", f"Pada K = {n_clusters} (rentang -1 s.d. 1, makin tinggi makin baik)", ANDON_GREEN if sil_hc > 0.4 else ANDON_AMBER)

        st.write("")
        eval_col1, eval_col2 = st.columns(2)

        with eval_col1:
            st.markdown("**Within-Cluster Sum of Squares (Elbow Curve)**")
            wcss = compute_wcss(X_scaled, k_max=9)
            k_range = list(range(1, 10))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=k_range, y=wcss, mode="lines+markers",
                line=dict(color=NAVY_700, width=2),
                marker=dict(size=7, color=STEEL_500),
                name="WCSS",
            ))
            fig.add_trace(go.Scatter(
                x=[n_clusters], y=[wcss[n_clusters - 1]], mode="markers",
                marker=dict(size=14, color=ANDON_RED, symbol="diamond", line=dict(width=1, color="white")),
                name=f"K terpilih = {n_clusters}",
            ))
            fig.update_layout(
                xaxis_title="Cluster Configuration (K)", yaxis_title="Inertia Metric",
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(family="IBM Plex Sans", color=INK_SOFT, size=11),
                margin=dict(t=10, l=10, r=10, b=10), height=320,
                showlegend=False,
            )
            fig.update_xaxes(gridcolor="#EEF2F5")
            fig.update_yaxes(gridcolor="#EEF2F5")
            with st.container(border=True):
                st.plotly_chart(fig, width="stretch")

        with eval_col2:
            st.markdown("**Hierarchical Agglomerative Dendrogram**")
            fig2, ax = plt.subplots(figsize=(5, 3.4))
            fig2.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            linkage_matrix = sch.linkage(X_scaled, method="ward")
            sch.set_link_color_palette([NAVY_700, STEEL_500, "#8A5FBF", "#2F8F82", "#C9A227"])
            sch.dendrogram(
                linkage_matrix, ax=ax, no_labels=True,
                truncate_mode="lastp", p=30,
                color_threshold=4, above_threshold_color=STEEL_500,
            )
            sch.set_link_color_palette(None)
            ax.set_ylabel("Linkage Distance (Euclidean)", color=INK_SOFT, fontsize=10)
            ax.tick_params(colors=INK_SOFT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig2.tight_layout()
            with st.container(border=True):
                st.pyplot(fig2, width="stretch")
            st.caption("Dendrogram ditampilkan dalam mode truncated (30 klaster daun terakhir) agar tetap terbaca.")

except FileNotFoundError:
    st.error("System Core Error: The essential dataset 'defects_data.csv' was not located in the running execution context.")
except pd.errors.EmptyDataError:
    st.error("File CSV kosong atau tidak memiliki data yang dapat dibaca.")
except pd.errors.ParserError:
    st.error("Gagal membaca file CSV. Pastikan file menggunakan format CSV valid dengan delimiter koma (,).")