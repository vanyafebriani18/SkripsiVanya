import io
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Segmentasi Pengguna Mobile Banking",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

LOGO_PATH = "assets/logo.png"
try:
    st.logo(LOGO_PATH)
except Exception:
    pass

st.markdown(
    """
    <style>
    .main {background-color: #f7faf8;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .title-box {
        background: linear-gradient(135deg, #004d24, #0f8f55);
        padding: 26px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 22px rgba(0, 77, 36, 0.15);
    }
    .title-box h1 {margin: 0; font-size: 32px;}
    .title-box p {margin: 8px 0 0 0; font-size: 15px; opacity: 0.95;}
    .section-card {
        background-color: white;
        border: 1px solid #e5eee9;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.03);
    }
    .small-note {
        background-color: #eef8f1;
        border-left: 5px solid #004d24;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 10px 0 16px 0;
    }
    .warning-note {
        background-color: #fff7e6;
        border-left: 5px solid #f59e0b;
        padding: 14px 16px;
        border-radius: 10px;
        margin: 10px 0 16px 0;
    }
    div[data-testid="stMetricValue"] {font-size: 26px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-box">
        <h1>Dashboard Segmentasi Pengguna Mobile Banking</h1>
        <p>Implementasi K-Means Clustering, Elbow Method, dan Silhouette Coefficient untuk rekomendasi strategi pemasaran berbasis data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# KOLOM SESUAI PENULISAN SKRIPSI
# =========================================================
# Variabel utama ini adalah input langsung ke K-Means Clustering.
FEATURE_COLUMNS = [
    "Saldo_Rata_Rata",
    "Frekuensi_Transaksi",
    "Nominal_Transaksi_Bulanan",
    "Keragaman_Fitur",
    "Respons_Promosi",
    "Kepuasan_Layanan",
    "Kepercayaan_Keamanan",
]

# Data profil tidak dipakai sebagai input K-Means, tetapi dipakai untuk interpretasi cluster.
# Disesuaikan dengan kuesioner dan Bab 3: usia, jenis kelamin, pekerjaan, pendapatan,
# serta gambaran umum penggunaan digital banking.
PROFILE_COLUMNS = [
    "Usia",
    "Jenis_Kelamin",
    "Jenis_Pekerjaan",
    "Pendapatan_Bulanan",
    "Jenis_Layanan_Digital",
    "Aplikasi_Digital_Banking",
]

# Data pendukung rekomendasi tidak dipakai sebagai input K-Means.
# Disesuaikan dengan Section 5 kuesioner dan penulisan Bab 3/Bab 4.
SUPPORT_COLUMNS = [
    "Tujuan_Penggunaan",
    "Tujuan_Keuangan",
    "Minat_Tabungan_Deposito",
    "Jenis_Promosi_Disukai",
    "Preferensi_Promo_Tabungan_Deposito",
    "Faktor_Minat_Tabungan_Deposito",
    "Media_Promosi_Diperhatikan",
    "Kendala_Penggunaan",
    "Faktor_Peningkatan_Minat",
    "Aktivitas_Pembayaran_Digital",
]

# Item skala yang digunakan untuk Uji Instrumen Kuesioner.
# Item ini berasal dari jawaban skala Likert pada kuesioner.
INSTRUMENT_COLUMNS = [
    "Respons_Promosi",
    "Kepuasan_Layanan",
    "Kepercayaan_Keamanan",
]

DISPLAY_NAMES = {
    "Saldo_Rata_Rata": "Saldo Rata-rata",
    "Frekuensi_Transaksi": "Frekuensi Transaksi",
    "Nominal_Transaksi_Bulanan": "Nominal Transaksi Bulanan",
    "Keragaman_Fitur": "Keragaman Fitur yang Digunakan",
    "Respons_Promosi": "Respons terhadap Promosi",
    "Kepuasan_Layanan": "Kepuasan Layanan",
    "Kepercayaan_Keamanan": "Kepercayaan terhadap Keamanan",
    "Usia": "Usia",
    "Jenis_Kelamin": "Jenis Kelamin",
    "Jenis_Pekerjaan": "Jenis Pekerjaan",
    "Pendapatan_Bulanan": "Pendapatan Bulanan",
    "Jenis_Layanan_Digital": "Jenis Layanan Digital Banking",
    "Aplikasi_Digital_Banking": "Aplikasi/Layanan Digital Banking yang Paling Sering Digunakan",
    "Tujuan_Penggunaan": "Tujuan Penggunaan Mobile Banking",
    "Tujuan_Keuangan": "Tujuan Keuangan",
    "Minat_Tabungan_Deposito": "Minat Tabungan Digital/Deposito",
    "Jenis_Promosi_Disukai": "Jenis Promosi yang Disukai",
    "Preferensi_Promo_Tabungan_Deposito": "Preferensi Promo Tabungan/Deposito",
    "Faktor_Minat_Tabungan_Deposito": "Faktor Minat Tabungan/Deposito",
    "Media_Promosi_Diperhatikan": "Media Promosi yang Diperhatikan",
    "Kendala_Penggunaan": "Kendala Penggunaan",
    "Faktor_Peningkatan_Minat": "Faktor yang Meningkatkan Minat Penggunaan",
    "Aktivitas_Pembayaran_Digital": "Aktivitas Pembayaran Digital yang Sering Dilakukan",
}

# Alias kolom dibuat untuk membaca nama kolom Google Form yang panjang.
# Program tetap memakai nama kanonik di atas, tetapi bisa menerima kolom hasil export Google Form.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Customer_ID": ["customer_id", "customerid", "kode_responden", "responden_id", "id_responden", "nomor_responden"],
    "Jenis_Layanan_Digital": [
        "jenis_layanan_digital_banking_yang_paling_sering_anda_gunakan_adalah",
        "jenis_layanan_digital_banking_yang_paling_sering_anda_gunakan",
        "jenis_layanan_digital_banking",
        "jenis_layanan_digital",
    ],
    "Aplikasi_Digital_Banking": [
        "nama_aplikasi_layanan_digital_banking_yang_paling_sering_anda_gunakan",
        "aplikasi_digital_banking_yang_paling_sering_digunakan",
        "nama_aplikasi_layanan_digital_banking",
        "aplikasi_layanan_digital_banking",
        "bank_digunakan",
        "mobile_banking_yang_digunakan",
    ],
    "Usia": ["usia_anda_saat_ini", "usia", "umur", "age"],
    "Jenis_Kelamin": ["jenis_kelamin", "gender"],
    "Jenis_Pekerjaan": ["pekerjaan_saat_ini", "jenis_pekerjaan", "pekerjaan", "profesi", "occupation", "job"],
    "Pendapatan_Bulanan": ["pendapatan_bulanan_anda", "pendapatan_bulanan", "pendapatan", "penghasilan", "income", "monthly_income"],
    "Saldo_Rata_Rata": [
        "saldo_rata_rata_dalam_3_bulan_terakhir_berapa_rata_rata_saldo",
        "saldo_rata_rata",
        "rata_rata_saldo",
        "saldo",
        "average_balance",
        "balance",
    ],
    "Frekuensi_Transaksi": [
        "frekuensi_transaksi_dalam_3_bulan_terakhir_berapa_kali_rata_rata_anda_melakukan_transaksi",
        "frekuensi_transaksi",
        "jumlah_transaksi",
        "frekuensi",
        "freq",
        "transaction_frequency",
    ],
    "Nominal_Transaksi_Bulanan": [
        "nominal_transaksi_bulanan_dalam_3_bulan_terakhir_berapa_perkiraan_total_nominal_transaksi",
        "nominal_transaksi_bulanan",
        "nominal_transaksi",
        "total_nominal_transaksi",
        "total_transaksi",
        "monthly_transaction_value",
        "transaction_amount",
    ],
    "Keragaman_Fitur": [
        "keragaman_fitur_yang_digunakan_fitur_digital_banking_mobile_banking_apa_saja_yang_anda_gunakan",
        "keragaman_fitur_yang_digunakan",
        "keragaman_fitur",
        "jumlah_fitur",
        "fitur_digunakan",
        "feature_diversity",
        "used_features",
    ],
    "Respons_Promosi": [
        "respons_terhadap_promosi_saya_tertarik_menggunakan_digital_banking_mobile_banking_ketika_terdapat_promosi",
        "respons_terhadap_promosi",
        "respons_promosi",
        "respon_promosi",
        "promo_response",
        "promotion_response",
    ],
    "Kepuasan_Layanan": [
        "kepuasan_layanan_saya_merasa_puas_dengan_kemudahan_kecepatan_dan_kenyamanan_layanan",
        "kepuasan_layanan",
        "kepuasan",
        "satisfaction",
        "service_satisfaction",
    ],
    "Kepercayaan_Keamanan": [
        "kepercayaan_terhadap_keamanan_layanan_saya_percaya_bahwa_layanan_digital_banking_mobile_banking_yang_saya_gunakan_aman",
        "kepercayaan_terhadap_keamanan_layanan",
        "kepercayaan_keamanan",
        "security_trust",
        "trust",
    ],
    "Tujuan_Penggunaan": [
        "tujuan_anda_menggunakan_digital_banking_mobile_banking_adalah",
        "tujuan_penggunaan",
        "tujuan_menggunakan_mobile_banking",
        "alasan_penggunaan",
        "tujuan_utama_penggunaan",
    ],
    "Tujuan_Keuangan": [
        "dalam_beberapa_bulan_ke_depan_tujuan_keuangan_yang_ingin_anda_capai_adalah",
        "tujuan_keuangan",
        "rencana_keuangan",
        "financial_goal",
    ],
    "Minat_Tabungan_Deposito": [
        "saya_tertarik_menggunakan_fitur_tabungan_digital_tabungan_berjangka_atau_deposito_melalui_mobile_banking",
        "minat_tabungan_deposito",
        "minat_tabungan",
        "minat_deposito",
        "saving_deposit_interest",
    ],
    "Jenis_Promosi_Disukai": [
        "jenis_promosi_digital_banking_mobile_banking_yang_menarik_bagi_anda_adalah",
        "jenis_promosi_disukai",
        "promo_disukai",
        "promosi_disukai",
        "preferred_promo",
    ],
    "Preferensi_Promo_Tabungan_Deposito": [
        "jika_terdapat_promo_deposito_atau_tabungan_digital_bentuk_promo_yang_paling_menarik_bagi_anda_adalah",
        "preferensi_promo_deposito",
        "preferensi_promo_tabungan",
        "promo_deposito_tabungan",
    ],
    "Faktor_Minat_Tabungan_Deposito": [
        "faktor_yang_dapat_membuat_anda_tertarik_menggunakan_produk_tabungan_digital_atau_deposito_melalui_mobile_banking_adalah",
        "faktor_minat_tabungan_deposito",
        "faktor_tabungan_deposito",
    ],
    "Media_Promosi_Diperhatikan": [
        "media_promosi_digital_banking_mobile_banking_yang_sering_anda_perhatikan_adalah",
        "media_promosi_diperhatikan",
        "media_promosi",
        "preferred_channel",
        "channel_promosi",
    ],
    "Kendala_Penggunaan": [
        "kendala_yang_pernah_anda_alami_saat_menggunakan_digital_banking_mobile_banking_adalah",
        "kendala_penggunaan",
        "kendala",
        "hambatan",
        "usage_barrier",
    ],
    "Faktor_Peningkatan_Minat": [
        "hal_yang_dapat_meningkatkan_minat_anda_dalam_menggunakan_digital_banking_mobile_banking_adalah",
        "menurut_anda_hal_apa_yang_paling_dapat_meningkatkan_minat_penggunaan_digital_banking_mobile_banking",
        "faktor_peningkatan_minat_penggunaan",
        "peningkatan_minat_penggunaan",
    ],
    "Aktivitas_Pembayaran_Digital": [
        "dalam_penggunaan_digital_banking_mobile_banking_aktivitas_pembayaran_digital_yang_paling_sering_anda_lakukan_adalah",
        "aktivitas_pembayaran_digital_yang_sering_dilakukan",
        "aktivitas_pembayaran_digital",
        "qris",
    ],
}

LIKERT_MAP = {
    "sangat rendah": 1,
    "sangat tidak setuju": 1,
    "tidak pernah": 1,
    "tidak tertarik": 1,
    "rendah": 2,
    "tidak setuju": 2,
    "jarang": 2,
    "kurang tertarik": 2,
    "sedang": 3,
    "netral": 3,
    "kadang-kadang": 3,
    "kadang kadang": 3,
    "cukup": 3,
    "tinggi": 4,
    "setuju": 4,
    "sering": 4,
    "tertarik": 4,
    "sangat tinggi": 5,
    "sangat setuju": 5,
    "selalu": 5,
    "sangat tertarik": 5,
}

# =========================================================
# FUNGSI BANTUAN
# =========================================================
def normalize_text(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Mengubah nama kolom upload agar sesuai dengan nama kolom penelitian.

    Revisi ini dibuat agar program dapat membaca kolom hasil export Google Form
    yang biasanya berbentuk pertanyaan panjang, misalnya "Saldo Rata-Rata\nDalam 3 bulan...".
    Program tetap mempertahankan nama kanonik seperti Saldo_Rata_Rata,
    Frekuensi_Transaksi, dan variabel lain sesuai penulisan skripsi.
    """
    normalized_columns = {col: normalize_text(col) for col in df.columns}
    rename_map: Dict[str, str] = {}
    used_original_cols = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        alias_norms = [normalize_text(alias) for alias in aliases + [canonical]]
        alias_norms = sorted(set(alias_norms), key=len, reverse=True)

        best_col = None
        best_score = -1

        for original_col, normalized_col in normalized_columns.items():
            if original_col in used_original_cols:
                continue

            for alias in alias_norms:
                score = -1
                if normalized_col == alias:
                    score = 1000 + len(alias)
                elif len(alias) > 3 and normalized_col.startswith(alias):
                    score = 700 + len(alias)
                elif len(alias) > 3 and alias in normalized_col:
                    score = 400 + len(alias)

                if score > best_score:
                    best_score = score
                    best_col = original_col

        if best_col is not None and best_score >= 0:
            rename_map[best_col] = canonical
            used_original_cols.add(best_col)

    return df.rename(columns=rename_map)


def rupiah_text_to_number(text: str) -> float:
    """Konversi jawaban rentang rupiah atau teks Likert menjadi angka."""
    if pd.isna(text):
        return np.nan

    original = str(text).strip().lower()
    normalized = re.sub(r"\s+", " ", original)

    if normalized in LIKERT_MAP:
        return float(LIKERT_MAP[normalized])

    if "tidak ingin menyebutkan" in normalized or "tidak bersedia" in normalized:
        return np.nan
    if "belum memiliki pendapatan" in normalized or "belum bekerja" in normalized:
        return 0.0

    cleaned = normalized.replace("rp", " ").replace("idr", " ")
    cleaned = cleaned.replace(",", ".")
    multiplier = 1.0
    if "juta" in cleaned:
        multiplier = 1_000_000.0
    elif "ribu" in cleaned:
        multiplier = 1_000.0

    nums = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not nums:
        return np.nan

    values = [float(n) for n in nums]

    # Menangani format 2.000.000 jika tidak ada kata juta/ribu.
    if multiplier == 1.0:
        digit_groups = re.findall(r"\d+", cleaned)
        if len(digit_groups) >= 2 and any(len(group) == 3 for group in digit_groups[1:]):
            reconstructed = []
            buffer = digit_groups[0]
            for group in digit_groups[1:]:
                if len(group) == 3:
                    buffer += group
                else:
                    reconstructed.append(float(buffer))
                    buffer = group
            reconstructed.append(float(buffer))
            values = reconstructed

    values = [v * multiplier for v in values]
    if "<" in normalized or "kurang" in normalized or "dibawah" in normalized or "di bawah" in normalized:
        return values[0] / 2
    if ">" in normalized or "lebih" in normalized or "diatas" in normalized or "di atas" in normalized:
        return values[-1] * 1.25
    return float(np.mean(values))


def count_selected_options(value) -> float:
    """Menghitung jumlah fitur jika jawaban berupa pilihan ganda dipisahkan koma/titik koma."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    numeric = rupiah_text_to_number(text)
    if not np.isnan(numeric):
        return numeric
    # Jika jawaban fitur berupa daftar pilihan.
    parts = [p.strip() for p in re.split(r"[,;|]+", text) if p.strip()]
    return float(len(parts)) if parts else np.nan


def to_numeric_series(series: pd.Series, col_name: str = "") -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    if col_name == "Keragaman_Fitur":
        return series.apply(count_selected_options)
    return series.apply(rupiah_text_to_number)


def optional_numeric_profile(processed: pd.DataFrame) -> pd.DataFrame:
    """Konversi usia dan pendapatan untuk kebutuhan interpretasi, bukan input clustering."""
    result = processed.copy()
    for col in ["Usia", "Pendapatan_Bulanan"]:
        if col in result.columns:
            result[col] = to_numeric_series(result[col], col)
    return result


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, StandardScaler, List[str]]:
    df = standardize_column_names(df.copy())

    # Kolom ini pernah ada pada rancangan lama, tetapi sudah tidak digunakan dalam penulisan final.
    # Kolom dihapus agar tidak muncul sebagai input, profil, maupun data pendukung.
    if "Tingkat_Aktivitas_Penggunaan" in df.columns:
        df = df.drop(columns=["Tingkat_Aktivitas_Penggunaan"])

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        return df, pd.DataFrame(), np.array([]), StandardScaler(), missing

    processed = optional_numeric_profile(df.copy())
    for col in FEATURE_COLUMNS:
        processed[col] = to_numeric_series(processed[col], col)

    X_raw = processed[FEATURE_COLUMNS]
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    processed[FEATURE_COLUMNS] = X_imputed
    return processed, pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS), X_scaled, scaler, []

def get_instrument_data(processed: pd.DataFrame) -> pd.DataFrame:
    """Mengambil item skala untuk uji validitas dan reliabilitas kuesioner.

    Uji instrumen dilakukan pada item yang berbentuk skala Likert/persepsi.
    Kolom yang tersedia akan dikonversi ke numerik, lalu data kosong diisi dengan median.
    """
    available_cols = [col for col in INSTRUMENT_COLUMNS if col in processed.columns]
    if not available_cols:
        return pd.DataFrame()

    instrument = pd.DataFrame()
    for col in available_cols:
        instrument[col] = to_numeric_series(processed[col], col)

    instrument = instrument.dropna(how="all")
    if instrument.empty:
        return pd.DataFrame()

    for col in instrument.columns:
        if instrument[col].isna().all():
            instrument[col] = 0
        else:
            instrument[col] = instrument[col].fillna(instrument[col].median())
    return instrument


def calculate_validity_table(instrument_df: pd.DataFrame, r_threshold: float = 0.30) -> pd.DataFrame:
    """Menghitung korelasi item-total sebagai uji validitas sederhana.

    Item dinilai berdasarkan korelasi item-total sebagai pendukung Bab IV.
    Karena indikator berasal dari konstruk yang berbeda, hasil ini tidak digunakan sebagai dasar tunggal untuk menghapus variabel.
    """
    if instrument_df.empty or instrument_df.shape[1] < 2:
        return pd.DataFrame()

    records = []
    for col in instrument_df.columns:
        other_cols = [c for c in instrument_df.columns if c != col]
        total_score = instrument_df[other_cols].sum(axis=1)
        if instrument_df[col].std(ddof=0) == 0 or total_score.std(ddof=0) == 0:
            r_value = np.nan
        else:
            r_value = instrument_df[col].corr(total_score)
        records.append(
            {
                "Item": col,
                "Nama Item": DISPLAY_NAMES.get(col, col),
                "r hitung": r_value,
                "Batas r": r_threshold,
                "Keterangan": "Memenuhi batas korelasi" if pd.notna(r_value) and r_value >= r_threshold else "Perlu interpretasi hati-hati",
            }
        )
    return pd.DataFrame(records)


def cronbach_alpha(instrument_df: pd.DataFrame) -> float:
    """Menghitung Cronbach's Alpha untuk uji reliabilitas."""
    if instrument_df.empty or instrument_df.shape[1] < 2:
        return np.nan
    item_variances = instrument_df.var(axis=0, ddof=1)
    total_score = instrument_df.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    k = instrument_df.shape[1]
    if total_variance == 0 or pd.isna(total_variance):
        return np.nan
    return float((k / (k - 1)) * (1 - item_variances.sum() / total_variance))


def reliability_label(alpha: float, threshold: float = 0.70) -> str:
    if np.isnan(alpha):
        return "Belum dapat dihitung"
    if alpha >= threshold:
        return "Memenuhi batas Alpha"
    return "Tidak dijadikan dasar utama (indikator berbeda konstruk)"



def calculate_k_scores(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    max_allowed = min(k_max, len(X_scaled) - 1)
    records = []
    for k in range(k_min, max_allowed + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_
        silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
        records.append({"K": k, "Inertia": inertia, "Silhouette": silhouette})
    return pd.DataFrame(records)


def fit_kmeans(X_scaled: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray, float]:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else np.nan
    return model, labels, score


def choose_best_k(k_scores: pd.DataFrame) -> int:
    if k_scores.empty or k_scores["Silhouette"].isna().all():
        return 3
    return int(k_scores.loc[k_scores["Silhouette"].idxmax(), "K"])


def dominant_text(series: pd.Series, default: str = "-") -> str:
    if series is None or series.empty:
        return default
    values: List[str] = []
    for item in series.dropna().astype(str):
        parts = [p.strip() for p in re.split(r"[,;|]+", item) if p.strip()]
        values.extend(parts if parts else [item.strip()])
    if not values:
        return default
    return pd.Series(values).value_counts().index[0]


def build_cluster_profile(df_result: pd.DataFrame) -> pd.DataFrame:
    """Membuat profil cluster dari 7 variabel utama, lalu menambahkan data profil/pendukung jika tersedia."""
    base = df_result.groupby("Cluster")[FEATURE_COLUMNS].mean()
    base["Jumlah_Anggota"] = df_result.groupby("Cluster").size()

    # Skor relatif untuk pemberian label interpretatif.
    z = (base[FEATURE_COLUMNS] - df_result[FEATURE_COLUMNS].mean()) / df_result[FEATURE_COLUMNS].std(ddof=0).replace(0, 1)
    base["Skor_Nilai_Transaksi"] = z[["Saldo_Rata_Rata", "Nominal_Transaksi_Bulanan"]].mean(axis=1)
    base["Skor_Intensitas"] = z[["Frekuensi_Transaksi", "Keragaman_Fitur"]].mean(axis=1)
    base["Skor_Pengalaman"] = z[["Respons_Promosi", "Kepuasan_Layanan", "Kepercayaan_Keamanan"]].mean(axis=1)

    label_map = {}
    remaining = list(base.index)

    if remaining:
        high_value_idx = (base.loc[remaining, "Skor_Nilai_Transaksi"] + base.loc[remaining, "Skor_Intensitas"]).idxmax()
        label_map[high_value_idx] = "High-Value Digital Loyalist"
        remaining.remove(high_value_idx)

    if remaining:
        active_idx = base.loc[remaining, "Skor_Intensitas"].idxmax()
        label_map[active_idx] = "Digital Active User"
        remaining.remove(active_idx)

    if remaining:
        dormant_idx = base.loc[remaining, "Skor_Intensitas"].idxmin()
        label_map[dormant_idx] = "Dormant / Low Engagement User"
        remaining.remove(dormant_idx)

    if remaining:
        promo_idx = base.loc[remaining, "Respons_Promosi"].idxmax()
        label_map[promo_idx] = "Promo-Sensitive Reactivation Target"
        remaining.remove(promo_idx)

    for idx in remaining:
        label_map[idx] = "Regular Moderate User"

    base["Nama_Segmen"] = [label_map.get(idx, "Regular Moderate User") for idx in base.index]
    base["Rekomendasi_Strategi"] = base["Nama_Segmen"].apply(recommendation_for_segment)

    # Data profil untuk interpretasi.
    for col in ["Usia", "Pendapatan_Bulanan"]:
        if col in df_result.columns:
            base[col] = df_result.groupby("Cluster")[col].mean()

    for col in ["Jenis_Kelamin", "Jenis_Pekerjaan", "Jenis_Layanan_Digital", "Aplikasi_Digital_Banking", "Bank_Digunakan"] + SUPPORT_COLUMNS:
        if col in df_result.columns:
            base[col + "_Dominan"] = df_result.groupby("Cluster")[col].apply(dominant_text)

    base = base.reset_index()
    ordered = ["Cluster", "Nama_Segmen", "Jumlah_Anggota"] + FEATURE_COLUMNS
    optional = [c for c in ["Usia", "Pendapatan_Bulanan", "Jenis_Kelamin_Dominan", "Jenis_Pekerjaan_Dominan", "Jenis_Layanan_Digital_Dominan", "Aplikasi_Digital_Banking_Dominan", "Bank_Digunakan_Dominan"] if c in base.columns]
    support = [c + "_Dominan" for c in SUPPORT_COLUMNS if c + "_Dominan" in base.columns]
    score_cols = ["Skor_Nilai_Transaksi", "Skor_Intensitas", "Skor_Pengalaman"]
    return base[ordered + optional + support + score_cols + ["Rekomendasi_Strategi"]]


def recommendation_for_segment(name: str) -> str:
    recs = {
        "High-Value Digital Loyalist": (
            "Prioritaskan program loyalitas, reward transaksi bernilai tinggi, penawaran tabungan/deposito atau produk simpanan premium, "
            "serta komunikasi personal untuk menjaga loyalitas pengguna bernilai tinggi."
        ),
        "Digital Active User": (
            "Dorong penggunaan fitur melalui cashback QRIS, promo merchant digital, gamifikasi transaksi, edukasi fitur lanjutan, "
            "dan cross-selling layanan pembayaran digital."
        ),
        "Promo-Sensitive Reactivation Target": (
            "Gunakan promosi yang lebih personal seperti voucher reaktivasi, cashback transaksi pertama, bebas biaya transfer, "
            "dan push notification dengan batas waktu promo."
        ),
        "Dormant / Low Engagement User": (
            "Fokus pada edukasi fitur, peningkatan rasa aman, onboarding ulang, panduan penggunaan sederhana, "
            "serta promosi ringan untuk mendorong transaksi awal."
        ),
        "Regular Moderate User": (
            "Berikan edukasi fitur yang belum digunakan, promo berkala, rekomendasi layanan sesuai kebutuhan, "
            "dan program peningkatan engagement agar pengguna menjadi lebih aktif."
        ),
    }
    return recs.get(name, "Lakukan strategi pemasaran personal berdasarkan karakteristik cluster.")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def ensure_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Customer_ID" not in result.columns:
        result.insert(0, "Customer_ID", [f"CUST-{i:04d}" for i in range(1, len(result) + 1)])
    else:
        result["Customer_ID"] = result["Customer_ID"].fillna("").astype(str).str.strip()
        missing_mask = result["Customer_ID"].eq("")
        generated_ids = [f"CUST-{i:04d}" for i in range(1, len(result) + 1)]
        result.loc[missing_mask, "Customer_ID"] = [generated_ids[i] for i in range(len(result)) if missing_mask.iloc[i]]
        ordered_cols = ["Customer_ID"] + [col for col in result.columns if col != "Customer_ID"]
        result = result[ordered_cols]
    return result


def prepare_display_table(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    money_cols = ["Saldo_Rata_Rata", "Nominal_Transaksi_Bulanan", "Pendapatan_Bulanan"]
    score_cols = [
        "Frekuensi_Transaksi",
        "Keragaman_Fitur",
        "Respons_Promosi",
        "Kepuasan_Layanan",
        "Kepercayaan_Keamanan",
        "Minat_Tabungan_Deposito",
        "Usia",
        "Skor_Nilai_Transaksi",
        "Skor_Intensitas",
        "Skor_Pengalaman",
    ]
    for col in money_cols:
        if col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce")
            if numeric.notna().any():
                result[col] = numeric.round(0)
    for col in score_cols:
        if col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce")
            if numeric.notna().any():
                result[col] = numeric.round(2)
    for col in ["Inertia", "Silhouette", "r hitung", "Batas r", "Cronbach's Alpha", "Batas Alpha"]:
        if col in result.columns:
            numeric = pd.to_numeric(result[col], errors="coerce")
            if numeric.notna().any():
                result[col] = numeric.round(3)
    return result


def make_summary_text(profile: pd.DataFrame, k: int, score: float, k_recommended: int) -> str:
    lines = [
        "RINGKASAN HASIL SEGMENTASI PENGGUNA MOBILE BANKING",
        "====================================================",
        f"Jumlah cluster yang digunakan (K): {k}",
        f"Rekomendasi K berdasarkan Silhouette: {k_recommended}",
        f"Nilai Silhouette Coefficient pada K terpilih: {score:.3f}",
        "",
        "Catatan metodologis:",
        "- Input K-Means hanya menggunakan tujuh variabel utama sesuai penulisan skripsi.",
        "- Data profil responden dan data pendukung rekomendasi tidak digunakan sebagai input clustering.",
        "- Label segmen diberikan berdasarkan interpretasi rata-rata karakteristik setiap cluster.",
        "",
        "Interpretasi cluster:",
    ]
    for _, row in profile.iterrows():
        lines.extend(
            [
                f"\nCluster {int(row['Cluster'])} - {row['Nama_Segmen']}",
                f"Jumlah anggota: {int(row['Jumlah_Anggota'])}",
                "Rata-rata variabel utama:",
            ]
        )
        for col in FEATURE_COLUMNS:
            lines.append(f"- {DISPLAY_NAMES[col]}: {row[col]:.2f}")
        if "Usia" in profile.columns:
            lines.append(f"Profil pendukung - Usia rata-rata: {row['Usia']:.1f} tahun")
        if "Pendapatan_Bulanan" in profile.columns:
            lines.append(f"Profil pendukung - Pendapatan rata-rata: Rp {row['Pendapatan_Bulanan']:,.0f}")
        if "Jenis_Promosi_Disukai_Dominan" in profile.columns:
            lines.append(f"Promosi dominan: {row['Jenis_Promosi_Disukai_Dominan']}")
        if "Kendala_Penggunaan_Dominan" in profile.columns:
            lines.append(f"Kendala dominan: {row['Kendala_Penggunaan_Dominan']}")
        lines.append(f"Rekomendasi strategi: {row['Rekomendasi_Strategi']}")
    return "\n".join(lines)


def quality_label(score: float) -> str:
    if np.isnan(score):
        return "Nilai Silhouette belum tersedia."
    if score < 0:
        return "Kualitas cluster kurang baik. Terdapat kemungkinan data kurang sesuai dengan cluster."
    if score < 0.25:
        return "Kualitas cluster rendah. Perlu evaluasi ulang variabel, data, atau jumlah cluster."
    if score < 0.50:
        return "Kualitas cluster sedang. Hasil masih dapat digunakan dengan dukungan interpretasi karakteristik cluster."
    if score < 0.70:
        return "Kualitas cluster cukup baik. Pemisahan antar cluster terlihat cukup jelas."
    return "Kualitas cluster baik. Pemisahan antar cluster terlihat kuat."


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 1. Input Data")
st.sidebar.caption("Upload file CSV/XLSX hasil kuesioner penelitian.")

uploaded_file = st.sidebar.file_uploader("Upload dataset penelitian", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            raw_df = pd.read_csv(uploaded_file)
        else:
            raw_df = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"File tidak dapat dibaca: {exc}")
        st.stop()
else:
    raw_df = pd.DataFrame()

st.sidebar.markdown("## 2. Parameter Clustering")
auto_k = st.sidebar.checkbox("Gunakan rekomendasi K dari Silhouette", value=False)
manual_k = st.sidebar.slider("Jumlah cluster (K)", min_value=2, max_value=8, value=4, step=1)

st.sidebar.markdown("## 3. Uji Instrumen Kuesioner")
validity_threshold = st.sidebar.number_input(
    "Batas r validitas", min_value=0.00, max_value=1.00, value=0.30, step=0.01
)
reliability_threshold = st.sidebar.number_input(
    "Batas Cronbach's Alpha", min_value=0.00, max_value=1.00, value=0.70, step=0.01
)

# =========================================================
# INFORMASI FORMAT DATA
# =========================================================
with st.expander("Format Kolom Dataset yang Disarankan", expanded=True):
    st.markdown(
        """
        <div class="small-note">
        <b>Catatan penting:</b> sesuai penulisan skripsi, K-Means hanya memakai <b>7 variabel utama</b>.
        Data profil responden dan data pendukung rekomendasi hanya dipakai untuk interpretasi hasil cluster.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Input utama K-Means**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Kolom Program": FEATURE_COLUMNS,
                    "Keterangan": [DISPLAY_NAMES[c] for c in FEATURE_COLUMNS],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col2:
        st.markdown("**Data profil responden**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Kolom Program": PROFILE_COLUMNS,
                    "Fungsi": ["Interpretasi"] * len(PROFILE_COLUMNS),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with col3:
        st.markdown("**Data pendukung rekomendasi**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Kolom Program": SUPPORT_COLUMNS,
                    "Fungsi": ["Rekomendasi"] * len(SUPPORT_COLUMNS),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# VALIDASI DAN PREPROCESSING
# =========================================================
if raw_df.empty:
    st.info("Silakan upload file CSV/XLSX hasil kuesioner untuk menjalankan analisis.")
    st.stop()

processed_df, scaled_df, X_scaled, scaler, missing_cols = preprocess_data(raw_df)
processed_df = ensure_customer_id(processed_df)

if missing_cols:
    st.error("Dataset belum memiliki kolom variabel utama yang diperlukan untuk K-Means.")
    st.write("Kolom yang belum ditemukan:", missing_cols)
    st.write("Kolom yang tersedia setelah penyesuaian nama:", list(processed_df.columns))
    st.stop()

if len(processed_df) <= manual_k:
    st.error("Jumlah data harus lebih besar dari jumlah cluster.")
    st.stop()

# Evaluasi K
k_scores = calculate_k_scores(X_scaled, k_min=2, k_max=8)
recommended_k = choose_best_k(k_scores)
selected_k = recommended_k if auto_k else manual_k

if selected_k >= len(processed_df):
    selected_k = max(2, len(processed_df) - 1)

model, labels, sil_score = fit_kmeans(X_scaled, selected_k)
result_df = processed_df.copy()
result_df["Cluster"] = labels
profile_df = build_cluster_profile(result_df)

# Merge nama segmen ke data hasil
segment_map = dict(zip(profile_df["Cluster"], profile_df["Nama_Segmen"]))
result_df["Nama_Segmen"] = result_df["Cluster"].map(segment_map)

# =========================================================
# RINGKASAN ATAS
# =========================================================
st.markdown("## Ringkasan Hasil Analisis")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Jumlah Data", f"{len(result_df):,}".replace(",", "."))
m2.metric("Variabel K-Means", len(FEATURE_COLUMNS))
m3.metric("Jumlah Cluster", selected_k)
m4.metric("Silhouette", f"{sil_score:.3f}")
m5.metric("Rekomendasi K", recommended_k)

st.info(quality_label(sil_score))

st.markdown(
    """
    <div class="small-note">
    <b>Penegasan metodologi:</b> variabel yang masuk ke proses K-Means adalah 7 variabel utama.
    Usia, jenis kelamin, pekerjaan, pendapatan bulanan, jenis layanan, aplikasi digital banking, dan data pendukung rekomendasi tidak ikut dihitung sebagai input cluster.
    Data tersebut hanya membantu interpretasi karakteristik cluster dan penyusunan rekomendasi strategi pemasaran.
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# TAB ANALISIS
# =========================================================
tab1, tab_instrumen, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data", "Uji Instrumen", "Elbow dan Silhouette", "Profil Cluster", "Visualisasi", "Rekomendasi Strategi"]
)

with tab1:
    st.markdown("### Dataset Setelah Preprocessing")
    st.caption("Data variabel utama telah dikonversi ke bentuk numerik dan nilai kosong diisi menggunakan median.")
    st.dataframe(prepare_display_table(result_df).head(30), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            label="Download hasil clustering CSV",
            data=dataframe_to_csv_bytes(prepare_display_table(result_df)),
            file_name="hasil_clustering_pengguna_mobile_banking.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            label="Download profil cluster CSV",
            data=dataframe_to_csv_bytes(prepare_display_table(profile_df)),
            file_name="profil_cluster_pengguna_mobile_banking.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            label="Download data terstandardisasi CSV",
            data=dataframe_to_csv_bytes(prepare_display_table(scaled_df.assign(Customer_ID=result_df["Customer_ID"].values, Cluster=labels, Nama_Segmen=result_df["Nama_Segmen"].values))),
            file_name="data_standardisasi_clustering.csv",
            mime="text/csv",
        )

with tab_instrumen:
    st.markdown("### Uji Instrumen Kuesioner")
    st.caption(
        "Uji instrumen ini ditambahkan sesuai kebutuhan Bab IV: Uji Validitas dan Uji Reliabilitas. "
        "Perhitungan dilakukan pada item skala Likert/persepsi yang digunakan sebagai indikator terpisah."
    )

    instrument_df = get_instrument_data(processed_df)
    if instrument_df.empty or instrument_df.shape[1] < 2:
        st.warning(
            "Item skala Likert yang tersedia belum cukup untuk menghitung uji validitas dan reliabilitas. "
            "Pastikan kolom Respons terhadap Promosi, Kepuasan Layanan, Kepercayaan Keamanan, "
            "atau Minat Tabungan/Deposito terbaca oleh program."
        )
    else:
        st.markdown(
            """
            <div class="small-note">
            <b>Catatan:</b> Uji validitas dihitung menggunakan korelasi item-total, sedangkan Cronbach's Alpha ditampilkan sebagai informasi tambahan.
            Item pada kuesioner ini digunakan sebagai indikator terpisah dalam proses clustering, bukan sebagai satu konstruk pengukuran tunggal. Oleh karena itu, hasil reliabilitas gabungan tidak dijadikan dasar utama untuk menghapus variabel penelitian.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Data Item Skala yang Diuji")
        st.dataframe(
            instrument_df.rename(columns={c: DISPLAY_NAMES.get(c, c) for c in instrument_df.columns}).head(30),
            use_container_width=True,
        )

        st.markdown("#### Uji Validitas")
        validity_df = calculate_validity_table(instrument_df, validity_threshold)
        st.dataframe(prepare_display_table(validity_df), use_container_width=True, hide_index=True)

        st.markdown("#### Uji Reliabilitas")
        alpha_value = cronbach_alpha(instrument_df)
        rel_status = reliability_label(alpha_value, reliability_threshold)

        r1, r2, r3 = st.columns(3)
        r1.metric("Jumlah Item Skala", instrument_df.shape[1])
        r2.metric("Cronbach's Alpha", "-" if np.isnan(alpha_value) else f"{alpha_value:.3f}")
        r3.metric("Keterangan", rel_status)

        reliability_df = pd.DataFrame(
            [
                {
                    "Jumlah Item": instrument_df.shape[1],
                    "Jumlah Responden": instrument_df.shape[0],
                    "Cronbach's Alpha": alpha_value,
                    "Batas Alpha": reliability_threshold,
                    "Keterangan": rel_status,
                }
            ]
        )
        st.dataframe(prepare_display_table(reliability_df), use_container_width=True, hide_index=True)

        c_valid, c_rel = st.columns(2)
        with c_valid:
            st.download_button(
                label="Download hasil uji validitas CSV",
                data=dataframe_to_csv_bytes(prepare_display_table(validity_df)),
                file_name="uji_validitas_kuesioner.csv",
                mime="text/csv",
            )
        with c_rel:
            st.download_button(
                label="Download hasil uji reliabilitas CSV",
                data=dataframe_to_csv_bytes(prepare_display_table(reliability_df)),
                file_name="uji_reliabilitas_kuesioner.csv",
                mime="text/csv",
            )

        st.markdown(
            """
            <div class="warning-note">
            Hasil uji instrumen ini digunakan sebagai pendukung Bab IV dan perlu diinterpretasikan secara hati-hati karena item berasal dari indikator yang berbeda. Jika dosen meminta r tabel berdasarkan jumlah responden tertentu, batas r validitas pada sidebar dapat disesuaikan.
            </div>
            """,
            unsafe_allow_html=True,
        )


with tab2:
    st.markdown("### Evaluasi Jumlah Cluster")
    fig_elbow = go.Figure()
    fig_elbow.add_trace(
        go.Scatter(
            x=k_scores["K"],
            y=k_scores["Inertia"],
            mode="lines+markers",
            name="Inertia",
        )
    )
    fig_elbow.update_layout(
        title="Elbow Method - Inertia per K",
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="Inertia",
        height=430,
    )

    fig_sil = go.Figure()
    fig_sil.add_trace(
        go.Scatter(
            x=k_scores["K"],
            y=k_scores["Silhouette"],
            mode="lines+markers",
            name="Silhouette",
        )
    )
    fig_sil.update_layout(
        title="Silhouette Coefficient per K",
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="Silhouette",
        height=430,
    )

    col_elbow, col_sil = st.columns(2)
    with col_elbow:
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.caption("Inertia yang semakin kecil menunjukkan jarak data ke centroid semakin rendah. Titik siku dapat menjadi pertimbangan jumlah cluster.")
    with col_sil:
        st.plotly_chart(fig_sil, use_container_width=True)
        st.caption(f"Nilai Silhouette tertinggi pada data ini berada pada K = {recommended_k}.")

    st.markdown("### Tabel Evaluasi K")
    st.dataframe(prepare_display_table(k_scores), use_container_width=True)

    st.markdown(
        f"""
        <div class="warning-note">
        Pada data yang sedang dianalisis, rekomendasi K berdasarkan Silhouette adalah <b>{recommended_k}</b>.
        Jika jumlah cluster yang dipakai berbeda, jelaskan bahwa pemilihan K juga mempertimbangkan kebutuhan interpretasi segmentasi dan strategi pemasaran.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab3:
    st.markdown("### Profil Cluster")
    st.caption("Profil cluster dihitung berdasarkan rata-rata variabel utama. Data profil dan data pendukung digunakan sebagai interpretasi tambahan.")
    st.dataframe(prepare_display_table(profile_df), use_container_width=True)

    fig_count = px.bar(
        profile_df,
        x="Nama_Segmen",
        y="Jumlah_Anggota",
        color="Nama_Segmen",
        title="Jumlah Anggota Setiap Cluster",
        text="Jumlah_Anggota",
    )
    fig_count.update_layout(showlegend=False, height=430, xaxis_title="Segmen", yaxis_title="Jumlah Anggota")
    st.plotly_chart(fig_count, use_container_width=True)

    profile_scaled = profile_df[["Nama_Segmen"] + FEATURE_COLUMNS].copy()
    for col in FEATURE_COLUMNS:
        mean = result_df[col].mean()
        std = result_df[col].std(ddof=0) or 1
        profile_scaled[col] = (profile_scaled[col] - mean) / std
    heatmap_data = profile_scaled.set_index("Nama_Segmen")[FEATURE_COLUMNS]
    fig_heat = px.imshow(
        heatmap_data,
        labels=dict(x="Variabel Utama", y="Segmen", color="Skor Relatif"),
        x=[DISPLAY_NAMES[c] for c in FEATURE_COLUMNS],
        y=heatmap_data.index,
        aspect="auto",
        title="Heatmap Profil Cluster Berdasarkan 7 Variabel Utama",
    )
    fig_heat.update_layout(height=430)
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.markdown("### Scatter Plot Antar Variabel Utama")
    st.caption("Visualisasi menggunakan variabel asli yang menjadi input K-Means, tanpa transformasi PCA.")
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("Variabel X", FEATURE_COLUMNS, format_func=lambda x: DISPLAY_NAMES[x], index=0)
    with col_y:
        y_axis = st.selectbox("Variabel Y", FEATURE_COLUMNS, format_func=lambda x: DISPLAY_NAMES[x], index=1)

    hover_cols = ["Cluster"] + FEATURE_COLUMNS
    for opt in ["Usia", "Jenis_Kelamin", "Pendapatan_Bulanan", "Jenis_Pekerjaan", "Jenis_Promosi_Disukai", "Kendala_Penggunaan", "Aktivitas_Pembayaran_Digital"]:
        if opt in result_df.columns:
            hover_cols.append(opt)

    fig_scatter = px.scatter(
        result_df,
        x=x_axis,
        y=y_axis,
        color="Nama_Segmen",
        hover_data=hover_cols,
        title=f"Scatter Plot {DISPLAY_NAMES[x_axis]} vs {DISPLAY_NAMES[y_axis]}",
    )
    fig_scatter.update_layout(height=520)
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab5:
    st.markdown("### Interpretasi dan Rekomendasi Strategi Pemasaran")
    st.caption("Rekomendasi disusun berdasarkan profil rata-rata cluster dan diperkuat oleh data pendukung yang tersedia.")

    for _, row in profile_df.iterrows():
        profile_parts = []
        if "Usia" in row.index and not pd.isna(row["Usia"]):
            profile_parts.append(f"usia rata-rata {row['Usia']:.1f} tahun")
        if "Pendapatan_Bulanan" in row.index and not pd.isna(row["Pendapatan_Bulanan"]):
            profile_parts.append(f"pendapatan rata-rata Rp {row['Pendapatan_Bulanan']:,.0f}".replace(",", "."))
        if "Jenis_Pekerjaan_Dominan" in row.index:
            profile_parts.append(f"pekerjaan dominan {row['Jenis_Pekerjaan_Dominan']}")

        support_parts = []
        if "Jenis_Promosi_Disukai_Dominan" in row.index:
            support_parts.append(f"promosi dominan: {row['Jenis_Promosi_Disukai_Dominan']}")
        if "Kendala_Penggunaan_Dominan" in row.index:
            support_parts.append(f"kendala dominan: {row['Kendala_Penggunaan_Dominan']}")
        if "Minat_Tabungan_Deposito_Dominan" in row.index:
            support_parts.append(f"minat tabungan/deposito: {row['Minat_Tabungan_Deposito_Dominan']}")
        if "Media_Promosi_Diperhatikan_Dominan" in row.index:
            support_parts.append(f"media promosi dominan: {row['Media_Promosi_Diperhatikan_Dominan']}")
        if "Aktivitas_Pembayaran_Digital_Dominan" in row.index:
            support_parts.append(f"aktivitas pembayaran digital dominan: {row['Aktivitas_Pembayaran_Digital_Dominan']}")

        st.markdown(
            f"""
            <div class="section-card">
                <h4>Cluster {int(row['Cluster'])} - {row['Nama_Segmen']}</h4>
                <p><b>Jumlah anggota:</b> {int(row['Jumlah_Anggota'])} pengguna</p>
                <p><b>Karakteristik utama:</b> saldo rata-rata Rp {row['Saldo_Rata_Rata']:,.0f}, frekuensi transaksi {row['Frekuensi_Transaksi']:.1f}, nominal transaksi bulanan Rp {row['Nominal_Transaksi_Bulanan']:,.0f}, keragaman fitur {row['Keragaman_Fitur']:.1f}, respons promosi {row['Respons_Promosi']:.1f}, kepuasan layanan {row['Kepuasan_Layanan']:.1f}, dan kepercayaan keamanan {row['Kepercayaan_Keamanan']:.1f}.</p>
                <p><b>Profil pendukung:</b> {", ".join(profile_parts) if profile_parts else "data profil tidak tersedia"}.</p>
                <p><b>Data pendukung rekomendasi:</b> {", ".join(support_parts) if support_parts else "data pendukung tidak tersedia"}.</p>
                <p><b>Rekomendasi strategi:</b> {row['Rekomendasi_Strategi']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    summary_text = make_summary_text(profile_df, selected_k, sil_score, recommended_k)
    st.download_button(
        label="Download ringkasan interpretasi TXT",
        data=summary_text.encode("utf-8"),
        file_name="ringkasan_interpretasi_segmentasi.txt",
        mime="text/plain",
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Program ini dibuat untuk kebutuhan penelitian skripsi: segmentasi pengguna mobile banking berdasarkan perilaku penggunaan 3 bulan terakhir menggunakan K-Means Clustering."
)