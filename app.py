from __future__ import annotations

import base64
import io
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
PAGE_TITLE = "Segmentasi Pengguna Mobile Banking"
GREEN = "#004d24"
LEVEL_THRESHOLD = 0.50  # satuan simpangan baku terhadap rata-rata seluruh data

FEATURE_COLUMNS = [
    "Saldo_Rata_Rata",
    "Frekuensi_Transaksi",
    "Nominal_Transaksi_Bulanan",
    "Keragaman_Fitur",
    "Respons_Promosi",
    "Kepuasan_Layanan",
    "Kepercayaan_Keamanan",
]

PROFILE_COLUMNS = [
    "Customer_ID",
    "Usia",
    "Jenis_Kelamin",
    "Jenis_Pekerjaan",
    "Pendapatan_Bulanan",
    "Jenis_Layanan_Digital",
    "Aplikasi_Digital_Banking",
]

SUPPORT_COLUMNS = [
    "Tujuan_Penggunaan",
    "Tujuan_Keuangan",
    "Faktor_Peningkatan_Minat_Awal",
    "Minat_Tabungan_Deposito",
    "Jenis_Promosi_Disukai",
    "Preferensi_Promo_Tabungan_Deposito",
    "Faktor_Minat_Tabungan_Deposito",
    "Media_Promosi_Diperhatikan",
    "Kendala_Penggunaan",
    "Faktor_Peningkatan_Minat",
    "Aktivitas_Pembayaran_Digital",
]

SCREENING_COLUMNS = [
    "Bersedia_Menjadi_Responden",
    "Pernah_Menggunakan_Mobile_Banking",
    "Transaksi_3_Bulan_Terakhir",
]

DISPLAY_NAMES = {
    "Saldo_Rata_Rata": "Saldo rata-rata",
    "Frekuensi_Transaksi": "Frekuensi transaksi",
    "Nominal_Transaksi_Bulanan": "Nominal transaksi bulanan",
    "Keragaman_Fitur": "Keragaman fitur",
    "Respons_Promosi": "Respons terhadap promosi",
    "Kepuasan_Layanan": "Kepuasan layanan",
    "Kepercayaan_Keamanan": "Kepercayaan terhadap keamanan",
    "Usia": "Usia",
    "Jenis_Kelamin": "Jenis kelamin",
    "Jenis_Pekerjaan": "Pekerjaan",
    "Pendapatan_Bulanan": "Pendapatan bulanan",
    "Jenis_Layanan_Digital": "Jenis layanan digital",
    "Aplikasi_Digital_Banking": "Aplikasi digital banking",
}

COLUMN_ALIASES: Dict[str, List[str]] = {
    "Customer_ID": [
        "customer_id",
        "customerid",
        "id_responden",
        "kode_responden",
        "nomor_responden",
    ],
    "Bersedia_Menjadi_Responden": [
        "apakah_anda_bersedia_menjadi_responden_penelitian_ini",
        "bersedia_menjadi_responden",
        "persetujuan_responden",
    ],
    "Pernah_Menggunakan_Mobile_Banking": [
        "apakah_anda_pernah_menggunakan_layanan_digital_banking_mobile_banking",
        "apakah_anda_pernah_menggunakan_aplikasi_mobile_banking",
        "pernah_menggunakan_mobile_banking",
        "pengguna_mobile_banking",
    ],
    "Transaksi_3_Bulan_Terakhir": [
        "apakah_anda_pernah_melakukan_transaksi_melalui_digital_banking_mobile_banking",
        "apakah_anda_melakukan_transaksi_melalui_mobile_banking_dalam_3_bulan_terakhir",
        "transaksi_3_bulan_terakhir",
        "melakukan_transaksi_3_bulan_terakhir",
    ],
    "Jenis_Layanan_Digital": [
        "jenis_layanan_digital_banking_yang_paling_sering_anda_gunakan",
        "jenis_layanan_digital",
        "jenis_layanan",
    ],
    "Aplikasi_Digital_Banking": [
        "nama_aplikasi_layanan_digital_banking_yang_paling_sering_anda_gunakan",
        "aplikasi_layanan_digital_banking",
        "aplikasi_digital_banking",
        "nama_aplikasi",
        "bank_digunakan",
        "mobile_banking_yang_digunakan",
    ],
    "Usia": [
        "usia_anda_saat_ini",
        "usia",
        "umur",
        "age",
    ],
    "Jenis_Kelamin": [
        "jenis_kelamin",
        "gender",
    ],
    "Jenis_Pekerjaan": [
        "pekerjaan_saat_ini",
        "jenis_pekerjaan",
        "pekerjaan",
        "profesi",
        "occupation",
        "job",
    ],
    "Pendapatan_Bulanan": [
        "pendapatan_bulanan_anda",
        "pendapatan_bulanan",
        "penghasilan_bulanan",
        "pendapatan",
        "penghasilan",
        "income",
        "monthly_income",
    ],
    "Saldo_Rata_Rata": [
        "saldo_rata_rata_dalam_3_bulan_terakhir_berapa_rata_rata_saldo",
        "saldo_rata_rata",
        "rata_rata_saldo",
        "average_balance",
        "balance",
    ],
    "Frekuensi_Transaksi": [
        "frekuensi_transaksi_dalam_3_bulan_terakhir_berapa_kali_rata_rata_anda_melakukan_transaksi",
        "frekuensi_transaksi",
        "jumlah_transaksi",
        "berapa_kali_rata_rata_anda_melakukan_transaksi",
        "transaction_frequency",
        "freq",
    ],
    "Nominal_Transaksi_Bulanan": [
        "nominal_transaksi_bulanan_dalam_3_bulan_terakhir_berapa_perkiraan_total_nominal_transaksi",
        "nominal_transaksi_bulanan",
        "total_nominal_transaksi",
        "nominal_transaksi",
        "transaction_amount",
        "monthly_transaction_value",
    ],
    "Keragaman_Fitur": [
        "keragaman_fitur_yang_digunakan",
        "keragaman_fitur",
        "jumlah_fitur",
        "feature_diversity",
    ],
    "Fitur_Digunakan": [
        "keragaman_fitur_yang_digunakan_fitur_digital_banking_mobile_banking_apa_saja_yang_anda_gunakan",
        "fitur_digital_banking_mobile_banking_apa_saja_yang_anda_gunakan",
        "fitur_apa_saja_yang_anda_gunakan",
        "fitur_digunakan",
        "used_features",
    ],
    "Respons_Promosi": [
        "respons_terhadap_promosi_saya_tertarik_menggunakan_digital_banking_mobile_banking_ketika_terdapat_promosi",
        "respons_terhadap_promosi",
        "respons_promosi",
        "respon_promosi",
        "promotion_response",
    ],
    "Kepuasan_Layanan": [
        "kepuasan_layanan_saya_merasa_puas_dengan_kemudahan_kecepatan_dan_kenyamanan_layanan",
        "kepuasan_layanan",
        "kepuasan",
        "service_satisfaction",
        "satisfaction",
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
    ],
    "Tujuan_Keuangan": [
        "dalam_beberapa_bulan_ke_depan_tujuan_keuangan_yang_ingin_anda_capai_adalah",
        "tujuan_keuangan",
    ],
    "Faktor_Peningkatan_Minat_Awal": [
        "menurut_anda_hal_apa_yang_paling_dapat_meningkatkan_minat_penggunaan_digital",
        "faktor_peningkatan_minat_awal",
    ],
    "Minat_Tabungan_Deposito": [
        "saya_tertarik_menggunakan_fitur_tabungan_digital_tabungan_berjangka_atau_deposito",
        "minat_tabungan_deposito",
        "minat_produk_simpanan",
    ],
    "Jenis_Promosi_Disukai": [
        "jenis_promosi_digital_banking_mobile_banking_yang_menarik_bagi_anda_adalah",
        "jenis_promosi_disukai",
        "promosi_yang_disukai",
    ],
    "Preferensi_Promo_Tabungan_Deposito": [
        "jika_terdapat_promo_deposito_atau_tabungan_digital_bentuk_promo_yang_paling",
        "preferensi_promo_tabungan_deposito",
    ],
    "Faktor_Minat_Tabungan_Deposito": [
        "faktor_yang_dapat_membuat_anda_tertarik_menggunakan_produk_tabungan_digital",
        "faktor_minat_tabungan_deposito",
    ],
    "Media_Promosi_Diperhatikan": [
        "media_promosi_digital_banking_mobile_banking_yang_sering_anda_perhatikan",
        "media_promosi_diperhatikan",
        "media_promosi_yang_diperhatikan",
    ],
    "Kendala_Penggunaan": [
        "kendala_yang_pernah_anda_alami_saat_menggunakan_digital_banking_mobile_banking",
        "kendala_penggunaan",
        "kendala_mobile_banking",
        "hambatan_penggunaan",
    ],
    "Faktor_Peningkatan_Minat": [
        "hal_yang_dapat_meningkatkan_minat_anda_dalam_menggunakan_digital_banking_mobile_banking",
        "faktor_peningkatan_minat",
        "peningkatan_minat_penggunaan",
    ],
    "Aktivitas_Pembayaran_Digital": [
        "dalam_penggunaan_digital_banking_mobile_banking_aktivitas_pembayaran_digital",
        "aktivitas_pembayaran_digital",
        "aktivitas_pembayaran_digital_yang_sering_dilakukan",
    ],
}

LIKERT_MAP = {
    "sangat_tidak_setuju": 1,
    "tidak_setuju": 2,
    "netral": 3,
    "ragu_ragu": 3,
    "cukup": 3,
    "setuju": 4,
    "sangat_setuju": 5,
    "sangat_tidak_tertarik": 1,
    "tidak_tertarik": 2,
    "cukup_tertarik": 3,
    "tertarik": 4,
    "sangat_tertarik": 5,
    "sangat_tidak_puas": 1,
    "tidak_puas": 2,
    "cukup_puas": 3,
    "puas": 4,
    "sangat_puas": 5,
    "sangat_tidak_percaya": 1,
    "tidak_percaya": 2,
    "cukup_percaya": 3,
    "percaya": 4,
    "sangat_percaya": 5,
}


@dataclass
class CleaningReport:
    baris_awal: int
    baris_kosong_dihapus: int
    duplikat_dihapus: int
    tidak_lolos_screening: int
    usia_tidak_memenuhi: int
    fitur_utama_tidak_lengkap: int
    baris_valid: int
    outlier_ekstrem_terdeteksi: int
    kolom_konstan: List[str]

def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def canonicalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Mencocokkan kolom Google Forms dengan nama kolom kanonik penelitian."""
    normalized_original = {column: normalize_text(column) for column in df.columns}
    rename_map: Dict[str, str] = {}
    used_original: set = set()

    # Kolom yang sangat umum diproses belakangan untuk mencegah salah cocok.
    targets = list(COLUMN_ALIASES.keys())
    targets.sort(
        key=lambda target: max(len(normalize_text(a)) for a in COLUMN_ALIASES[target]),
        reverse=True,
    )

    for target in targets:
        aliases = [normalize_text(target)] + [
            normalize_text(alias) for alias in COLUMN_ALIASES[target]
        ]
        aliases = sorted(set(aliases), key=len, reverse=True)
        candidates: List[Tuple[int, int, str]] = []

        for original, normalized in normalized_original.items():
            if original in used_original:
                continue
            for alias in aliases:
                if not alias:
                    continue
                score = -1
                if normalized == alias:
                    score = 10_000 + len(alias)
                elif len(alias) >= 8 and normalized.startswith(alias):
                    score = 7_000 + len(alias)
                elif len(alias) >= 8 and alias in normalized:
                    score = 4_000 + len(alias)
                if score >= 0:
                    candidates.append((score, len(alias), original))
                    break

        if candidates:
            candidates.sort(reverse=True)
            chosen = candidates[0][2]
            rename_map[chosen] = target
            used_original.add(chosen)

    return df.rename(columns=rename_map).copy(), rename_map


def _extract_numbers(raw: str) -> List[float]:
    # Pemisah titik dan koma pada jawaban rupiah diperlakukan sebagai pemisah ribuan.
    cleaned = raw.lower().replace("rp", "").replace("idr", "")
    cleaned = cleaned.replace(".", "").replace(",", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", cleaned)]


def parse_range_midpoint(value: object) -> float:
    """Mengubah angka/rentang jawaban menjadi nilai numerik representatif."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    raw = str(value).strip()
    normalized = normalize_text(raw)
    if normalized in LIKERT_MAP:
        return float(LIKERT_MAP[normalized])

    numbers = _extract_numbers(raw)
    if not numbers:
        return np.nan

    lower = raw.lower()
    multiplier = 1.0
    if "juta" in lower or re.search(r"\bjt\b", lower):
        multiplier = 1_000_000.0
    elif "ribu" in lower or re.search(r"\brb\b", lower):
        multiplier = 1_000.0
    numbers = [number * multiplier for number in numbers]

    if len(numbers) >= 2:
        return float((numbers[0] + numbers[1]) / 2.0)
    boundary = numbers[0]
    if any(token in lower for token in ["kurang dari", "di bawah", "<"]):
        return float(boundary / 2.0)
    if any(token in lower for token in ["lebih dari", "di atas", ">", "atau lebih"]):
        return float(boundary * 1.25)
    return float(boundary)


def parse_likert(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if 1 <= number <= 5 else np.nan
    normalized = normalize_text(value)
    if normalized in LIKERT_MAP:
        return float(LIKERT_MAP[normalized])
    match = re.search(r"(?<!\d)([1-5])(?!\d)", str(value))
    return float(match.group(1)) if match else np.nan


def split_choices(value: object) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [str(value)]
    raw_parts = re.split(r"[,;|]\s*", str(value).strip())
    cleaned: List[str] = []
    seen = set()
    for part in raw_parts:
        item = re.sub(r"\s+", " ", part).strip(" .")
        key = normalize_text(item)
        if item and key and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned


def count_multiselect(value: object) -> float:
    choices = split_choices(value)
    return float(len(choices)) if choices else np.nan


def parse_yes_no(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None
    normalized = normalize_text(value)
    yes_tokens = {
        "ya",
        "iya",
        "yes",
        "pernah",
        "sudah",
        "aktif",
        "ya_saya_bersedia",
    }
    no_tokens = {"tidak", "no", "belum", "tidak_pernah", "tidak_bersedia"}
    if normalized in yes_tokens or normalized.startswith("ya_"):
        return True
    if normalized in no_tokens or normalized.startswith("tidak_"):
        return False
    return None


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    return pd.read_excel(uploaded_file)


def _extreme_outlier_count(df: pd.DataFrame, columns: Sequence[str]) -> int:
    mask = pd.Series(False, index=df.index)
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            continue
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        mask |= series.notna() & ((series < lower) | (series > upper))
    return int(mask.sum())


def build_clean_dataset(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, CleaningReport, Dict[str, str]]:
    """Cleaning objektif: screening, duplikat, usia, dan kelengkapan 7 fitur."""
    df, rename_map = canonicalize_columns(raw_df)
    baris_awal = len(df)

    empty_mask = df.isna().all(axis=1)
    baris_kosong_dihapus = int(empty_mask.sum())
    df = df.loc[~empty_mask].copy()

    if "Customer_ID" not in df.columns:
        df["Customer_ID"] = [f"R{i:04d}" for i in range(1, len(df) + 1)]

    # Timestamp dan ID tidak dipakai saat mendeteksi jawaban identik.
    duplicate_exclusions = {
        "Customer_ID",
        "Timestamp",
        "timestamp",
        "Waktu_Pengisian",
    }
    duplicate_subset = [
        column for column in df.columns if column not in duplicate_exclusions
    ]
    duplicate_mask = df.duplicated(subset=duplicate_subset, keep="first")
    duplikat_dihapus = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    screening_mask = pd.Series(True, index=df.index)
    screening_used = False
    for column in SCREENING_COLUMNS:
        if column in df.columns:
            screening_used = True
            parsed = df[column].map(parse_yes_no)
            screening_mask &= parsed.fillna(False)
    if not screening_used:
        screening_mask[:] = True
    tidak_lolos_screening = int((~screening_mask).sum())
    df = df.loc[screening_mask].copy()

    usia_tidak_memenuhi = 0
    if "Usia" in df.columns:
        df["Usia"] = df["Usia"].map(parse_range_midpoint)
        age_mask = df["Usia"].isna() | (df["Usia"] >= 17)
        usia_tidak_memenuhi = int((~age_mask).sum())
        df = df.loc[age_mask].copy()

    if "Pendapatan_Bulanan" in df.columns:
        df["Pendapatan_Bulanan"] = df["Pendapatan_Bulanan"].map(
            parse_range_midpoint
        )

    for column in [
        "Saldo_Rata_Rata",
        "Frekuensi_Transaksi",
        "Nominal_Transaksi_Bulanan",
    ]:
        if column in df.columns:
            df[column] = df[column].map(parse_range_midpoint)

    if "Keragaman_Fitur" in df.columns:
        df["Keragaman_Fitur"] = df["Keragaman_Fitur"].map(parse_range_midpoint)
    elif "Fitur_Digunakan" in df.columns:
        df["Keragaman_Fitur"] = df["Fitur_Digunakan"].map(count_multiselect)

    for column in [
        "Respons_Promosi",
        "Kepuasan_Layanan",
        "Kepercayaan_Keamanan",
        "Minat_Tabungan_Deposito",
    ]:
        if column in df.columns:
            df[column] = df[column].map(parse_likert)

    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        readable = ", ".join(DISPLAY_NAMES.get(c, c) for c in missing_features)
        raise ValueError(
            "Kolom utama belum lengkap: "
            + readable
            + ". Pastikan file merupakan hasil ekspor kuesioner final."
        )

    # Batas logis dasar. Nilai di luar batas dijadikan kosong lalu baris dikeluarkan.
    logical_rules = {
        "Saldo_Rata_Rata": (0, None),
        "Frekuensi_Transaksi": (0, 1_000),
        "Nominal_Transaksi_Bulanan": (0, None),
        "Keragaman_Fitur": (1, 50),
        "Respons_Promosi": (1, 5),
        "Kepuasan_Layanan": (1, 5),
        "Kepercayaan_Keamanan": (1, 5),
    }
    for column, (lower, upper) in logical_rules.items():
        invalid = df[column].notna() & (df[column] < lower)
        if upper is not None:
            invalid |= df[column].notna() & (df[column] > upper)
        df.loc[invalid, column] = np.nan

    incomplete_mask = df[FEATURE_COLUMNS].isna().any(axis=1)
    fitur_utama_tidak_lengkap = int(incomplete_mask.sum())
    df = df.loc[~incomplete_mask].copy()

    if len(df) < 3:
        raise ValueError("Data valid kurang dari 3 baris setelah proses cleaning.")

    constant_columns = [
        column for column in FEATURE_COLUMNS if df[column].nunique(dropna=True) <= 1
    ]
    outlier_count = _extreme_outlier_count(
        df,
        [
            "Saldo_Rata_Rata",
            "Frekuensi_Transaksi",
            "Nominal_Transaksi_Bulanan",
            "Keragaman_Fitur",
        ],
    )

    report = CleaningReport(
        baris_awal=baris_awal,
        baris_kosong_dihapus=baris_kosong_dihapus,
        duplikat_dihapus=duplikat_dihapus,
        tidak_lolos_screening=tidak_lolos_screening,
        usia_tidak_memenuhi=usia_tidak_memenuhi,
        fitur_utama_tidak_lengkap=fitur_utama_tidak_lengkap,
        baris_valid=len(df),
        outlier_ekstrem_terdeteksi=outlier_count,
        kolom_konstan=constant_columns,
    )
    return df.reset_index(drop=True), report, rename_map


# =========================================================
# FUNGSI CLUSTERING DAN INTERPRETASI
# =========================================================
def evaluate_k(x_scaled: np.ndarray, max_k: int) -> pd.DataFrame:
    rows: List[dict] = []
    for k in range(2, max_k + 1):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            max_iter=300,
            random_state=42,
        )
        labels = model.fit_predict(x_scaled)
        score = np.nan
        if 1 < len(np.unique(labels)) < len(x_scaled):
            score = silhouette_score(x_scaled, labels)
        rows.append({"K": k, "Inertia": model.inertia_, "Silhouette": score})
    return pd.DataFrame(rows)


def quality_label(score: float) -> str:
    if pd.isna(score):
        return "Tidak dapat dihitung"
    if score >= 0.70:
        return "Pemisahan sangat kuat"
    if score >= 0.50:
        return "Pemisahan cukup kuat"
    if score >= 0.25:
        return "Struktur cluster dapat digunakan dengan kehati-hatian"
    return "Struktur cluster lemah"


def relative_level(z_value: float, threshold: float = LEVEL_THRESHOLD) -> str:
    if z_value >= threshold:
        return "Tinggi"
    if z_value <= -threshold:
        return "Rendah"
    return "Sedang"


def determine_base_segment_label(z_row: pd.Series) -> str:
    """Menentukan nama segmen dari centroid relatif (z-score)."""
    balance = float(z_row["Saldo_Rata_Rata"])
    frequency = float(z_row["Frekuensi_Transaksi"])
    nominal = float(z_row["Nominal_Transaksi_Bulanan"])
    feature = float(z_row["Keragaman_Fitur"])
    promotion = float(z_row["Respons_Promosi"])
    satisfaction = float(z_row["Kepuasan_Layanan"])
    security = float(z_row["Kepercayaan_Keamanan"])

    financial = float(np.mean([balance, nominal]))
    activity = float(np.mean([frequency, feature]))
    experience = float(np.mean([satisfaction, security]))
    mean_absolute_position = float(
        np.mean(np.abs(z_row[FEATURE_COLUMNS].to_numpy(dtype=float)))
    )

    # Urutan aturan dibuat agar nama mencerminkan karakteristik paling kuat.
    if financial >= LEVEL_THRESHOLD and (frequency >= LEVEL_THRESHOLD or activity >= 0.35):
        return "High-Value Active User"
    if financial >= LEVEL_THRESHOLD:
        return "High-Value Selective User"
    if feature >= LEVEL_THRESHOLD and frequency >= -0.10:
        return "Feature-Intensive Active User"
    if activity >= LEVEL_THRESHOLD:
        return "Digital Active User"
    if promotion >= LEVEL_THRESHOLD:
        return "Promo-Responsive User"
    if satisfaction <= -LEVEL_THRESHOLD and security <= -LEVEL_THRESHOLD:
        return "Service and Security-Sensitive User"
    if satisfaction <= -LEVEL_THRESHOLD:
        return "Service-Sensitive User"
    if security <= -LEVEL_THRESHOLD:
        return "Security-Sensitive User"
    if activity <= -LEVEL_THRESHOLD and financial <= -0.35:
        return "Low-Engagement User"
    if promotion <= -LEVEL_THRESHOLD:
        return "Promotion-Independent User"
    if experience >= LEVEL_THRESHOLD:
        return "Satisfied and Trusted User"
    if mean_absolute_position <= 0.40:
        return "Routine Digital User"
    return "Developing Digital User"


def assign_segment_labels(centroid_z: pd.DataFrame) -> Dict[int, str]:
    """Memberi label berbasis aturan dan membedakan label yang kebetulan sama."""
    mapping: Dict[int, str] = {}
    label_counts: Counter = Counter()
    indexed = centroid_z.set_index("Cluster")

    for cluster_id, row in indexed.iterrows():
        base = determine_base_segment_label(row)
        mapping[int(cluster_id)] = base
        label_counts[base] += 1

    # Jika dua centroid masuk arketipe sama, tambahkan fokus fitur terkuat agar
    # nama tetap informatif tanpa mengubah hasil K-Means.
    for base, count in label_counts.items():
        if count <= 1:
            continue
        cluster_ids = [cid for cid, label in mapping.items() if label == base]
        for cluster_id in cluster_ids:
            row = indexed.loc[cluster_id, FEATURE_COLUMNS]
            strongest = row.abs().idxmax()
            qualifier = DISPLAY_NAMES[strongest].title()
            mapping[cluster_id] = f"{base} – {qualifier}"
    return mapping


def build_behavior_description(z_row: pd.Series) -> str:
    high = [
        DISPLAY_NAMES[column]
        for column in FEATURE_COLUMNS
        if relative_level(float(z_row[column])) == "Tinggi"
    ]
    medium = [
        DISPLAY_NAMES[column]
        for column in FEATURE_COLUMNS
        if relative_level(float(z_row[column])) == "Sedang"
    ]
    low = [
        DISPLAY_NAMES[column]
        for column in FEATURE_COLUMNS
        if relative_level(float(z_row[column])) == "Rendah"
    ]

    sentences: List[str] = []
    if high:
        sentences.append(
            "Dibandingkan rata-rata seluruh responden, level tinggi terdapat pada "
            + ", ".join(high)
            + "."
        )
    if medium:
        prefix = (
            "Sementara itu, level sedang terdapat pada "
            if sentences
            else "Dibandingkan rata-rata seluruh responden, level sedang terdapat pada "
        )
        sentences.append(prefix + ", ".join(medium) + ".")
    if low:
        sentences.append("Level rendah terlihat pada " + ", ".join(low) + ".")

    balance = float(z_row["Saldo_Rata_Rata"])
    frequency = float(z_row["Frekuensi_Transaksi"])
    nominal = float(z_row["Nominal_Transaksi_Bulanan"])
    feature = float(z_row["Keragaman_Fitur"])
    promotion = float(z_row["Respons_Promosi"])
    satisfaction = float(z_row["Kepuasan_Layanan"])
    security = float(z_row["Kepercayaan_Keamanan"])

    financial = float(np.mean([balance, nominal]))
    activity = float(np.mean([frequency, feature]))
    experience = float(np.mean([satisfaction, security]))

    interpretation: List[str] = []
    if financial >= LEVEL_THRESHOLD and (frequency >= LEVEL_THRESHOLD or activity >= 0.35):
        interpretation.append(
            "menggabungkan nilai finansial yang tinggi dengan transaksi yang aktif"
        )
    elif financial >= LEVEL_THRESHOLD:
        interpretation.append(
            "memiliki nilai finansial tinggi, tetapi penggunaan layanan cenderung lebih selektif"
        )
    elif feature >= LEVEL_THRESHOLD and frequency >= -0.10:
        interpretation.append(
            "menonjol pada luasnya pemanfaatan fitur meskipun nilai finansialnya tidak termasuk yang tertinggi"
        )
    elif activity >= LEVEL_THRESHOLD:
        interpretation.append("aktif menggunakan layanan digital banking")
    elif activity <= -LEVEL_THRESHOLD:
        interpretation.append("masih memiliki keterlibatan digital yang rendah")
    else:
        interpretation.append("menunjukkan pola penggunaan rutin di sekitar rata-rata sampel")

    if promotion >= LEVEL_THRESHOLD:
        interpretation.append("dan cukup responsif terhadap promosi")
    elif promotion <= -LEVEL_THRESHOLD:
        interpretation.append("serta cenderung tidak menjadikan promosi sebagai pendorong utama")

    if satisfaction <= -LEVEL_THRESHOLD:
        interpretation.append("dengan kepuasan layanan yang masih perlu ditingkatkan")
    elif security <= -LEVEL_THRESHOLD:
        interpretation.append("dengan kepercayaan keamanan yang masih perlu diperkuat")
    elif experience >= LEVEL_THRESHOLD:
        interpretation.append("disertai kepuasan dan kepercayaan yang positif")

    sentences.append("Secara umum, kelompok ini " + " ".join(interpretation) + ".")
    return " ".join(sentences)


def _format_top_categories(
    series: pd.Series,
    top_n: int = 3,
    split_multi: bool = False,
) -> List[Tuple[str, float, int]]:
    valid = series.dropna().astype(str).str.strip()
    valid = valid[valid != ""]
    denominator = len(valid)
    if denominator == 0:
        return []

    counter: Counter = Counter()
    if split_multi:
        for value in valid:
            for choice in split_choices(value):
                counter[choice] += 1
    else:
        counter.update(valid.tolist())

    return [
        (label, count / denominator * 100.0, count)
        for label, count in counter.most_common(top_n)
    ]


def format_distribution(items: Sequence[Tuple[str, float, int]]) -> str:
    return ", ".join(f"{label} {percentage:.1f}%" for label, percentage, _ in items)


def build_profile_summary(cluster_df: pd.DataFrame) -> str:
    sentences: List[str] = []

    if "Usia" in cluster_df.columns:
        ages = pd.to_numeric(cluster_df["Usia"], errors="coerce").dropna()
        if not ages.empty:
            sentences.append(
                "Usia rata-rata "
                f"{ages.mean():.1f} tahun, median {ages.median():.0f} tahun, "
                f"dengan rentang {ages.min():.0f}–{ages.max():.0f} tahun"
            )

    if "Pendapatan_Bulanan" in cluster_df.columns:
        income = pd.to_numeric(cluster_df["Pendapatan_Bulanan"], errors="coerce").dropna()
        if not income.empty:
            sentences.append(
                f"pendapatan rata-rata {format_rupiah(income.mean())} "
                f"dan median {format_rupiah(income.median())}"
            )

    if "Jenis_Pekerjaan" in cluster_df.columns:
        top_jobs = _format_top_categories(cluster_df["Jenis_Pekerjaan"], top_n=3)
        if top_jobs:
            top_share = top_jobs[0][1]
            if top_share > 50:
                wording = "komposisi pekerjaan didominasi oleh "
            elif top_share >= 35:
                wording = "kategori pekerjaan terbesar adalah "
            else:
                wording = "komposisi pekerjaan beragam, dengan kategori terbanyak "
            sentences.append(wording + format_distribution(top_jobs))

    if "Jenis_Kelamin" in cluster_df.columns:
        top_gender = _format_top_categories(cluster_df["Jenis_Kelamin"], top_n=2)
        if top_gender:
            sentences.append("komposisi jenis kelamin: " + format_distribution(top_gender))

    if not sentences:
        return "Data profil pendukung tidak tersedia."
    return "; ".join(sentences) + "."


def _support_top(
    cluster_df: pd.DataFrame,
    column: str,
    top_n: int = 2,
    split_multi: bool = True,
) -> List[Tuple[str, float, int]]:
    if column not in cluster_df.columns:
        return []
    return _format_top_categories(cluster_df[column], top_n=top_n, split_multi=split_multi)


def build_support_summary(cluster_df: pd.DataFrame) -> Tuple[str, Dict[str, str]]:
    summaries: List[str] = []
    context: Dict[str, str] = {}

    mapping = [
        ("Jenis_Promosi_Disukai", "promosi yang paling banyak dipilih", "promo"),
        ("Media_Promosi_Diperhatikan", "kanal promosi yang paling diperhatikan", "media"),
        ("Kendala_Penggunaan", "kendala yang paling sering disebut", "kendala"),
        ("Aktivitas_Pembayaran_Digital", "aktivitas pembayaran yang paling sering dilakukan", "aktivitas"),
        ("Tujuan_Keuangan", "tujuan keuangan yang paling banyak disebut", "tujuan"),
    ]
    for column, label, key in mapping:
        top = _support_top(cluster_df, column, top_n=2, split_multi=True)
        if top:
            summaries.append(label + ": " + format_distribution(top))
            context[key] = top[0][0]

    if "Minat_Tabungan_Deposito" in cluster_df.columns:
        interest = pd.to_numeric(
            cluster_df["Minat_Tabungan_Deposito"], errors="coerce"
        ).dropna()
        if not interest.empty:
            summaries.append(
                f"rata-rata minat terhadap tabungan/deposito {interest.mean():.2f} dari 5"
            )
            context["minat_tabungan"] = f"{interest.mean():.2f}"

    if not summaries:
        return "Data pendukung rekomendasi tidak tersedia.", context
    return "; ".join(summaries) + ".", context


def generate_strategy(z_row: pd.Series, context: Mapping[str, str]) -> str:
    balance = float(z_row["Saldo_Rata_Rata"])
    frequency = float(z_row["Frekuensi_Transaksi"])
    nominal = float(z_row["Nominal_Transaksi_Bulanan"])
    feature = float(z_row["Keragaman_Fitur"])
    satisfaction = float(z_row["Kepuasan_Layanan"])
    security = float(z_row["Kepercayaan_Keamanan"])
    promo_response = float(z_row["Respons_Promosi"])

    financial = float(np.mean([balance, nominal]))
    activity = float(np.mean([frequency, feature]))
    recommendations: List[str] = []

    # Strategi inti selalu berasal dari tujuh variabel perilaku.
    if frequency >= LEVEL_THRESHOLD and feature >= LEVEL_THRESHOLD:
        recommendations.append(
            "pertahankan keterlibatan melalui program loyalitas, gamifikasi transaksi, dan akses fitur lanjutan"
        )
    elif frequency >= LEVEL_THRESHOLD:
        recommendations.append(
            "pertahankan frekuensi transaksi sekaligus memperluas penggunaan fitur yang relevan"
        )
    elif feature >= LEVEL_THRESHOLD:
        recommendations.append(
            "ubah pemanfaatan fitur yang luas menjadi transaksi yang lebih rutin melalui use-case harian dan program loyalitas"
        )
    elif activity <= -LEVEL_THRESHOLD:
        recommendations.append(
            "tingkatkan aktivitas melalui onboarding ulang, tutorial singkat, pengingat transaksi, dan kampanye reaktivasi"
        )
    else:
        recommendations.append(
            "dorong peningkatan penggunaan bertahap melalui rekomendasi fitur sesuai kebutuhan rutin"
        )

    if financial >= LEVEL_THRESHOLD:
        recommendations.append(
            "lakukan cross-selling produk simpanan, deposito, investasi, atau layanan bernilai tambah"
        )
    elif financial <= -LEVEL_THRESHOLD:
        recommendations.append(
            "utamakan produk berbiaya rendah, fleksibel, dan mudah digunakan sebelum menawarkan produk bernilai tinggi"
        )

    if promo_response >= LEVEL_THRESHOLD:
        recommendations.append(
            "gunakan promosi terukur seperti cashback, bebas biaya, atau reward sesuai pola transaksi"
        )
    elif promo_response <= -LEVEL_THRESHOLD:
        recommendations.append(
            "jangan mengandalkan diskon sebagai pesan utama; tonjolkan kemudahan, stabilitas, dan manfaat fungsional layanan"
        )

    if satisfaction <= -LEVEL_THRESHOLD:
        recommendations.append(
            "prioritaskan perbaikan kecepatan, kestabilan aplikasi, dan dukungan layanan sebelum promosi agresif"
        )
    elif satisfaction >= LEVEL_THRESHOLD and security >= LEVEL_THRESHOLD:
        recommendations.append(
            "manfaatkan kepuasan dan kepercayaan yang tinggi untuk program referral serta advokasi pengguna"
        )

    if security <= -LEVEL_THRESHOLD:
        recommendations.append(
            "perkuat edukasi keamanan, komunikasi antifraud, dan transparansi perlindungan data"
        )

    # Data pendukung hanya menyesuaikan bentuk dan kanal, bukan menentukan nama segmen.
    promo = context.get("promo")
    media = context.get("media")
    kendala = context.get("kendala")
    aktivitas_utama = context.get("aktivitas")
    if promo and promo_response > -LEVEL_THRESHOLD:
        recommendations.append(f"sesuaikan bentuk penawaran dengan preferensi utama: {promo}")
    if media:
        recommendations.append(f"sampaikan kampanye terutama melalui {media}")
    if kendala:
        recommendations.append(f"sertakan solusi terhadap kendala utama: {kendala}")
    if aktivitas_utama:
        recommendations.append(
            f"hubungkan penawaran dengan aktivitas utama pengguna: {aktivitas_utama}"
        )

    # Maksimal lima butir agar narasi tetap fokus.
    unique: List[str] = []
    seen = set()
    for item in recommendations:
        key = normalize_text(item)
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return "; ".join(unique[:5]).capitalize() + "."


def build_cluster_profile(
    result_df: pd.DataFrame,
    centroid_original: pd.DataFrame,
    centroid_z: pd.DataFrame,
) -> pd.DataFrame:
    labels = assign_segment_labels(centroid_z)
    rows: List[dict] = []

    z_indexed = centroid_z.set_index("Cluster")
    original_indexed = centroid_original.set_index("Cluster")

    for cluster_id in sorted(result_df["Cluster"].unique()):
        cluster_id = int(cluster_id)
        cluster_data = result_df.loc[result_df["Cluster"] == cluster_id].copy()
        z_row = z_indexed.loc[cluster_id]
        original_row = original_indexed.loc[cluster_id]

        support_summary, context = build_support_summary(cluster_data)
        strategy = generate_strategy(z_row, context)

        row = {
            "Cluster": cluster_id,
            "Nama_Segmen": labels[cluster_id],
            "Jumlah_Anggota": len(cluster_data),
            "Persentase_Anggota": len(cluster_data) / len(result_df) * 100.0,
            "Deskripsi_Perilaku": build_behavior_description(z_row),
            "Profil_Pendukung": build_profile_summary(cluster_data),
            "Data_Pendukung_Rekomendasi": support_summary,
            "Rekomendasi_Strategi": strategy,
        }
        for column in FEATURE_COLUMNS:
            row[column] = float(original_row[column])
            row[f"Level_{column}"] = relative_level(float(z_row[column]))
            row[f"Z_{column}"] = float(z_row[column])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)


def run_clustering(
    clean_df: pd.DataFrame,
    selected_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, StandardScaler, KMeans]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(clean_df[FEATURE_COLUMNS])

    model = KMeans(
        n_clusters=selected_k,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=42,
    )
    labels = model.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)

    result_df = clean_df.copy()
    result_df["Cluster"] = labels.astype(int)

    centroid_original_values = scaler.inverse_transform(model.cluster_centers_)
    centroid_original = pd.DataFrame(
        centroid_original_values,
        columns=FEATURE_COLUMNS,
    )
    centroid_original.insert(0, "Cluster", np.arange(selected_k, dtype=int))

    centroid_z = pd.DataFrame(model.cluster_centers_, columns=FEATURE_COLUMNS)
    centroid_z.insert(0, "Cluster", np.arange(selected_k, dtype=int))

    return result_df, centroid_original, centroid_z, score, scaler, model


# =========================================================
# FUNGSI FORMAT DAN EKSPOR
# =========================================================
def format_rupiah(value: float) -> str:
    if pd.isna(value):
        return "-"
    return "Rp " + f"{float(value):,.0f}".replace(",", ".")


def format_decimal(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def make_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Customer_ID": "R001",
                "Bersedia_Menjadi_Responden": "Ya",
                "Pernah_Menggunakan_Mobile_Banking": "Ya",
                "Transaksi_3_Bulan_Terakhir": "Ya",
                "Jenis_Layanan_Digital": "Mobile banking",
                "Aplikasi_Digital_Banking": "BCA Mobile",
                "Usia": 24,
                "Jenis_Kelamin": "Perempuan",
                "Jenis_Pekerjaan": "Karyawan swasta",
                "Pendapatan_Bulanan": "Rp5.000.001 – Rp10.000.000",
                "Saldo_Rata_Rata": "Rp3.000.000 – Rp5.000.000",
                "Frekuensi_Transaksi": "10-20 kali",
                "Nominal_Transaksi_Bulanan": "Rp5.000.001 – Rp9.000.000",
                "Fitur_Digunakan": "Cek saldo/mutasi rekening, Transfer antarbank, Pembayaran QRIS, Top up e-wallet",
                "Respons_Promosi": 4,
                "Kepuasan_Layanan": 4,
                "Kepercayaan_Keamanan": 4,
                "Tujuan_Penggunaan": "Transfer Uang, Pembayaran QRIS",
                "Tujuan_Keuangan": "Menabung lebih rutin",
                "Minat_Tabungan_Deposito": 4,
                "Jenis_Promosi_Disukai": "Cashback QRIS, Bebas biaya transfer/admin",
                "Media_Promosi_Diperhatikan": "Notifikasi aplikasi mobile banking",
                "Kendala_Penggunaan": "Jaringan internet tidak stabil",
                "Aktivitas_Pembayaran_Digital": "Pembayaran QRIS di merchant/restoran",
            }
        ]
    )


def dataframe_to_excel(
    result_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    centroid_original: pd.DataFrame,
    centroid_z: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    cleaning_report: CleaningReport,
) -> bytes:
    report_df = pd.DataFrame(
        {
            "Pemeriksaan": [
                "Data awal",
                "Baris kosong dihapus",
                "Duplikat dihapus",
                "Tidak lolos screening",
                "Usia di bawah 17 tahun",
                "Fitur utama tidak lengkap",
                "Outlier ekstrem terdeteksi (tidak otomatis dihapus)",
                "Data valid",
                "Kolom konstan",
            ],
            "Hasil": [
                cleaning_report.baris_awal,
                cleaning_report.baris_kosong_dihapus,
                cleaning_report.duplikat_dihapus,
                cleaning_report.tidak_lolos_screening,
                cleaning_report.usia_tidak_memenuhi,
                cleaning_report.fitur_utama_tidak_lengkap,
                cleaning_report.outlier_ekstrem_terdeteksi,
                cleaning_report.baris_valid,
                ", ".join(cleaning_report.kolom_konstan) or "Tidak ada",
            ],
        }
    )

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result_df.to_excel(writer, sheet_name="Hasil_Clustering", index=False)
        profile_df.to_excel(writer, sheet_name="Profil_Cluster", index=False)
        centroid_original.to_excel(writer, sheet_name="Centroid_Asli", index=False)
        centroid_z.to_excel(writer, sheet_name="Centroid_Relatif", index=False)
        evaluation_df.to_excel(writer, sheet_name="Evaluasi_K", index=False)
        report_df.to_excel(writer, sheet_name="Ringkasan_Cleaning", index=False)
    return output.getvalue()


def make_summary_text(
    profile_df: pd.DataFrame,
    selected_k: int,
    silhouette: float,
    recommended_k: int,
) -> str:
    lines = [
        "RINGKASAN HASIL SEGMENTASI PENGGUNA MOBILE BANKING",
        "====================================================",
        f"Jumlah cluster yang digunakan (K): {selected_k}",
        f"Rekomendasi K berdasarkan nilai Silhouette tertinggi: {recommended_k}",
        f"Nilai Silhouette Coefficient: {silhouette:.3f}",
        f"Interpretasi kualitas: {quality_label(silhouette)}",
        "",
        "Catatan metodologis:",
        "- K-Means hanya menggunakan tujuh variabel utama penelitian.",
        "- Label segmen ditentukan dari posisi centroid terhadap rata-rata seluruh responden.",
        "- Level tinggi/sedang/rendah bersifat relatif terhadap sampel, bukan penilaian absolut.",
        "- Usia, pekerjaan, pendapatan, preferensi promosi, media, dan kendala hanya digunakan sebagai data pendukung.",
        "",
    ]

    for _, row in profile_df.iterrows():
        lines.extend(
            [
                f"Cluster {int(row['Cluster'])} - {row['Nama_Segmen']}",
                f"Jumlah anggota: {int(row['Jumlah_Anggota'])} ({row['Persentase_Anggota']:.1f}%)",
                f"Deskripsi perilaku: {row['Deskripsi_Perilaku']}",
                f"Profil pendukung: {row['Profil_Pendukung']}",
                f"Data pendukung rekomendasi: {row['Data_Pendukung_Rekomendasi']}",
                f"Rekomendasi strategi: {row['Rekomendasi_Strategi']}",
                "",
            ]
        )
    return "\n".join(lines)


# =========================================================
# LOGO DAN TAMPILAN
# =========================================================
def find_logo_path() -> Optional[Path]:
    """Mencari logo pada root repository atau folder assets."""
    candidates = [
        Path(__file__).resolve().parent / "logo.png",
        Path(__file__).resolve().parent / "assets" / "logo.png",
        Path("logo.png"),
        Path("assets/logo.png"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def logo_data_uri() -> str:
    """Mengubah logo lokal menjadi data URI agar pasti tampil di header HTML."""
    logo_path = find_logo_path()
    if logo_path is None:
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# =========================================================
# TAMPILAN STREAMLIT
# =========================================================
def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Menampilkan logo kecil di navigasi/sidebar Streamlit bila fitur tersedia.
    logo_path = find_logo_path()
    if logo_path is not None:
        try:
            st.logo(str(logo_path))
        except Exception:
            pass

    st.markdown(
        f"""
        <style>
        .stApp {{background: #f7faf8;}}
        .block-container {{padding-top: 1.25rem; padding-bottom: 2rem;}}
 .hero {{
    background: white;
    color: #0D6EFD;
    padding: 20px 28px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.10);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 22px;
    min-height: 112px;
}}

.hero h1 {{
    margin: 0;
    font-size: 32px;
    line-height: 1.22;
    font-weight: 750;
    color: #000080;
}}
        .hero-logo-box {{
            background: rgba(255,255,255,.97);
            border-radius: 14px;
            padding: 9px 14px;
            min-width: 108px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 12px rgba(0,0,0,.08);
        }}
        .hero-logo {{
            display: block;
            max-height: 78px;
            max-width: 125px;
            width: auto;
            object-fit: contain;
        }}
        .cluster-card {{
            background: white;
            border: 1px solid #e3ece7;
            border-radius: 18px;
            padding: 22px 26px;
            margin: 12px 0 18px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,.035);
        }}
        .cluster-card h3 {{margin-top: 0; color: #293042;}}
        .note {{
            background: #eef8f1;
            border-left: 5px solid {GREEN};
            padding: 13px 15px;
            border-radius: 9px;
            margin: 8px 0 14px 0;
        }}
        .warn {{
            background: #fff7e6;
            border-left: 5px solid #f59e0b;
            padding: 13px 15px;
            border-radius: 9px;
            margin: 8px 0 14px 0;
        }}
        div[data-testid="stMetricValue"] {{font-size: 25px;}}
        @media (max-width: 760px) {{
            .hero {{padding: 20px; min-height: 0;}}
            .hero h1 {{font-size: 25px;}}
            .hero-logo-box {{min-width: 78px; padding: 7px 10px;}}
            .hero-logo {{max-height: 58px; max-width: 90px;}}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _levels_table(profile_row: pd.Series) -> pd.DataFrame:
    rows = []
    for column in FEATURE_COLUMNS:
        value = float(profile_row[column])
        if column in {"Saldo_Rata_Rata", "Nominal_Transaksi_Bulanan"}:
            formatted = format_rupiah(value)
        else:
            formatted = format_decimal(value, 1)
        rows.append(
            {
                "Variabel utama": DISPLAY_NAMES[column],
                "Rata-rata cluster": formatted,
                "Posisi relatif": profile_row[f"Level_{column}"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    configure_page()

    logo_uri = logo_data_uri()
    logo_html = (
        f'<div class="hero-logo-box"><img class="hero-logo" src="{logo_uri}" alt="Logo"></div>'
        if logo_uri
        else ""
    )
    st.markdown(
        f"""
        <div class="hero">
          <h1>Dashboard Segmentasi Pengguna Mobile Banking</h1>
          {logo_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_df = make_template()
    with st.sidebar:
        st.header("Input Penelitian")
        uploaded_file = st.file_uploader(
            "Unggah data hasil kuesioner",
            type=["csv", "xlsx"],
            help="Gunakan file CSV atau XLSX hasil ekspor Google Forms.",
        )
        st.download_button(
            "Unduh template kolom",
            template_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="template_data_segmentasi.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("---")
        st.caption(
            "Tujuh variabel utama menjadi input K-Means. Data demografi dan preferensi hanya digunakan untuk memperkaya interpretasi dan strategi."
        )

    if uploaded_file is None:
        st.info("Silakan unggah file CSV/XLSX hasil kuesioner melalui panel di sebelah kiri.")
        return

    try:
        raw_df = load_uploaded_file(uploaded_file)
        clean_df, cleaning_report, rename_map = build_clean_dataset(raw_df)
    except Exception as exc:
        st.error(f"Data belum dapat diproses: {exc}")
        return

    constant_features = cleaning_report.kolom_konstan
    if constant_features:
        st.error(
            "Variabel utama berikut tidak memiliki variasi dan tidak layak untuk K-Means: "
            + ", ".join(DISPLAY_NAMES.get(c, c) for c in constant_features)
        )
        return

    scaler_for_eval = StandardScaler()
    x_scaled_for_eval = scaler_for_eval.fit_transform(clean_df[FEATURE_COLUMNS])
    max_k = min(8, len(clean_df) - 1)
    if max_k < 2:
        st.error("Jumlah data valid terlalu sedikit untuk membentuk cluster.")
        return

    evaluation_df = evaluate_k(x_scaled_for_eval, max_k=max_k)

    # Sesuai arahan dosen pembimbing, hasil akhir difokuskan pada K = 4 atau K = 5.
    allowed_k = [
        int(k) for k in evaluation_df["K"].tolist()
        if int(k) in (4, 5)
    ]
    if not allowed_k:
        st.error("Data valid belum cukup untuk membentuk K = 4 atau K = 5.")
        return

    valid_eval_45 = evaluation_df[
        evaluation_df["K"].isin(allowed_k)
    ].dropna(subset=["Silhouette"])
    if valid_eval_45.empty:
        recommended_k = allowed_k[0]
    else:
        recommended_k = int(
            valid_eval_45.loc[valid_eval_45["Silhouette"].idxmax(), "K"]
        )

    # K dipilih otomatis berdasarkan nilai Silhouette tertinggi antara K = 4 dan K = 5.
    # Tidak ada slider/indikator K maupun tulisan rekomendasi di bawah upload file.
    selected_k = recommended_k

    result_df, centroid_original, centroid_z, sil_score, _, _ = run_clustering(
        clean_df,
        selected_k,
    )
    profile_df = build_cluster_profile(result_df, centroid_original, centroid_z)
    segment_map = profile_df.set_index("Cluster")["Nama_Segmen"].to_dict()
    result_df["Nama_Segmen"] = result_df["Cluster"].map(segment_map)

    tabs = st.tabs(
        [
            "1. Validasi & Preprocessing",
            "2. Penentuan K",
            "3. Hasil Clustering",
            "4. Interpretasi & Strategi",
            "5. Unduh Hasil",
        ]
    )

    # -----------------------------------------------------
    # TAB 1
    # -----------------------------------------------------
    with tabs[0]:
        st.subheader("Validasi dan Preprocessing Data")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data awal", cleaning_report.baris_awal)
        c2.metric("Data valid", cleaning_report.baris_valid)
        c3.metric("Duplikat dihapus", cleaning_report.duplikat_dihapus)
        c4.metric("Tidak lolos screening", cleaning_report.tidak_lolos_screening)

        report_df = pd.DataFrame(
            {
                "Pemeriksaan": [
                    "Baris kosong",
                    "Usia di bawah 17 tahun",
                    "Fitur utama tidak lengkap/tidak logis",
                    "Outlier ekstrem terdeteksi",
                    "Kolom utama konstan",
                ],
                "Hasil": [
                    cleaning_report.baris_kosong_dihapus,
                    cleaning_report.usia_tidak_memenuhi,
                    cleaning_report.fitur_utama_tidak_lengkap,
                    cleaning_report.outlier_ekstrem_terdeteksi,
                    ", ".join(cleaning_report.kolom_konstan) or "Tidak ada",
                ],
                "Tindakan": [
                    "Dihapus",
                    "Dihapus",
                    "Dihapus",
                    "Ditampilkan sebagai informasi; tidak otomatis dihapus",
                    "Analisis dihentikan bila ada",
                ],
            }
        )
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        if cleaning_report.outlier_ekstrem_terdeteksi:
            st.markdown(
                f"""
                <div class="warn">
                Terdapat <b>{cleaning_report.outlier_ekstrem_terdeteksi}</b> responden yang terdeteksi sebagai outlier ekstrem berdasarkan aturan 3×IQR. Data tidak dihapus otomatis karena nilai ekstrem masih dapat merupakan kondisi responden yang valid.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("Lihat pemetaan nama kolom"):
            mapping_df = pd.DataFrame(
                [
                    {"Kolom asli": original, "Dibaca sebagai": canonical}
                    for original, canonical in rename_map.items()
                ]
            )
            st.dataframe(mapping_df, use_container_width=True, hide_index=True)

        st.markdown("**Tujuh variabel yang masuk ke K-Means**")
        st.dataframe(
            clean_df[FEATURE_COLUMNS].describe().T.reset_index().rename(
                columns={"index": "Variabel"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        preview_columns = [
            column
            for column in PROFILE_COLUMNS + FEATURE_COLUMNS
            if column in clean_df.columns
        ]
        st.markdown("**Preview data setelah preprocessing**")
        st.dataframe(
            clean_df[preview_columns].head(30),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # TAB 2
    # -----------------------------------------------------
    with tabs[1]:
        st.subheader("Penentuan Jumlah Cluster")
        c1, c2, c3 = st.columns(3)
        c1.metric("K digunakan", selected_k)
        c2.metric("K rekomendasi Silhouette", recommended_k)
        c3.metric("Silhouette K terpilih", f"{sil_score:.3f}")

        fig_elbow = px.line(
            evaluation_df,
            x="K",
            y="Inertia",
            markers=True,
            title="Elbow Method: Perubahan Inertia pada Setiap K",
        )
        fig_elbow.update_layout(xaxis_dtick=1)
        st.plotly_chart(fig_elbow, use_container_width=True)

        fig_sil = px.line(
            evaluation_df,
            x="K",
            y="Silhouette",
            markers=True,
            title="Perbandingan Silhouette Coefficient",
        )
        fig_sil.update_layout(xaxis_dtick=1)
        st.plotly_chart(fig_sil, use_container_width=True)

        st.dataframe(
            evaluation_df.style.format({"Inertia": "{:.3f}", "Silhouette": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.info(
            "Pemilihan K tidak hanya mengikuti satu angka. Titik siku pada Elbow digunakan bersama nilai Silhouette dan kemudahan interpretasi setiap cluster."
        )

    # -----------------------------------------------------
    # TAB 3
    # -----------------------------------------------------
    with tabs[2]:
        st.subheader("Hasil K-Means Clustering")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah responden", len(result_df))
        c2.metric("Jumlah cluster", selected_k)
        c3.metric("Kualitas struktur", quality_label(sil_score))

        st.caption(
            "Visualisasi hasil hanya menampilkan jumlah anggota pada setiap cluster. "
            "Nama segmen dijelaskan pada tab Interpretasi & Strategi."
        )

        # Hasil clustering divisualisasikan hanya melalui jumlah anggota per cluster.
        # Tidak memakai sumbu X/Y, tidak memilih dua variabel, dan tidak menampilkan nama segmen.
        distribution = (
            result_df.groupby("Cluster")
            .size()
            .reset_index(name="Jumlah_Anggota")
            .sort_values("Cluster")
        )
        distribution["Label_Cluster"] = distribution["Cluster"].map(
            lambda cluster_id: f"Cluster {int(cluster_id)}"
        )

        fig_count = px.bar(
            distribution,
            x="Label_Cluster",
            y="Jumlah_Anggota",
            text="Jumlah_Anggota",
            color="Label_Cluster",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Distribusi Anggota Setiap Cluster",
            labels={
                "Label_Cluster": "Cluster",
                "Jumlah_Anggota": "Jumlah Anggota",
            },
        )
        fig_count.update_traces(textposition="inside")
        fig_count.update_layout(
            showlegend=False,
            xaxis={"categoryorder": "array", "categoryarray": distribution["Label_Cluster"].tolist()},
        )
        st.plotly_chart(fig_count, use_container_width=True)

        st.markdown("**Centroid dalam skala asli**")
        st.dataframe(
            centroid_original.style.format(
                {
                    "Saldo_Rata_Rata": "{:,.0f}",
                    "Frekuensi_Transaksi": "{:.2f}",
                    "Nominal_Transaksi_Bulanan": "{:,.0f}",
                    "Keragaman_Fitur": "{:.2f}",
                    "Respons_Promosi": "{:.2f}",
                    "Kepuasan_Layanan": "{:.2f}",
                    "Kepercayaan_Keamanan": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------
    # TAB 4
    # -----------------------------------------------------
    with tabs[3]:
        st.subheader("Interpretasi Cluster dan Rekomendasi Strategi")
        st.markdown(
            """
            <div class="note">
            Nama segmen dan narasi perilaku disusun melalui <i>cluster profiling</i> berdasarkan posisi centroid terhadap rata-rata seluruh responden pada tujuh variabel utama. Algoritma K-Means menghasilkan keanggotaan cluster dan centroid, sedangkan nama segmen merupakan label deskriptif yang diberikan peneliti setelah membaca karakteristik centroid. Usia, pekerjaan, pendapatan, preferensi promosi, media, dan kendala tidak menentukan keanggotaan cluster; semuanya hanya memperkaya interpretasi dan rekomendasi.
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Dasar interpretasi dan referensi"):
            st.markdown(
                f"""
                **Dasar interpretasi**

                - K-Means membentuk kelompok berdasarkan kedekatan data terhadap centroid.
                - Profil setiap cluster dibaca dari centroid tujuh variabel utama yang telah distandardisasi.
                - Nilai z-score ≥ **+{LEVEL_THRESHOLD:.2f}** dikategorikan relatif tinggi, nilai ≤ **−{LEVEL_THRESHOLD:.2f}** relatif rendah, dan nilai di antaranya relatif sedang. Batas ini merupakan aturan operasional penelitian agar interpretasi konsisten, bukan ketentuan mutlak algoritma K-Means.
                - Nama segmen dibuat peneliti berdasarkan kombinasi karakteristik centroid yang paling menonjol. Nama segmen bukan keluaran otomatis Python.

                **Referensi metodologis**

                1. J. MacQueen, “Some Methods for Classification and Analysis of Multivariate Observations,” *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, vol. 1, pp. 281–297, 1967.
                2. L. Kaufman and P. J. Rousseeuw, *Finding Groups in Data: An Introduction to Cluster Analysis*. New York: Wiley, 1990, doi: 10.1002/9780470316801.
                3. G. Punj and D. W. Stewart, “Cluster Analysis in Marketing Research: Review and Suggestions for Application,” *Journal of Marketing Research*, vol. 20, no. 2, pp. 134–148, 1983, doi: 10.1177/002224378302000204.
                """
            )

        for _, row in profile_df.iterrows():
            st.markdown(
                f"""
                <div class="cluster-card">
                    <h3>Cluster {int(row['Cluster'])} – {row['Nama_Segmen']}</h3>
                    <p><b>Jumlah anggota:</b> {int(row['Jumlah_Anggota'])} pengguna ({row['Persentase_Anggota']:.1f}%)</p>
                    <p><b>Deskripsi perilaku:</b> {row['Deskripsi_Perilaku']}</p>
                    <p><b>Profil pendukung:</b> {row['Profil_Pendukung']}</p>
                    <p><b>Data pendukung rekomendasi:</b> {row['Data_Pendukung_Rekomendasi']}</p>
                    <p><b>Rekomendasi strategi:</b> {row['Rekomendasi_Strategi']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                _levels_table(row),
                use_container_width=True,
                hide_index=True,
            )

    # -----------------------------------------------------
    # TAB 5
    # -----------------------------------------------------
    with tabs[4]:
        st.subheader("Unduh Hasil Analisis")
        excel_bytes = dataframe_to_excel(
            result_df,
            profile_df,
            centroid_original,
            centroid_z,
            evaluation_df,
            cleaning_report,
        )
        summary_text = make_summary_text(
            profile_df,
            selected_k,
            sil_score,
            recommended_k,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "Unduh hasil lengkap XLSX",
                excel_bytes,
                file_name="hasil_segmentasi_mobile_banking.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Unduh hasil responden CSV",
                result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="hasil_cluster_responden.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                "Unduh ringkasan interpretasi TXT",
                summary_text.encode("utf-8"),
                file_name="ringkasan_interpretasi_segmentasi.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.markdown("**Profil cluster yang akan masuk ke Bab IV**")
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Program penelitian: segmentasi pengguna mobile banking berdasarkan perilaku penggunaan tiga bulan terakhir menggunakan K-Means Clustering."
    )


if __name__ == "__main__":
    main()
