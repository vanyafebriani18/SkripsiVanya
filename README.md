Dashboard Segmentasi Pengguna Mobile Banking — Revisi Final

Program ini digunakan untuk penelitian:

**Analisis Segmentasi Nasabah Digital Banking Berdasarkan Perilaku Penggunaan 3 Bulan Terakhir Menggunakan K-Means Clustering untuk Strategi Pemasaran.**

1. Seluruh menu, fungsi, tabel, ekspor, dan narasi 
2. Input K-Means menggunakan tujuh variabel utama:
   - Saldo rata-rata
   - Frekuensi transaksi
   - Nominal transaksi bulanan
   - Keragaman fitur
   - Respons terhadap promosi
   - Kepuasan layanan
   - Kepercayaan terhadap keamanan
3. Label cluster ditentukan dari posisi centroid pada data yang telah distandardisasi, sehingga dibandingkan dengan rata-rata seluruh responden.
4. Deskripsi cluster menjelaskan posisi **tinggi, sedang, atau rendah** pada seluruh tujuh variabel utama.
5. Usia, jenis kelamin, pekerjaan, pendapatan, aplikasi, preferensi promosi, media, dan kendala hanya menjadi data pendukung.
6. Narasi pekerjaan menampilkan tiga kategori terbesar beserta persentasenya. Istilah “didominasi” hanya digunakan jika satu kategori melebihi 50% anggota cluster.
7. Rekomendasi strategi disusun dari karakteristik perilaku cluster terlebih dahulu, kemudian disesuaikan dengan preferensi dan kendala anggota.
8. Ekspor XLSX berisi hasil clustering, profil cluster, centroid asli, centroid relatif, evaluasi K, dan ringkasan cleaning.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Jika perintah `streamlit` tidak dikenali:

```bash
python -m streamlit run app.py
```
