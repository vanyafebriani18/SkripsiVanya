Program ini merupakan revisi dari dashboard Streamlit untuk penelitian skripsi Vanya Febriani.

1. Program sudah dapat membaca nama kolom panjang dari export Google Form, misalnya:
   - `Saldo Rata-Rata\nDalam 3 bulan terakhir...`
   - `Frekuensi Transaksi\nDalam 3 bulan terakhir...`
   - `Nominal Transaksi Bulanan\nDalam 3 bulan terakhir...`
   - `Keragaman Fitur yang Digunakan\nFitur digital banking/mobile banking...`
   - `Respons terhadap Promosi...`
   - `Kepuasan Layanan...`
   - `Kepercayaan terhadap Keamanan Layanan...`

2. Input utama K-Means tetap hanya 7 variabel utama sesuai Bab 3:
   - `Saldo_Rata_Rata`
   - `Frekuensi_Transaksi`
   - `Nominal_Transaksi_Bulanan`
   - `Keragaman_Fitur`
   - `Respons_Promosi`
   - `Kepuasan_Layanan`
   - `Kepercayaan_Keamanan`

3. Data profil responden tidak masuk ke K-Means, tetapi digunakan untuk interpretasi:
   - `Usia`
   - `Jenis_Kelamin`
   - `Jenis_Pekerjaan`
   - `Pendapatan_Bulanan`
   - `Jenis_Layanan_Digital`
   - `Aplikasi_Digital_Banking`

4. Data pendukung Section 5 tidak masuk ke K-Means, tetapi digunakan untuk rekomendasi strategi pemasaran:
   - `Tujuan_Penggunaan`
   - `Tujuan_Keuangan`
   - `Minat_Tabungan_Deposito`
   - `Jenis_Promosi_Disukai`
   - `Preferensi_Promo_Tabungan_Deposito`
   - `Faktor_Minat_Tabungan_Deposito`
   - `Media_Promosi_Diperhatikan`
   - `Kendala_Penggunaan`
   - `Faktor_Peningkatan_Minat`
   - `Aktivitas_Pembayaran_Digital`

5. Program menambahkan menu 
   - Uji Validitas menggunakan korelasi item-total.
   - Uji Reliabilitas menggunakan Cronbach's Alpha.


```bash
pip install -r requirements.txt
streamlit run app.py
```

Jika command `streamlit` tidak terbaca di Windows:

```bash
python -m streamlit run app.py
```
