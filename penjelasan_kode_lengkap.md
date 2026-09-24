# Penjelasan Kode Lengkap

> Dokumen ini menjelaskan **keseluruhan** codebase proyek deteksi *clickbait* Bahasa
> Indonesia berbasis IndoBERT, dari data mentah sampai pelaporan hasil evaluasi.
> Ditulis untuk membantu Anda memahami dan mempertahankan (defend) metodologi
> eksperimen secara menyeluruh. Istilah teknis standar (DataFrame, dictionary,
> pipeline, intersection, dll.) sengaja dibiarkan dalam bahasa Inggris.
>
> Catatan penting: dokumen ini memisahkan **logika metodologi** (bagaimana
> eksperimen dirancang) dari **temuan data** (angka hasil). Semua yang dijelaskan
> di sini adalah metodologi/arsitektur kode — bukan klaim hasil.

---

## Daftar Isi
1. [Ringkasan Alur Eksperimen](#1-ringkasan-alur-eksperimen)
2. [Rincian Modul dan Skrip](#2-rincian-modul-dan-skrip)
3. [Justifikasi Metodologi](#3-justifikasi-metodologi)

---

## 1. Ringkasan Alur Eksperimen

Proyek ini menjawab empat pertanyaan penelitian tentang **robustness** (ketahanan)
model IndoBERT untuk deteksi *clickbait* Bahasa Indonesia:

- **RQ1 — Domain shift:** Seberapa turun performa model bila diterapkan ke domain
  yang berbeda dari domain latihnya? (matriks *cross-domain* 5×5)
- **RQ2 — Ketahanan terhadap perturbation:** Seberapa turun performa ketika teks
  diberi gangguan (perturbation) dengan intensitas meningkat?
- **RQ3 — Efek interaksi:** Apakah efek *domain shift* dan *perturbation* bersifat
  aditif atau saling memperkuat ketika digabung?
- **RQ4 — Deployment:** Apakah temuan dapat dieksplorasi secara interaktif
  (dashboard Streamlit)?

### Alur data end-to-end

```
CSV mentah per domain  (id, source, date, title, Label, url)
        │
        ▼
[data_manager] load + validasi + rename kolom (title→text, Label→label) + tambah kolom domain
        │
        ▼
Stratified split per domain (70/15/15)  →  disimpan ke output/data_splits/
        │
        ├───────────────► PELATIHAN  (train_pipeline.py)
        │        Untuk tiap varian model × tiap domain:
        │          ModelTrainer.train(train_df, val_df)
        │            - ClickbaitDataset melakukan tokenisasi (IndoBERT tokenizer)
        │            - IndoBERTClassifier + AdamW + linear warmup scheduler
        │            - early stopping berdasarkan macro-F1 validasi
        │          → checkpoint: checkpoints/<model>/<Domain>/best_model.pt
        │
        └───────────────► EVALUASI  (evaluate_pipeline.py)
                 - Memuat KEMBALI test split yang sama dari output/data_splits/
                 - Memuat checkpoint spesialis per domain
                 EvaluationEngine.run_complete_evaluation (5 fase):
                   1. In-domain
                   2. Cross-domain 5×5
                   2.1 Domain shift (Source Drop / Target Drop)
                   3. Membuat perturbation SEKALI (semantic + typo)
                   4. Evaluasi in-domain × perturbation (dengan filter common_ids)
                   5. Evaluasi cross-domain × perturbation
                 → error analysis + statistical analysis
                 → hasil disimpan ke JSON
        │
        ▼
KONSUMEN HASIL: print_evaluation_summary.py, dashboard_app.py, sel-sel notebook
```

### Dua mekanisme perturbation (konsep kunci)

Sistem menguji **dua jenis gangguan yang independen**, masing-masing pada tiga
tingkat intensitas:

| Jenis (`ptype`) | Deskripsi | Level intensitas |
|---|---|---|
| `semantic` | Substitusi kata dengan sinonim (menjaga makna) | low=10% / medium=20% / high=30% (fraksi **kata**) |
| `typo` | Kesalahan ketik level karakter (keyboard) | low=10% / medium=30% / high=50% (fraksi **karakter**) |

Keduanya dihitung dan dilaporkan secara **terpisah** — ID sampel `semantic` tidak
pernah dicampur dengan ID sampel `typo`.

---

## 2. Rincian Modul dan Skrip

### 2.1 `src/config.py` — Pusat Konfigurasi

**Peran:** Satu sumber kebenaran (*single source of truth*) untuk semua
hyperparameter, path, threshold, dan skema data.

**Isi utama:** Sepuluh `@dataclass` — `ModelConfig`, `TrainingConfig`,
`DataConfig`, `PerturbationConfig`, `EvaluationConfig`, `StatisticalConfig`,
`PathConfig`, `DashboardConfig`, `LoggingConfig`, `SystemConfig` — yang
digabungkan oleh kelas `Config`. Ada satu instance global `config = Config()`
dan fungsi `get_config()`.

Beberapa nilai penting:
- 3 varian model: `INDOBERT_BASE/LARGE/LITE`.
- 5 domain: Technology, Politics, Health, Sport, Education, dengan `DOMAIN_FILES`
  yang memetakan domain ke nama file CSV.
- Split 70/15/15, skema kolom CSV wajib `[id, source, date, title, Label, url]`.
- Parameter statistik: ROPE=0.01, alpha=0.05, 10000 sampel bootstrap.
- `validate()` memastikan split berjumlah 1.0 dan urutan intensitas benar;
  `setup_environment()` menyetel seed dan membuat folder.

**Koneksi:** Diimpor oleh hampir semua modul lewat `from config import config`.

---

### 2.2 `src/data_manager.py` — Pemuatan, Validasi, dan Split Data

**Peran:** Mengubah CSV mentah menjadi struktur data terstandardisasi, lalu
membaginya menjadi train/val/test per domain.

**Kelas & fungsi inti:**
- `DataManager.from_dataset_directory(...)` — *factory* yang dipakai kedua pipeline
  untuk membangun `DataManager` dari sebuah folder dataset.
- `load_datasets(...)` — membaca CSV tiap domain, memvalidasi skema lewat
  `_validate_dataframe`, lalu **mengganti nama kolom** `title→text` dan
  `Label→label` serta menambah kolom `domain`. Inilah **kontrak skema** untuk
  seluruh sistem: setelah tahap ini, semua DataFrame memakai kolom
  `text`, `label`, `domain` (dan `id` tetap dipertahankan sebagai identitas
  sampel).
- `stratified_split_by_domain(...)` — melakukan split **secara independen untuk
  tiap domain**, agar model spesialis dan pengujian *cross-domain* yang jujur
  dimungkinkan.
- `export_domain_splits(...)` / `load_domain_splits(...)` — menyimpan dan memuat
  kembali split ke/dari disk (`<output>/data_splits/<domain>/{train,val,test}.csv`).
  Ini yang menjamin pelatihan dan evaluasi memakai partisi test yang **sama persis**.
- `DatasetValidator` — pengecekan keseimbangan kelas, kualitas teks, dan distribusi
  domain (statis, tanpa mengubah data).

**Koneksi:** Menjadi sumber `test_data_by_domain` untuk `evaluation_engine`, dan
sumber `domain_splits` untuk `train_pipeline`.

---

### 2.3 `src/model_trainer.py` — Stack Pelatihan PyTorch

**Peran:** Semua yang berhubungan dengan pelatihan dan inferensi model.

**Kelas & fungsi inti:**
- `ClickbaitDataset` (subclass `torch.utils.data.Dataset`) — melakukan tokenisasi
  judul berita untuk diberikan ke model.
- `IndoBERTClassifier` — pembungkus model HuggingFace dengan *classification head*.
- `ModelTrainer` — inti loop pelatihan: `initialize_model`, `prepare_data_loaders`,
  `configure_optimizer` (AdamW + linear warmup), `train_epoch`, `evaluate`,
  `train` (dengan *early stopping* pada macro-F1), `save_checkpoint`/
  `load_checkpoint`, dan `predict(...)` yang mengembalikan `(y_pred, y_proba)` —
  ini adalah pintu inferensi yang dipakai semua evaluator.
- `HyperparameterSearch` — pencarian grid untuk hyperparameter.
- `_copy_to_drive(...)` — menyalin file checkpoint dengan mekanisme *retry* (untuk
  Google Colab/Drive).

**Koneksi:** `evaluation_engine` memanggil `ModelTrainer.predict(...)`;
`train_pipeline` memanggil `ModelTrainer.train(...)`.

---

### 2.4 `src/perturbation_engine.py` — Mesin Perturbation (dua mekanisme)

**Peran:** Menghasilkan versi teks yang telah diberi gangguan, untuk menguji
ketahanan model.

**Komponen inti:**
- `PerturbationEngine` — API publik. Metode kunci:
  - `apply_perturbation(text, level)` dan `apply_perturbation_with_metadata(...)`
    yang **memilih mekanisme berdasarkan nama level**:
    level `low/medium/high` → mekanisme *semantic*; level
    `typo_low/typo_medium/typo_high` → mekanisme *typo*.
  - `apply_to_dataframe(df, level=...)` — memberi perturbation ke seluruh baris
    dan menambahkan kolom-kolom metadata audit.
- `SemanticWordSubstitution` — substitusi kata berbasis sinonim. Menggunakan:
  - `IndonesianThesaurus` (dari `dataset/data/dict.json`) untuk mencari kandidat
    sinonim (kaskade: langsung → sinonim induk → hasil *stemming* Sastrawi),
  - `_SimilarityChecker` (IndoBERT kontekstual) untuk menilai kemiripan
    *kata asli dalam konteks asli* vs *kata kandidat dalam konteks baru*, dengan
    batas cosine similarity [0.80, 0.95].
- `TypoCharacterPerturbation` — kesalahan ketik level karakter (substitusi/hapus/
  sisip/tukar berbasis tetangga keyboard QWERTY). Loop kandidat dibatasi
  `max_attempts`; menerima kandidat hanya bila **rasio edit karakter aktual**
  masuk pita target (target ± toleransi). Rasio edit dihitung dengan pembagi
  **jumlah karakter alfabet** (konsisten dengan cara target dihitung).
  Cosine similarity TF-IDF karakter (`_TypoSimilarityChecker`) **direkam sebagai
  diagnostik**, bukan sebagai gerbang penerimaan.

**Kolom metadata penting** yang dihasilkan `apply_to_dataframe` (dipakai hilir):
`original_text`, `perturbed_text`, `perturbation_level`, `perturbation_type`,
`is_same_as_original`, `similarity_in_range`, `perturbation_in_range`,
`perturbation_success`, `actual_ratio_all_words`, `similarity_score_mean/min/max`,
dan (khusus typo) `char_edit_ratio`, `char_cosine_similarity`, `total_chars`.

`perturbation_success` didefinisikan sebagai: baris berubah dari aslinya **dan**
memenuhi kriteria "in range" mekanismenya.

**Koneksi:** `evaluation_engine` memanggil `apply_to_dataframe(...)` untuk membuat
data perturbed sekali di awal.

---

### 2.5 `src/evaluation_engine.py` — Orkestrator Evaluasi (inti eksperimen)

**Peran:** Menjalankan seluruh skenario evaluasi dan membentuk struktur hasil.

**Komponen inti:**
- `MetricsCalculator` — menghitung accuracy/precision/recall/macro-F1, confusion
  matrix, metrik robustness (`calculate_robustness_metrics`), domain shift SD/TD
  (`calculate_domain_shift_metrics`), dan *prediction flip*
  (`calculate_prediction_flip`, yang mensyaratkan panjang array clean = perturbed
  = label).
- `InDomainEvaluator`, `CrossDomainEvaluator`, `PerturbationEvaluator` — tiga tipe
  evaluator.
- `EvaluationEngine.run_complete_evaluation(...)` — metode utama yang menjalankan
  5 fase (lihat di bawah).
- `generate_all_perturbations(...)` — membuat kedua mekanisme (semantic + typo)
  untuk tiap domain sekali saja, disimpan ke
  `<output>/<domain>/<ptype>/<level>.csv`.
- `_successful_ids(df)` dan `_mean_achieved_intensity(df)` — helper untuk penyaringan.
- `aggregate_results(...)`, `generate_summary_report(...)`, `save_results(...)`.

**Struktur hasil `results['perturbation'][domain]`:**
```python
{
    'clean': {...},                               # baseline penuh (test set utuh)
    'semantic': {'clean': {...}, 'low': {...}, 'medium': {...}, 'high': {...}},
    'typo':     {'clean': {...}, 'low': {...}, 'medium': {...}, 'high': {...}},
}
```
- `['clean']` global = baseline test set penuh (dipertahankan untuk kompatibilitas).
- `['semantic']['clean']` dan `['typo']['clean']` = **aligned clean** khusus tiap
  mekanisme (lihat filter di bawah).

**Filter `common_ids` (Fase 4 — konsep penting untuk defense):**
Untuk **tiap domain** dan **tiap mekanisme secara independen**:
1. Kumpulkan ID sampel yang **berhasil** diperturbasi di tiap level.
2. Hitung **intersection**:
   `common_ids = sukses_low ∩ sukses_medium ∩ sukses_high`
   (berdasarkan ID sampel yang eksak — bukan sekadar memangkas ke jumlah terkecil).
3. **Aligned clean** = baris asli (belum diperturbasi) yang ID-nya ada di
   `common_ids`.
4. Level Low/Medium/High dinilai **hanya** pada `common_ids`, dan degradasinya
   diukur terhadap aligned clean mekanisme tersebut.

Sebelum menghitung F1, engine mencetak jumlah sampel yang lolos filter untuk
Aligned Clean / Low / Medium / High per mekanisme (transparansi). Metrik yang
tetap dilaporkan: *success rate* per mekanisme & level, intensitas aktual yang
dicapai, dan jumlah akhir `common_ids`.

**Lima fase `run_complete_evaluation`:**
1. In-domain (spesialis pada domainnya sendiri).
2. Cross-domain 5×5 (`evaluate_all_combinations`).
3. Domain shift (Source Drop / Target Drop) dari kombinasi src≠tgt.
4. In-domain × perturbation (dengan filter `common_ids` di atas).
5. Cross-domain × perturbation (`run_cross_domain_with_perturbations`), mengikuti
   struktur nested `clean / semantic / typo` yang sama.

**Konstanta:** `PERTURBATION_TYPES = ['semantic', 'typo']`,
`PERTURBATION_LEVELS = ['low', 'medium', 'high']` mengendalikan semua loop
perturbation sehingga tidak ada daftar level yang di-*hardcode* tersebar.

**Koneksi:** Dipanggil oleh `evaluate_pipeline`; hasilnya dikonsumsi oleh
`statistical_analyzer`, `error_analyzer`, `print_evaluation_summary`, dan dashboard.

---

### 2.6 `src/statistical_analyzer.py` — Analisis Statistik

**Peran:** Membandingkan model secara statistik menggunakan skor F1 per kondisi.

**Kelas inti:**
- `BayesianTester` — uji *Bayesian signed-rank* dan perbandingan hierarkis.
- `ROPEAnalyzer` — analisis *Region of Practical Equivalence* (probabilitas berada
  di dalam ROPE, threshold 0.01).
- `SignificanceTester` — uji frekuentis: *paired t-test*, *Mann-Whitney U*,
  *bootstrap confidence interval*.
- `ComparativeStatistics` — statistik degradasi dan peringkat robustness.
- `StatisticalAnalyzer` — fasad: `analyze_model_comparison(...)` (dipanggil pipeline
  evaluasi), `analyze_robustness`, `analyze_cross_domain_robustness`, dan
  generator laporan.

**Catatan penting:** modul ini beroperasi pada **array skor yang dikirim
pemanggil** (mis. vektor F1 per kondisi), bukan langsung membaca struktur
`results['perturbation']`. Jadi ia terpisah bersih dari perubahan struktur hasil.

**Koneksi:** `evaluate_pipeline.perform_statistical_analysis` membangun satu vektor
F1 per model (in-domain + semantic + typo per domain × level) lalu memanggil
`analyze_model_comparison` untuk tiap pasangan model.

---

### 2.7 `src/error_analyzer.py` — Analisis Kesalahan Linguistik

**Peran:** Menjelaskan **mengapa** performa turun, dengan mengaitkan kesalahan ke
pola linguistik.

**Fungsi & kelas inti:**
- `detect_patterns(text, domain)` — menandai pola seperti kata sensasional, klaim
  numerik, pertanyaan retoris, entitas bernama, dan jargon domain.
- `classify_error_type(y_true, y_pred)` — mengklasifikasikan tipe kesalahan (mis.
  false positive/negative).
- `ErrorAnalyzer.analyze(...)` — menelusuri struktur hasil per bagian
  (`_analyze_cross_domain`, `_analyze_perturbation`, `_analyse_predictions`) dan
  menghasilkan peringkat pemicu kesalahan.
- `attach_texts_to_results(...)` — menyuntikkan teks judul mentah ke dalam hasil
  agar analisis kesalahan dapat memeriksa headline yang salah diprediksi.

**Koneksi:** Dipanggil oleh `evaluate_pipeline` (dan opsional oleh
`print_evaluation_summary`) setelah evaluasi selesai.

---

### 2.8 `src/utils.py` — Lapisan Layanan Lintas-Modul

**Peran:** Kumpulan utilitas umum agar tidak ada logika I/O/format yang
diduplikasi.

**Kelas & fungsi inti:** `FileManager` (I/O JSON/CSV/pickle/text),
`LoggerManager`, `ReproducibilityHelper` (`set_seed`), `Timer` (context manager
untuk mengukur durasi), `DataValidator`, `HashHelper`, `MetricsFormatter`,
`ProgressTracker`, `ConfigValidator`, ditambah helper `get_timestamp`,
`safe_divide`, `flatten_dict`, `batch_iterator`, `retry_on_failure`,
`make_json_serializable` (mengubah tipe numpy/tuple agar bisa disimpan ke JSON).

**Koneksi:** Dipakai luas; `file_manager` dan `reproducibility` diekspos sebagai
singleton.

---

### 2.9 `src/debug_logger.py` — Observabilitas Opsional

**Peran:** Logging debug terstruktur yang bisa dinyalakan tanpa mengubah logika
pipeline. Semua hook berupa *no-op* (tidak melakukan apa-apa) kecuali diaktifkan
lewat `set_debug(True)` atau environment variable `DEBUG_PIPELINE=1` / flag
`--debug`.

**Isi:** Sekumpulan fungsi `dbg_*` yang dipanggil di tiap tahap (pemuatan data,
tokenisasi, batch pelatihan, evaluasi, perturbation, statistik, analisis
kesalahan).

**Koneksi:** Diimpor oleh modul-modul inti; tidak memengaruhi hasil saat nonaktif.

---

### 2.10 `src/train_pipeline.py` — Skrip Orkestrasi Pelatihan (CLI)

**Peran:** Menjalankan pelatihan end-to-end dari command line.

**Alur `TrainingPipeline`:**
- `load_and_prepare_data()` — memuat data lalu **menggunakan kembali** split yang
  sudah ada di `output/data_splits/` bila tersedia; hanya membuat split baru bila
  belum ada, dan menyimpannya sekali. CSV global (`train/val/test.csv`) ditulis
  ulang hanya saat split baru dibuat. Tujuannya: melatih varian model kedua/ketiga
  **tidak** memicu split ulang, sehingga test set tidak berubah.
- `train_single_model(...)` / `_train_with_defaults` / `_train_with_grid_search` —
  melatih satu spesialis per (model, domain).
- `run(...)` — loop model × domain; bila ada domain yang gagal, mengumpulkannya
  dan **melempar `RuntimeError`** di akhir agar evaluasi tidak berjalan di atas
  set model yang tidak lengkap.

**Argumen CLI:** `--model {base|large|lite|all}`, `--grid-search`, `--device`,
`--seed`, `--debug`, `--output-dir`, `--checkpoint-dir`, `--dataset-dir`.

**Koneksi:** Menghasilkan checkpoint + `data_splits/` yang dikonsumsi
`evaluate_pipeline`.

---

### 2.11 `src/evaluate_pipeline.py` — Skrip Orkestrasi Evaluasi (CLI)

**Peran:** Menjalankan evaluasi end-to-end dari command line.

**Alur `EvaluationPipeline`:**
- `load_data()` — memuat **satu** test split bersama dari
  `<train_output_dir>/data_splits/`. Bila folder itu tidak ada, pipeline
  **gagal cepat (fail-fast)** dengan `FileNotFoundError` — tidak membuat split
  baru — agar ketiga notebook model dijamin memakai split yang sama persis.
- `load_models(...)` — memuat checkpoint spesialis per domain untuk tiap varian.
- `evaluate_single_model(...)` — membangun `PerturbationEngine` + `EvaluationEngine`,
  menjalankan `run_complete_evaluation`, menyuntikkan teks
  (`attach_texts_to_results`), menjalankan `ErrorAnalyzer`, dan menyimpan hasil.
- `perform_statistical_analysis(...)` — membangun vektor F1 per model (in-domain +
  semantic + typo, per domain × level) dan membandingkan tiap pasangan model.
- `generate_summary(...)` — menulis ringkasan, termasuk `avg_semantic_f1` dan
  `avg_typo_f1` yang dipisah per mekanisme.

**Argumen CLI:** `--model`, `--checkpoint-dir`, `--dataset-dir`, `--output-dir`,
`--train-output-dir`, `--skip-perturbation`, `--thesaurus-path`, `--device`,
`--seed`, `--debug`.

**Koneksi:** Titik masuk utama yang dipanggil oleh sel eksekusi di notebook.

---

### 2.12 `src/print_evaluation_summary.py` — Pelaporan Konsol + Plot

**Peran:** Membaca JSON hasil dan mencetak ringkasan terformat (serta membuat plot
matplotlib).

**Isi:** Konstanta `PERTURB_TYPES`/`PERTURB_LEVELS`/`PERTURB_SEQUENCE`, helper
`_clean_metric`/`_pert_metric` untuk membaca struktur nested, fungsi cetak
in-domain, matriks cross-domain, domain shift, dan tabel perturbation **per
mekanisme**, serta dua fungsi plot yang menampilkan **dua subplot bersebelahan
(Semantic | Typo)** dengan baseline Clean bersama.

**Koneksi:** Dijalankan lewat CLI `python src/print_evaluation_summary.py
--results-dir ... --model ...`.

---

### 2.13 `src/dashboard_app.py` — Dashboard Streamlit

**Peran:** Antarmuka interaktif untuk mengeksplorasi hasil.

**Isi:** `ModelCache` (memuat model & hasil dengan cache), `SingleTextPredictor`
(prediksi headline langsung + demo perturbation dengan pilihan mekanisme + level),
`DomainShiftMatrix` (heatmap), `RobustnessAnalyzer` (kurva degradasi per mekanisme,
tanpa asumsi indeks tetap seperti `scores[3]`), dan `ReliabilitySummary`
(rekomendasi + tabel ringkasan dengan kolom `Perturbation Type`).

**Koneksi:** Membaca JSON hasil evaluasi; dijalankan dengan
`streamlit run src/dashboard_app.py`.

---

### 2.14 `src/test_suite.py` — Uji Unit/Integrasi (tanpa GPU)

**Peran:** Menguji hampir semua modul tanpa memerlukan GPU atau unduhan model
berat (torch/transformers/streamlit/Sastrawi di-*mock*).

**Isi:** Uji untuk config, utils, data_manager, perturbation (termasuk dispatch
level typo dan kunci metadata bersama), metrics, semua evaluator, engine evaluasi,
uji statistik, analisis kesalahan, integrasi pipeline, dan `TestSyntaxAllFiles`
yang mem-parse setiap file `.py` (penjaga regresi).

**Koneksi:** Dijalankan dengan `python -m pytest src/test_suite.py -v`.

---

### 2.15 Notebook Utama — `run_complete_pipeline.ipynb`

**Peran:** Driver end-to-end (berorientasi Google Colab): setup environment →
persiapan data → **pelatihan** (`!python src/train_pipeline.py`) → **evaluasi**
(`!python src/evaluate_pipeline.py`) → pemeriksaan/visualisasi hasil → dashboard.

**Struktur sel (garis besar):**
- Sel eksekusi (jangan diubah): pemanggilan `train_pipeline.py` dan
  `evaluate_pipeline.py`.
- Sel *sanity check* data: menghitung baris & distribusi label per CSV.
- Sel visualisasi pelatihan: kurva loss/F1 per domain dari `training_history.json`.
- Sel inspeksi hasil: tabel in-domain, matriks cross-domain, tabel perturbation
  **per mekanisme** (semantic & typo), kurva degradasi dua subplot, tabel
  cross-domain perturbation, analisis *prediction flip*, dan validasi intensitas
  aktual (semantic memakai rasio kata, typo memakai `char_edit_ratio`).
- Notebook membaca CSV perturbed dari `<domain>/<technique>/<level>.csv`.

**Catatan:** `2SepTryRun_Base.ipynb` dan `bef_run_complete_pipeline.ipynb` adalah
varian/eksperimen (*scratch*); notebook utama untuk alur lengkap adalah
`run_complete_pipeline.ipynb`.

---

### 2.16 Folder `dataset/`

- `data/*.csv` — lima dataset per domain (skema `id, source, date, title, Label,
  url`; `Label` biner 0/1).
- `data/dict.json` — tesaurus Bahasa Indonesia untuk substitusi sinonim.
- `Dataset-description.md` — dokumentasi dataset.
- `dataset_eda.ipynb` — eksplorasi data (EDA).
- `scrap.py` — skrip pengumpulan data (scraping).

---

## 3. Justifikasi Metodologi

Bagian ini menjelaskan **mengapa** kode dirancang seperti ini — penting untuk
membedakan keputusan metodologi dari temuan data.

### 3.1 Pemisahan tanggung jawab (separation of concerns)
Setiap tahap punya modul sendiri: `data_manager` (data), `model_trainer`
(pelatihan/inferensi), `perturbation_engine` (gangguan), `evaluation_engine`
(skenario evaluasi), `statistical_analyzer` (uji statistik), `error_analyzer`
(interpretasi). Ini membuat tiap keputusan bisa dipertahankan secara terisolasi
dan diuji sendiri-sendiri. Perubahan pada cara membuat perturbation, misalnya,
tidak menyentuh logika pemuatan data.

### 3.2 Split disimpan dan digunakan ulang (reproducibility)
Split dibuat **sekali** lalu disimpan ke `data_splits/`. Pelatihan menggunakan
kembali split bila sudah ada, dan evaluasi **gagal cepat** bila split tidak
ditemukan. Alasannya: karena tiga varian model dijalankan di tiga notebook
terpisah, ketiganya wajib dievaluasi pada test set yang **identik**. Split ulang
yang acak akan membuat perbandingan antar-model menjadi tidak valid.

### 3.3 Model spesialis per domain
Data dibagi secara independen per domain sehingga tiap domain punya train/val/test
sendiri dan modelnya sendiri. Inilah yang memungkinkan pengujian *cross-domain*
yang jujur (RQ1): sebuah spesialis benar-benar belum pernah melihat data domain
lain.

### 3.4 Dua mekanisme perturbation yang independen
`semantic` (sinonim) dan `typo` (karakter) menguji fenomena berbeda. Karena itu:
- Level diberi nama berbeda (`low` vs `typo_low`) agar tidak ada ambiguitas.
- ID `common_ids` dihitung **terpisah** per mekanisme — tidak pernah di-*intersect*
  antar mekanisme.
- Metrik dilaporkan per mekanisme (mis. `avg_semantic_f1`, `avg_typo_f1`).
Ini menjaga agar analisis RQ2 tidak mencampur dua sumber degradasi yang berbeda.

### 3.5 Cosine similarity: gerbang vs diagnostik
Untuk `semantic`, kemiripan kontekstual [0.80, 0.95] adalah **gerbang** (menentukan
kandidat diterima) karena tujuannya menjaga makna. Untuk `typo`, kemiripan TF-IDF
karakter hanya **direkam sebagai diagnostik**, bukan gerbang, karena mengetik
salah memang menurunkan kemiripan permukaan teks — memaksakan ambang yang sama
akan membuat level tinggi mustahil tercapai. Kriteria penerimaan `typo` adalah
**rasio edit karakter** yang masuk pita target.

### 3.6 Filter `common_ids` (mengapa memakai intersection, bukan truncation)
Agar kurva degradasi Clean→Low→Medium→High adil, keempat kondisi harus dinilai
pada **sampel yang sama persis**. Karena tidak semua sampel berhasil diperturbasi
di semua level, sistem mengambil **intersection ID** dari sampel yang sukses di
ketiga level, lalu menyaring semua kondisi (termasuk aligned clean) ke ID tersebut.
Menggunakan intersection ID **eksak** (bukan sekadar memangkas ke jumlah terkecil)
memastikan perbandingan benar-benar antar sampel yang identik — sehingga penurunan
F1 mencerminkan efek perturbation, bukan perbedaan komposisi sampel.

### 3.7 Aligned clean per mekanisme
Karena `common_ids` semantic ≠ `common_ids` typo, tiap mekanisme punya baseline
"clean" sendiri (`['semantic']['clean']`, `['typo']['clean']`) yang berisi versi
asli dari sampel-sampel itu. Baseline `['clean']` global (test set penuh) tetap
dipertahankan agar kode hilir lama tidak rusak. Degradasi tiap mekanisme diukur
terhadap aligned clean-nya sendiri.

### 3.8 Unit statistik yang benar
Perbandingan antar model memakai **skor F1 per kondisi** (per domain × kondisi)
sebagai unit observasi, bukan probabilitas prediksi mentah. Ini adalah unit yang
tepat untuk uji berpasangan/Bayesian, karena mencerminkan performa per kondisi
eksperimen, bukan kalibrasi model.

### 3.9 Transparansi jumlah sampel
Sebelum menghitung F1, engine mencetak jumlah sampel yang lolos filter untuk tiap
kondisi (Aligned Clean/Low/Medium/High) per mekanisme. Karena semua nilainya harus
sama dengan ukuran `common_ids`, cetakan ini sekaligus menjadi bukti bahwa filter
diterapkan konsisten — memudahkan Anda menjelaskan berapa banyak data yang benar-
benar dievaluasi.

### 3.10 Observabilitas tanpa biaya
`debug_logger` bersifat *no-op* saat nonaktif sehingga jejak debug bisa dinyalakan
untuk investigasi tanpa memengaruhi hasil ataupun performa saat produksi.

---

### Penutup
Seluruh desain ini memisahkan dengan tegas antara **logika metodologi**
(bagaimana data dibagi, bagaimana perturbation dibuat, bagaimana sampel disaring,
bagaimana metrik dihitung dan dibandingkan) dan **temuan data** (angka F1, tingkat
degradasi, dsb.). Saat *defense*, Anda dapat menjelaskan setiap keputusan di atas
sebagai pilihan metodologis yang independen dari nilai hasil akhirnya.
