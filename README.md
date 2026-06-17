# Indonesian Law RAG & Qwen3 Fine-Tuning

Proyek ini berisi dua notebook utama untuk membangun asisten AI berbahasa Indonesia pada domain hukum ketenagakerjaan:

1. `tuning.ipynb` untuk fine-tuning model `unsloth/Qwen3-4B` menggunakan dataset instruksi bahasa Indonesia.
2. `law_rag.ipynb` untuk membangun sistem Retrieval-Augmented Generation (RAG) berbasis dokumen PDF regulasi ketenagakerjaan.

Model hasil fine-tuning dipublikasikan di Hugging Face:

[rizkyayub/rizbuy-submission-qwen3-indonesian](https://huggingface.co/rizkyayub/rizbuy-submission-qwen3-indonesian)

## Ringkasan

Proyek ini menggabungkan fine-tuning LLM dan RAG agar model dapat menjawab pertanyaan pengguna dalam bahasa Indonesia berdasarkan dokumen hukum yang relevan. Fine-tuning dilakukan dengan Unsloth dan LoRA untuk membuat model lebih sesuai dengan gaya instruksi bahasa Indonesia, sedangkan RAG digunakan untuk mengambil konteks dari dokumen regulasi sebelum model menghasilkan jawaban.

## Fitur

- Fine-tuning `Qwen3-4B` dengan Unsloth dan LoRA.
- Dataset instruksi: `Ichsan2895/alpaca-gpt4-indonesian`.
- Tracking eksperimen menggunakan Weights & Biases.
- Upload model hasil fine-tuning ke Hugging Face Hub.
- RAG berbasis dokumen PDF menggunakan LangChain.
- Loading dokumen hukum dengan `PyMuPDFLoader`.
- Hybrid retrieval menggunakan BM25 dan vector search.
- Parent-child chunking untuk menjaga konteks dokumen.
- Reranking hasil retrieval menggunakan `BAAI/bge-reranker-large`.
- Antarmuka tanya jawab menggunakan Gradio.

## Struktur Proyek

```text
.
├── Knowledge_documents_RAG/
│   ├── PP Nomor 35 Tahun 2021.pdf
│   ├── PP Nomor 5 Tahun 2021.pdf
│   ├── PP Nomor 51 Tahun 2023.pdf
│   └── UU Nomor 6 Tahun 2023.pdf
├── law_rag.ipynb
├── tuning.ipynb
├── pyproject.toml
├── requirements.txt
├── link_huggingface.txt
└── README.md
```

Folder output training seperti `outputs_*`, `lora_model_*`, `wandb/`, cache Unsloth, dan file `.env` sebaiknya tidak di-commit ke GitHub.

## Kebutuhan Sistem

- Python 3.12 atau lebih baru.
- GPU NVIDIA dengan CUDA sangat direkomendasikan.
- Akun Hugging Face untuk download dan upload model.
- Akun Weights & Biases jika ingin menggunakan experiment tracking.

Dependency utama:

- `torch`
- `transformers`
- `unsloth`
- `trl`
- `datasets`
- `bitsandbytes`
- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `pymupdf`
- `chromadb`
- `rank-bm25`
- `gradio`
- `wandb`

## Instalasi

Clone repository:

```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository
```

Buat virtual environment:

```bash
python -m venv .venv
```

Aktifkan virtual environment.

Windows:

```bash
.venv\Scripts\activate
```

Linux atau macOS:

```bash
source .venv/bin/activate
```

Install dependency:

```bash
pip install -r requirements.txt
```

Jika menggunakan `uv`:

```bash
uv sync
```

## Konfigurasi Environment

Buat file `.env` di root project:

```env
HF_TOKEN=hf_your_huggingface_token
WANDB_TOKEN=your_wandb_token
```

`HF_TOKEN` digunakan untuk mengakses Hugging Face Hub. `WANDB_TOKEN` hanya diperlukan jika tracking eksperimen Weights & Biases diaktifkan.

Pastikan file `.env` tidak di-commit ke GitHub.

## Fine-Tuning Model

Notebook `tuning.ipynb` menjalankan fine-tuning model `unsloth/Qwen3-4B` menggunakan dataset:

```text
Ichsan2895/alpaca-gpt4-indonesian
```

Konfigurasi utama:

- Base model: `unsloth/Qwen3-4B`
- Sequence length: `1024`
- LoRA rank: `8`
- LoRA alpha: `16`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Optimizer: `paged_adamw_8bit`
- Scheduler: `cosine`
- Max steps: `800`
- Train/test split: `90% / 10%`

Eksperimen training:

| Run | Learning Rate | Gradient Accumulation | Warmup Steps |
| --- | --- | --- | --- |
| `Run_1_LR_1e-5` | `1e-5` | `4` | `50` |
| `Run_2_LR_2e-5` | `2e-5` | `8` | `70` |

Setelah training selesai, model disimpan secara lokal dalam folder `lora_model_*`. Notebook juga mengunggah model merged 16-bit ke Hugging Face Hub:

```text
rizkyayub/rizbuy-submission-qwen3-indonesian
```

## RAG Asisten Hukum

Notebook `law_rag.ipynb` membangun pipeline RAG untuk menjawab pertanyaan seputar regulasi ketenagakerjaan Indonesia.

Alur utama:

1. Load dokumen PDF dari folder `Knowledge_documents_RAG/`.
2. Pecah dokumen menjadi parent chunk dan child chunk.
3. Bangun retriever berbasis vector search.
4. Bangun BM25 retriever untuk lexical search.
5. Gabungkan retriever menggunakan hybrid retrieval.
6. Rerank hasil retrieval menggunakan cross-encoder.
7. Kirim konteks hasil retrieval ke model fine-tuned.
8. Tampilkan antarmuka tanya jawab menggunakan Gradio.

Konfigurasi chunking:

```text
Parent chunk size: 2500
Parent chunk overlap: 250
Child chunk size: 500
Child chunk overlap: 50
```

Model embedding:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Model reranker:

```text
BAAI/bge-reranker-large
```

Retriever:

```text
BM25 weight: 0.5
Vector retriever weight: 0.5
Reranker top_n: 2
```

Model generatif yang digunakan:

```text
rizkyayub/rizbuy-submission-qwen3-indonesian
```

## Menjalankan Aplikasi RAG

Buka notebook:

```bash
jupyter notebook law_rag.ipynb
```

Jalankan seluruh cell secara berurutan. Pada bagian akhir, Gradio akan dijalankan dengan:

```python
demo.launch(share=True)
```

Setelah dijalankan, Gradio akan memberikan URL publik yang dapat digunakan untuk mencoba asisten tanya jawab.

Contoh pertanyaan:

```text
Apa itu upah minimum?
```

## Hasil

Output akhir proyek ini adalah model LLM bahasa Indonesia yang sudah di-fine-tune dan pipeline RAG yang dapat menjawab pertanyaan hukum ketenagakerjaan berdasarkan dokumen regulasi.

Model tersedia di:

[https://huggingface.co/rizkyayub/rizbuy-submission-qwen3-indonesian](https://huggingface.co/rizkyayub/rizbuy-submission-qwen3-indonesian)

Repository ini tidak menyediakan nasihat hukum. Jawaban model perlu diverifikasi kembali dengan dokumen regulasi resmi atau ahli hukum terkait.
