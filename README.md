# LILaC: Late Interacting in Layered Component Graph

**Repository for the ARR May Submission: "LILaC: Late Interacting in Layered Component Graph for Open‑domain Multimodal Multihop Retrieval".**

---

![Idea Overview of LILaC](assets/images/system_overview.png)

---

## 🗺️ Project Overview

LILaC is a **retrieval‑first framework** for open‑domain question answering over multimodal documents. It introduces two key innovations:

1. **Layered Component Graph (LCG)** – documents are represented at *two granularities* (paragraph / table / image ↔ sentence / row / object) and are connected by explicit intra‑ & inter‑document edges.
2. **Late‑Interaction Subgraph Retrieval** – a beam‑search traversal that scores candidate edges on‑the‑fly via *subcomponent‑level late interaction*, guided by LLM‑driven query decomposition.

Together, these ideas deliver **state‑of‑the‑art retrieval on 4 / 5 public benchmarks without any task‑specific fine‑tuning**.

---

## ✨ Key Features

* **Layered component graph** — coarse nodes for speed, fine nodes for precision.
* **Plug‑and‑play embedders** — MM‑Embed, UniME, mmE5, or any CLIP‑like encoder.
* **Zero‑shot query decomposition** with Qwen‑2.5‑72B.
* **Edge‑wise late interaction** — strong signal without pre‑computing edge embeddings.
* **Five reproducible benchmarks** — MP‑DocVQA, SlideVQA, InfoVQA, MultimodalQA, MMCoQA.

---

## 🏗️ Repository Layout

The repo cleanly separates **inputs** (`datasets/`) from **pipeline outputs** (`artifacts/`) and from the code that produces them (`src/`):

```
LILaC/
├── src/                            # core Python package
│   ├── utils/                      # generic helpers (path resolution, IO, constants)
│   ├── models/                     # model wrappers (embedders, MLLMs)
│   ├── lilac/                      # LILaC algorithm itself
│   │   ├── basic_class/            # LCG data structures (Component, Text, Table, Image, Document, Graph)
│   │   ├── lcg_constructor/        # builds the layered component graph
│   │   ├── retriever/              # late-interaction retrieval + generation + query decomposition
│   │   └── prompts/                # generation / decomposition prompts
│   └── experiment/                 # experiment-only helpers (parsers, evaluators, visualization)
├── config/
│   ├── lcg_constructor/a.yaml      # LCG-construction paths and aliases
│   ├── retriever/                  # retriever_metadata.yaml, retriever_config.yaml, generator_config.yaml
│   ├── benchmarks/                 # benchmark_parser.yaml
│   ├── experiments/                # retrieval_accuracy_evaluator.yaml
│   └── models.yaml                 # model checkpoint paths
├── scripts/
│   ├── parse_multimodal_document/  # step0 … step8 (build the LCG and embed components)
│   ├── query_decomposition/        # decompose questions, estimate modality, embed sub-queries
│   ├── experiments/                # retriever_accuracy, end_to_end_accuracy, parameter_sensitivity
│   └── experiment_visualization/   # print result tables
├── datasets/<DS>/                  # ← read-only inputs (QAs, raw parsed_documents/, image_components/)
├── artifacts/<DS>/                 # ← everything the pipeline produces (safe to rm -rf)
├── algorithm_results/              # final retrieval / generation JSONLs
├── models/                         # downloaded checkpoints + download_*.sh helpers
├── conda_environments/             # per-embedder conda env specs
└── assets/                         # static figures, slides, posters
```

**Key invariant**: nothing in `datasets/<DS>/` is mutated by the pipeline. On the first run, the LCG constructor seeds `artifacts/<DS>/parsed_documents/dev/` from the corresponding seed in `datasets/<DS>/`, then mutates the copy. Wipe `artifacts/<DS>/` whenever you want a clean re-run.

---

## 🚀 Run LILaC

This section walks you from a *fresh clone with only the contents of `datasets/<DS>/`* through to retrieval + generation results. Every step has a corresponding shell script under `scripts/`; each script accepts `-b "<benchmark list>"` to subset and `-h` for help.

The default benchmark set is `MP-DocVQA SlideVQA InfoVQA MultimodalQA MMCoQA`. Drop benchmarks you don't need with `-b "MP-DocVQA"` etc.

<details>
<summary><b>0. What you start with</b> — input layout under <code>datasets/&lt;DS&gt;/</code> and Hugging Face download for MultimodalQA / MMCoQA.</summary>

For each benchmark `<DS>`, `datasets/<DS>/` contains:

| File / directory                       | Description                                                                 |
| -------------------------------------- | --------------------------------------------------------------------------- |
| `QAs_dev_labeled.json`                 | Dev‑split questions with gold component labels.                             |
| `QAs_dev.json`                         | Dev‑split questions (unlabeled), for some benchmarks.                       |
| `url_title_map_dev.json`               | (MultimodalQA, MMCoQA) URL → page title map.                                |
| `thumburl_url_map_dev.json`            | (MMCoQA only) Thumbnail → URL map.                                          |
| `parsed_documents/dev/*.json`          | One JSON per document — text / table / image entries, raw `hyperlinks`.     |
| `image_components/dev/*.{jpg,png,…}`   | Raw images referenced by `parsed_documents[...]["image"]`.                  |
| `pdf_pages/dev/`                       | (MultimodalQA, MMCoQA) PDF page renderings.                                 |

* For **MP‑DocVQA / SlideVQA / InfoVQA**, the above ship with this repository.
* For **MultimodalQA** and **MMCoQA**, `parsed_documents/` and `image_components/` are too large for git — download from Hugging Face and unpack into the right location:

```bash
# MultimodalQA
huggingface-cli download JoohyungYun/multimodalqa_doc --local-dir datasets/MultimodalQA --repo-type dataset
# MMCoQA
huggingface-cli download JoohyungYun/mmcoqa_doc      --local-dir datasets/MMCoQA  --repo-type dataset
```

</details>

<details>
<summary><b>1. Setup</b> — conda environments and model checkpoint download.</summary>

### 1‑1. Conda environments

The pipeline uses several conda environments (one per embedder, plus a shared one for the LLM-based steps). Build them once:

```bash
conda create -n mmembed python=3.10 -y && conda activate mmembed && pip install -r conda_environments/mmembed.txt
conda create -n unime   python=3.10 -y && conda activate unime   && pip install -r conda_environments/unime.txt
conda create -n mme5    python=3.10 -y && conda activate mme5    && pip install -r conda_environments/mme5.txt
conda create -n qwen2.5 python=3.10 -y && conda activate qwen2.5 && pip install -r conda_environments/qwen2.5.txt
```

Each shell script's header lists which env it expects. You're responsible for activating the right one before invoking.

### 1‑2. Download model checkpoints

```bash
./models/download_generator.sh    # Qwen2.5-VL-7B + Qwen2.5-72B-Instruct
./models/download_embedders.sh    # MM-Embed, UniME, mmE5
```

Checkpoints land under `models/<model_name>/`. Paths are repo-relative (`${REPO_ROOT}/models/...`), so they're portable across clones.

</details>

<details>
<summary><b>2. Build the Layered Component Graph</b> — seven steps that mutate <code>artifacts/&lt;DS&gt;/parsed_documents/</code> in place, then serialize the result for embedding. <i>(Paper §3.1 — LCG construction)</i></summary>

Each step writes its output under `artifacts/<DS>/` and leaves `datasets/<DS>/` untouched. On the first run the pipeline auto-seeds `artifacts/<DS>/parsed_documents/dev/` from the immutable copy in `datasets/<DS>/`, then mutates the copy across steps 1–6.

| Step | Script | Env | GPU | What it does | Paper |
| ---- | ------ | --- | --- | ------------ | ----- |
| 0 | `step0_summarize_images.sh` | `qwen2.5` | 1 | Captions every raw image with Qwen2.5‑VL‑7B → `artifacts/<DS>/image_summaries/dev/<name>.txt`. Used as textual context for purely-visual components during retrieval and generation. | §3.1, "image summarization" |
| 1 | `step1_dereference.sh` | `baseline` | – | Replaces raw hyperlinks (`<a href=…>` style) with structured `edges` and binds every image entry to its on-disk filename. Self-skips when `needs_dereference: false` in the YAML registry. | §3.1, "inter-document edges" |
| 2 | `step2_add_table_segments.sh` | `baseline` | – | Splits each table into per-row `table_segment` records (header + one data row each) — the *fine* layer of the LCG for tables. | §3.1, "fine-grained tables" |
| 3 | `step3_add_sentences.sh` | `baseline` | 4 | Splits paragraphs into sentences with **wtpsplit (SaT‑3l‑sm)**. Default `--mode all` chains `extract_text → split_sentences → add_to_parsed_documents`; pass `-m <one_mode>` to run just one. | §3.1, "fine-grained text" |
| 5 | `step5_add_subimages.sh` | `qwen2.5` | 1 | Open-vocabulary detection on each image via Qwen2.5‑VL‑7B, then crops sub-images and registers them as the *fine* image layer. Default `--mode all` chains `split → add_to_parsed_documents`. | §3.1, "fine-grained images" |
| 6 | `step6_add_edges.sh` | `baseline` | – | Propagates paragraph-level `edges` down to the corresponding sentences via character-offset matching. | §3.1, "intra-document edges" |
| 7 | `step7_serialize_components.sh` | `baseline` | – | Serializes every component into embedding-ready JSON under `artifacts/<DS>/embeddings/serializations/{text,sentence,table,table_segment+image,table_segment+summary,image,subimage,page}.json`. | §3.1, "serialization for encoding" |

```bash
./scripts/parse_multimodal_document/step0_summarize_images.sh   -b "MyDataset"
./scripts/parse_multimodal_document/step1_dereference.sh        -b "MyDataset"   # self-skips if needs_dereference=false
./scripts/parse_multimodal_document/step2_add_table_segments.sh -b "MyDataset"
./scripts/parse_multimodal_document/step3_add_sentences.sh      -b "MyDataset"
./scripts/parse_multimodal_document/step5_add_subimages.sh      -b "MyDataset"
./scripts/parse_multimodal_document/step6_add_edges.sh          -b "MyDataset"
./scripts/parse_multimodal_document/step7_serialize_components.sh -b "MyDataset"
```

> Step 4 is intentionally absent — propositions were dropped from the published pipeline.

</details>

<details>
<summary><b>3. Embed every component</b> — multi-GPU sharded encoder for MM-Embed / UniME / mmE5. <i>(Paper §3.2 — component encoding)</i></summary>

Each command runs the chosen embedder against the serialized corpora produced by Step 7 and writes one `(.pt, .json)` pair per component under `artifacts/<DS>/embeddings/<EMB>/`. Per‑embedder × per‑dataset × per‑component `(max_length, batch_size)` lives in `config/lcg_constructor/embedding_params.yaml`, so a new dataset usually requires zero changes here.

```bash
./scripts/parse_multimodal_document/step8_embed_mmembed.sh   -b "MyDataset"     # env: mmembed   (paper default)
./scripts/parse_multimodal_document/step8_embed_unime.sh     -b "MyDataset"     # env: unime
./scripts/parse_multimodal_document/step8_embed_mme5.sh      -b "MyDataset"     # env: mme5
```

* Subset components: `step8_embed_mmembed.sh -b "MP-DocVQA" -c "text sentence"`.
* Smaller GPU count: `NUM_GPUS=2 step8_embed_…`.

</details>

<details>
<summary><b>4. Decompose queries and embed sub-queries</b> — Qwen2.5-72B decomposes each question, then the same encoder embeds the sub-queries. <i>(Paper §3.3 — LLM-driven query decomposition)</i></summary>

```bash
# Q1 — Qwen2.5-72B splits each question into 1–5 component-targeting sub-queries.
./scripts/query_decomposition/query_decomposer.sh                           # env: qwen2.5  (4 GPUs)

# Q2 — Qwen2.5-72B tags each sub-query with its target modality (text | table | image).
./scripts/query_decomposition/modality_estimator.sh                         # env: qwen2.5  (4 GPUs)

# Q3 — Embed sub-queries with the same encoder used for the corpus.
./scripts/query_decomposition/queryset_embedder.sh --embedder mmembed       # env: matches embedder
```

Outputs:

* `artifacts/<DS>/query_decomposition/dev/subqueries.json` (Q1)
* `artifacts/<DS>/query_decomposition/dev/subqueries_with_modality.json` (Q2)
* `artifacts/<DS>/embeddings/<EMB>/subqueries_modality_{agnostic,aware}.{pt,json}` (Q3 — two prompts: a generic retrieval instruction and a per-modality instruction)

</details>

<details>
<summary><b>5. Retrieval</b> — late-interaction beam-search traversal over the LCG. <i>(Paper §3.4 — Late-Interaction Subgraph Retrieval)</i></summary>

```bash
# Reproduce Table 1 — R@k per benchmark and embedder.
./scripts/experiments/retriever_accuracy.sh                                 # all embedders × all benchmarks
./scripts/experiments/retriever_accuracy.sh -e "MM-Embed" -b "MP-DocVQA"

# Sensitivity sweep over beam width and number of iterations.
./scripts/experiments/parameter_sensitivity.sh
```

Outputs: `algorithm_results/LILaC/<DS>/retrieval/<run_name>/<run_name>.jsonl` — one row per question with the top‑K retrieved components, late-interaction scores, and traversal trace.

</details>

<details>
<summary><b>6. End-to-end generation</b> — feed top-K components into Qwen2.5-VL-7B and score. <i>(Paper §4.3 — end-to-end QA)</i></summary>

```bash
# Reproduce Table 2 — feed retrieval results into Qwen2.5-VL-7B for answer generation.
./scripts/experiments/end_to_end_accuracy.sh                                # env: qwen2.5
./scripts/experiments/end_to_end_accuracy.sh -e "MM-Embed" -b "MMCoQA"
```

Outputs: `algorithm_results/LILaC/<DS>/generation/<run_name>/<run_name>.jsonl`.

</details>

<details>
<summary><b>7. Visualise results</b> — pretty-print R@k / EM / F1 tables from the JSONL outputs.</summary>

```bash
./scripts/experiment_visualization/retriever_accuracy.sh        # prints R@k table per benchmark
./scripts/experiment_visualization/end_to_end_accuracy.sh       # prints EM / F1 table
```

</details>

---

## 🔍 Benchmarks

| Dataset       | Domain       | Components / Doc | SOTA (R\@3) |
| ------------- | ------------ | ---------------- | ----------- |
| MP‑DocVQA     | Industrial   | 3                | **83.6 %**  |
| SlideVQA      | Slides       | 3                | **92.8 %**  |
| InfoVQA       | Infographics | 3                | 86.9 %      |
| MultimodalQA  | Webpages     | 37               | **69.1 %**  |
| MMCoQA        | Webpages     | 32               | **55.8 %**  |

---

## 🧩 Extending LILaC

### Adding your own dataset

The pipeline is registry-driven — adding a new benchmark is a YAML edit (and, if your QA file uses a novel schema, one extra parser class). **No edits to existing source files are required.**

**1. Drop the inputs under `datasets/<MyDataset>/`:**
```
datasets/MyDataset/
├── QAs_dev_labeled.json              # standard QA schema (see qa_schema below)
├── parsed_documents/dev/*.json       # one JSON per document — keys: title, text, table, image
└── image_components/dev/*.{jpg,png}  # raw images referenced by parsed_documents[…]['image']
```

**2. Register the dataset in YAML** — add one entry to `config/retriever/retriever_metadata.yaml` (and the matching `needs_dereference` to `config/lcg_constructor/a.yaml`):

```yaml
dataset_metadata:
  MyDataset:
    name: MyDataset
    type: vqa                          # vqa | multimodalqa
    filename: QAs_dev_labeled.json
    qa_schema: vqa_evidence            # one of the registered schemas
    document_granularity: page         # page (single image per doc) | webpage
    needs_dereference: false           # true only if your docs have raw hyperlinks
    has_page_summaries: true           # true if step0 image summaries feed the generator

type_to_dataset:
  vqa:
    - MP-DocVQA
    - SlideVQA
    - InfoVQA
    - MyDataset                        # ← add here
```

If you need non-default embed parameters, add a section under `config/lcg_constructor/embedding_params.yaml`:
```yaml
embedders:
  MM-Embed:
    per_dataset:
      MyDataset:
        text:     {max_length: 512, batch_size: 24}
        sentence: {max_length: 128, batch_size: 32}
```

**3. (Optional) Register a new QA schema** — only if your QA JSON doesn't match `vqa_evidence` / `mmqa_component` / `mmqa_page`:

```python
# anywhere under src/experiment/parser/  (e.g. my_parser.py)
from src.experiment.parser.benchmark_parser import (
    register_qa_parser, BenchmarkParser, LabeledQA, Evidence,
)

@register_qa_parser("my_schema")
class MyCustomParser(BenchmarkParser):
    def parse(self):
        for qa in self.qa_pairs:
            # adapt your fields → LabeledQA + Evidence
            self.labeled_benchmark.add_labeled_qa_obj(qid, LabeledQA(...))
```

Then reference it from YAML as `qa_schema: my_schema`.

**4. Validate the setup:**
```bash
./scripts/validate_dataset.sh MyDataset
```
The validator checks file presence, YAML registration, capability flags, QA-schema parsability, and parsed-document structure. Fix anything it flags before running the pipeline.

**5. Run the same commands as for any other benchmark:**
```bash
./scripts/parse_multimodal_document/step0_summarize_images.sh   -b "MyDataset"
# step1 self-skips when needs_dereference=false
./scripts/parse_multimodal_document/step2_add_table_segments.sh -b "MyDataset"
./scripts/parse_multimodal_document/step3_add_sentences.sh      -b "MyDataset"
./scripts/parse_multimodal_document/step5_add_subimages.sh      -b "MyDataset"
./scripts/parse_multimodal_document/step6_add_edges.sh          -b "MyDataset"
./scripts/parse_multimodal_document/step7_serialize_components.sh
./scripts/parse_multimodal_document/step8_embed_mmembed.sh      -b "MyDataset"

./scripts/query_decomposition/query_decomposer.sh
./scripts/query_decomposition/modality_estimator.sh
./scripts/query_decomposition/queryset_embedder.sh --embedder mmembed

./scripts/experiments/retriever_accuracy.sh   -b "MyDataset"
./scripts/experiments/end_to_end_accuracy.sh  -b "MyDataset"
```

### Adding a custom embedder

1. Implement the `Embedder` interface in a new file under `src/models/embedder/` (`load_model`, `delete_model`, `encode_queries`, `encode_corpus`).
2. Add a corpus-side entry point `src/lilac/lcg_constructor/step8_embed_<your_embedder>.py` — a 5-line wrapper around `src.lilac.lcg_constructor.embed_runner.run_embedder(...)`. See the MM-Embed / UniME / mmE5 files for templates.
3. Add a query-side entry point under `src/lilac/retriever/query_decomposer/queryset_embedder_<your_embedder>.py`.
4. Add a section under `config/lcg_constructor/embedding_params.yaml` declaring the (dataset type → components) matrix and per-component (max_length, batch_size).
5. Add a thin shell wrapper under `scripts/parse_multimodal_document/step8_embed_<your_embedder>.sh`.

---

## 🙏 Acknowledgements

This work builds on PyTorch, FAISS, and Hugging Face Transformers, and uses the publicly released checkpoints of **MM‑Embed**, **UniME**, **mmE5**, and **Qwen2.5‑VL / Qwen2.5‑72B‑Instruct**.
