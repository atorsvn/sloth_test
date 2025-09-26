# agents.md – Gemma3:270M Finetune & GGUF Export Agent

This document describes an LLM-powered automation agent that **finetunes the Gemma 3 (270 M) model with Unsloth and exports the final weights as a GGUF file**.  
It is intended as both *documentation* and a *living contract* for automated fine-tuning pipelines.

---

## 1. Agent Overview 🤖

| Attribute | Value |
| --------- | ----- |
| **Purpose** | Fully automate the process of fine-tuning Gemma 3 (270 M) using Unsloth and exporting the model in GGUF format for efficient local inference (e.g., llama.cpp, koboldcpp, ollama). |
| **Core Technology** | Python 3.10+, [Unsloth](https://github.com/unslothai/unsloth), Hugging Face Transformers, BitsAndBytes for 4/8-bit training |
| **Execution Engine** | Local GPU cluster (CUDA 12+, PyTorch 2.x) or cloud GPU runtime |
| **Scope of Autonomy** | Accepts training datasets, plans a LoRA/QLoRA run, monitors training, converts & saves as GGUF |
| **Constraints** | VRAM must meet model+batch requirements (~8 GB for QLoRA), OS Linux/WSL2 |

---

## 2. Tools & Capabilities 🛠️

The agent has access to the following core tools:

| Tool | Description | Parameters | Example Call |
|------|------------|-----------|--------------|
| `prepare_dataset` | Downloads, tokenizes, and shards a dataset. | `source: str`, `output_dir: str`, `split: str` | `prepare_dataset("huggingface://my_corpus", "data/", "train")` |
| `finetune_unsloth` | Runs Unsloth training with LoRA/QLoRA. | `base_model: str`, `dataset: str`, `epochs: int`, `learning_rate: float`, `lora_r: int`, `quant: str` | `finetune_unsloth("gemma3-270m", "data/train", 3, 2e-5, 16, "q8_0")` |
| `export_gguf` | Converts a trained Unsloth checkpoint to GGUF. | `checkpoint_dir: str`, `quantization: str`, `output_file: str` | `export_gguf("runs/gemma3-270m", "q4_k_m", "models/gemma3-270m-finetuned.gguf")` |
| `run_eval` | Benchmarks the exported GGUF model. | `gguf_file: str`, `eval_set: str` | `run_eval("models/gemma3-270m-finetuned.gguf", "hellaswag")` |

> **Tip:** These tools are abstract instructions. An orchestrator (CrewAI, LangChain, Airflow, etc.) can bind them to shell commands like `unsloth finetune` and `convert-gguf`.

---

## 3. Development Flow ⚙️

Typical end-to-end run:

1. **User Intent:**  
   “Fine-tune Gemma 3 (270 M) on my instruction dataset and export as GGUF q4_k_m.”
2. **Planning:**  
   - Preprocess data via `prepare_dataset`.
   - Configure LoRA hyperparameters.
3. **Training:**  
   - Execute `finetune_unsloth` with chosen quantization and r-rank.
4. **Export:**  
   - Call `export_gguf` to produce the final `*.gguf` model.
5. **Validation:**  
   - Optionally run `run_eval` on held-out tasks.

---

## 4. Usage & Example Prompts 🗣️

| User Prompt | Agent Action | Result |
|------------|-------------|-------|
| “Fine-tune Gemma3:270M on `alpaca_cleaned` for 2 epochs and export q4_k_m.” | `prepare_dataset("alpaca_cleaned", "data/", "train")` → `finetune_unsloth(...)` → `export_gguf(...)` | Creates `models/gemma3-270m-finetuned.gguf`. |
| “Benchmark the new model on MMLU.” | `run_eval("models/gemma3-270m-finetuned.gguf", "mmlu")` | Prints accuracy/F1 metrics. |
| “Adjust to 4 epochs and lower LR to 1e-5.” | Updates hyperparameters and re-runs `finetune_unsloth`. | Produces a new GGUF file with updated training. |

---

## 5. Maintenance & Extension ➕

* **New Base Models** – Change the `base_model` parameter to fine-tune different Gemma or LLaMA family checkpoints.  
* **New Export Formats** – Add a tool like `export_safetensors` for alternate deployment targets.  
* **Dataset Upgrades** – Integrate additional dataset sources or streaming loaders.

---

## 6. Recommended Companion Files 📂

| File | Purpose |
|------|--------|
| `requirements.md` | Python dependencies (`unsloth`, `transformers`, `accelerate`, `bitsandbytes`, `gguf-convert`). |
| `design_doc.md` | High-level pipeline architecture and hardware requirements. |
| `tech_implementation.md` | Detailed hyperparameter search space, quantization methods, and checkpointing strategy. |
| `tests.md` | Smoke tests for dataset integrity and GGUF export correctness. |

---

## ✅ Key Idea

This `agents.md` acts as the **single source of truth**.  
By pairing it with LLM orchestration (e.g., CrewAI or LangChain), the entire lifecycle—from **data prep** to **finetuning** to **GGUF export**—can be executed automatically and reproducibly.
