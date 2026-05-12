# Wealth Alpaca — Hugging Face download + Bedrock reasoning-trace pipeline

This repo ships two scripts:

1. Pull three datasets from Hugging Face.
2. For **any** dataset in `input` / `output` form, call **Amazon Bedrock** via `deepseek_think_eval.py` and fill in an intermediate `<think>...</think>` trace while keeping **Output** fixed.

Examples below default to **wealth**; **CodeAlpaca** and **ChatDoctor** use the same schema—swap `--wealth-json` and `--output-jsonl` (the flag name is historical and not wealth-only).

## Requirements

```bash
pip install -r requirements.txt
```

**Python 3.10+**. Inference uses **Amazon Bedrock** (`boto3`); downloads use `huggingface_hub`.

Credentials: [Get started with the Amazon Bedrock API](https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started-api.html).

## Environment variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | Optional; higher HF Hub rate limits or private repos |
| `AWS_REGION` | Bedrock region (also `--region`) |
| `BEDROCK_MODEL_ID` | Default model id (overridden by `--model`) |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / … | Same as AWS CLI credential chain |

## Files

| File | Role |
|------|------|
| `run_hf_workflow.py` | Download three datasets + optionally invoke `deepseek_think_eval.py` |
| `deepseek_think_eval.py` | Per-row / JSONL batch trace generation (`--workers` for concurrency) |

### Hugging Face datasets (repo IDs are fixed in the script)

- `HaolinRPI/wealth-alpaca-lora-final-dataset-clean`
- `HaolinRPI/codealpaca-20k-final`
- `HaolinRPI/chatdoctor-healthcaremagic-merged-input`

Default download root: `<this repo>/datasets/`.

Primary filenames after download (aligned with the upload helper):

| Dataset | Folder under `datasets/` | Main file |
|---------|----------------------------|-----------|
| Wealth Alpaca | `wealth-alpaca-lora-final-dataset-clean/` | `final_dataset_clean_merged_input.json` |
| CodeAlpaca 20k | `codealpaca-20k-final/` | `final.json` |
| ChatDoctor | `chatdoctor-healthcaremagic-merged-input/` | `chatdoctor_healthcaremagic_merged_input.json` |

If `final.json` / `final_dataset_clean_merged_input.json` is a **JSON array**, `run_hf_workflow.py` converts it to a temporary NDJSON file before calling the pipeline; **NDJSON** (one object per line) is used as-is.

## Quick start

```bash
# Download all three datasets only
python run_hf_workflow.py

# Download, then run the pipeline on the default wealth file (output: datasets/wealth_trace_output.jsonl)
export AWS_REGION=us-east-1
# Configure AWS credentials (aws configure or env vars); ensure IAM allows Bedrock invoke
python run_hf_workflow.py --run-deepseek --workers 10 --limit 500 --region us-east-1
```

### Run Bedrock on each dataset separately (set outputs to avoid overwrites)

Assume snapshots already exist under `datasets/`. Adjust paths for Windows (`\`) if needed.

```bash
export AWS_REGION=us-east-1

# Wealth Alpaca
python run_hf_workflow.py --skip-download --run-deepseek \
  --wealth-json datasets/wealth-alpaca-lora-final-dataset-clean/final_dataset_clean_merged_input.json \
  --output-jsonl datasets/trace_wealth.jsonl \
  --workers 10 --region us-east-1

# CodeAlpaca 20k (final.json)
python run_hf_workflow.py --skip-download --run-deepseek \
  --wealth-json datasets/codealpaca-20k-final/final.json \
  --output-jsonl datasets/trace_codealpaca.jsonl \
  --workers 10 --region us-east-1

# ChatDoctor
python run_hf_workflow.py --skip-download --run-deepseek \
  --wealth-json datasets/chatdoctor-healthcaremagic-merged-input/chatdoctor_healthcaremagic_merged_input.json \
  --output-jsonl datasets/trace_chatdoctor.jsonl \
  --workers 10 --region us-east-1
```

Call the inference script directly (equivalent):

```bash
python deepseek_think_eval.py \
  --input-jsonl datasets/codealpaca-20k-final/final.json \
  --output-jsonl datasets/trace_codealpaca.jsonl \
  --workers 10 --region us-east-1
```

### More options

```bash
# Wealth only, limit rows (smoke test)
python run_hf_workflow.py --skip-download --run-deepseek \
  --wealth-json datasets/wealth-alpaca-lora-final-dataset-clean/final_dataset_clean_merged_input.json \
  --limit 50 --workers 10 --region us-east-1

# Print command only (no Bedrock)
python run_hf_workflow.py --skip-download --run-deepseek --dry-run \
  --wealth-json path/to/file.jsonl
```

See also: `python run_hf_workflow.py --help`, `python deepseek_think_eval.py --help`.

## Publish to GitHub (this folder is the release bundle)

With [Git](https://git-scm.com/downloads) installed:

```powershell
cd D:\Code\wealth-alpaca-workflow   # change to your path

git init
git add .gitignore README.md requirements.txt deepseek_think_eval.py run_hf_workflow.py
git commit -m "Release: HF datasets workflow + Bedrock trace pipeline"

git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```

With [GitHub CLI](https://cli.github.com/):

```powershell
cd D:\Code\wealth-alpaca-workflow
git init
git add .
git commit -m "Release: HF datasets workflow + Bedrock trace pipeline"
gh repo create <REPO_NAME> --public --source=. --remote=origin --push
```

**Do not commit** local `datasets/`, secrets, or `*_out.jsonl`—they are listed in `.gitignore`.

## License

Follow upstream licenses for data and models; license this repo as you choose for the scripts.
