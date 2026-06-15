# Qwen3-VL vs Metis-8B-RL Evaluation

This folder contains a lightweight reproduction harness for comparing:

- original model: `Qwen/Qwen3-VL-8B-Instruct`
- HDPO/Metis model: `Accio-Lab/Metis-8B-RL`

on the same JSONL evaluation set, using one 8-GPU server.

The main script is:

```bash
examples/eval/run_qwen_vs_metis_eval.sh
```

It runs:

1. Qwen3-VL-8B-Instruct with no tools.
2. Metis-8B-RL with the Metis tool policy.
3. A sample-level correctness comparison.

## Environment

Recommended Python: `>=3.10`.

Install inside your existing vLLM environment:

```bash
cd /path/to/HDPO

pip install -e ./verl
pip install -e ".[vllm,search_tool,python_code_dep]"
pip install openai requests
```

The repository pins `vllm<=0.11.0` in `pyproject.toml`. If your server already
has a working vLLM environment, keep that environment and install the repo
editable packages into it.

If your evaluation examples may trigger `text_search`, prepare one search
backend:

```bash
export SEARCH_PROVIDER=serper
export SERPER_API_KEY="YOUR_SERPER_KEY"
```

or:

```bash
export SEARCH_PROVIDER=brightdata
export BRIGHTDATA_API_TOKEN="YOUR_BRIGHTDATA_TOKEN"
export BRIGHTDATA_ZONE="YOUR_BRIGHTDATA_ZONE"
```

If your data only needs Python/image crop-style tools, search keys are not
needed.

## Data File

Prepare one JSONL file. Each line is a sample:

```json
{"id":"sample_001","question":"What is 2 + 3?","answer":"5"}
{"id":"sample_002","question":"Which curve is highest at x=10?","answer":"blue","images":["images/chart_002.png"]}
```

Required fields:

- `id`: stable sample id.
- `question`: question text. Aliases: `input`, `prompt`, `task`.
- `answer`: ground truth. Aliases: `ground_truth`, `gt`, `gts`, `label`.

Optional fields:

- `images`: image path, list of image paths, URLs, or data URLs.
- `extra_info`: extra fields passed to the Metis tool server.

Relative image paths are resolved from the JSONL folder, unless `DATA_ROOT` is
set.

Validate before launching vLLM:

```bash
python examples/eval/validate_eval_data.py \
  --data /path/to/eval.jsonl
```

If images live somewhere else:

```bash
python examples/eval/validate_eval_data.py \
  --data /path/to/eval.jsonl \
  --data-root /path/to/image_root
```

## Minimal Smoke Test

This only checks the wiring and exact-match judge on three text samples:

```bash
DATA_FILE=examples/eval/sample_eval.jsonl \
LIMIT=3 \
bash examples/eval/run_qwen_vs_metis_eval.sh
```

This will download/serve both models if they are not cached.

## Full 8-GPU Run

```bash
DATA_FILE=/path/to/eval.jsonl \
RUN_DIR=runs/qwen_vs_metis_hrbench8k_100 \
GPU_IDS=0,1,2,3,4,5,6,7 \
TP_SIZE=8 \
LIMIT=100 \
bash examples/eval/run_qwen_vs_metis_eval.sh
```

The script uses all 8 GPUs sequentially:

1. start vLLM for Qwen3-VL;
2. run no-tool inference;
3. stop Qwen3-VL vLLM;
4. start vLLM for Metis;
5. start the Metis tool server;
6. run tool-policy inference;
7. compare outputs.

## Placeholders You May Need to Fill

Required:

- `DATA_FILE=/path/to/eval.jsonl`

Usually optional:

- `RUN_DIR=...`: output folder.
- `LIMIT=100`: first N samples; `0` means all samples.
- `DATA_ROOT=/path/to/image_root`: if images are not relative to the JSONL.

Model paths:

- `BASE_MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct`
- `METIS_MODEL_PATH=Accio-Lab/Metis-8B-RL`

If models are already downloaded:

```bash
BASE_MODEL_PATH=/data/models/Qwen3-VL-8B-Instruct \
METIS_MODEL_PATH=/data/models/Metis-8B-RL \
DATA_FILE=/path/to/eval.jsonl \
bash examples/eval/run_qwen_vs_metis_eval.sh
```

vLLM knobs:

- `GPU_IDS=0,1,2,3,4,5,6,7`
- `TP_SIZE=8`
- `VLLM_PORT=8000`
- `VLLM_MAX_MODEL_LEN=32768`
- `VLLM_EXTRA_ARGS="--gpu-memory-utilization 0.9"`

Tool knobs:

- `TOOL_PORT=30569`
- `TOOL_WORKERS=16`

Judge mode:

- Default: `JUDGE_MODE=exact`
- For multiple-choice or exact-answer datasets, keep `exact`.
- For open-ended answers, use a shared external judge:

```bash
JUDGE_MODE=llm \
JUDGE_BASE_URL=http://JUDGE_HOST:8001/v1 \
JUDGE_MODEL=Qwen3-235B-Judge-or-other-judge \
DATA_FILE=/path/to/eval.jsonl \
bash examples/eval/run_qwen_vs_metis_eval.sh
```

Do not use each evaluated model as its own judge; that makes the comparison
hard to interpret.

## Output

The main output folder is `RUN_DIR`, for example:

```text
runs/qwen_vs_metis_YYYYMMDD_HHMMSS/
  base_no_tool.jsonl
  metis_policy.jsonl
  base_vllm.log
  metis_vllm.log
  tool_server.log
  compare/
    model_compare_summary.md
    model_compare_summary.json
    model_compare_records.csv
    model_compare_records.jsonl
```

`base_no_tool.jsonl` and `metis_policy.jsonl` contain one row per sample:

```json
{
  "id": "...",
  "input": "...",
  "gts": "...",
  "output": "<reason>...</reason><answer>...</answer>",
  "answer": "...",
  "accuracy": 1,
  "tool_calls": 0,
  "tool_interact_info": [],
  "mode": "no_tool",
  "model_alias": "base",
  "judge_mode": "exact"
}
```

`compare/model_compare_records.csv` is the file to inspect first:

```text
key,category,base_correct,metis_correct,base_tool_calls,metis_tool_calls,base_answer,metis_answer
```

Categories:

- `metis_only`: Metis fixes a Qwen3-VL error.
- `base_only`: Metis regresses on a Qwen3-VL-correct sample.
- `both_correct`: both models are correct.
- `both_wrong`: neither model solves it.

`compare/model_compare_summary.md` gives aggregate accuracy, delta accuracy,
Metis ToolRate, and the four category counts.

## Dataset Choice

The HDPO repo does not include ready-to-run benchmark JSONL files. For a quick
reproduction, convert a small slice from one of the paper benchmarks:

- `HRBench8K`: high-resolution visual details, good for testing crop/zoom utility.
- `CharXiv(RQ)`: chart reasoning, good for code/crop assistance.
- `WeMath`: harder multimodal math, good for no-tool vs tool-policy gaps.

Start with 30-100 samples before scaling. The goal is to see the sample-level
pattern first, not to reproduce all 13 benchmark tables in one run.

## Download and Prepare Data

### Option A: MathVista, Easiest Smoke-Test Dataset

`MathVista` is the easiest public VLM benchmark to prepare from Hugging Face.
It is not the sharpest dataset for tool-boundary failures, but it is good for
checking the full pipeline.

Install dataset helpers if needed:

```bash
pip install datasets pillow
```

Convert a small split:

```bash
python examples/eval/prepare_hf_vlm_jsonl.py \
  --dataset AI4Math/MathVista \
  --split testmini \
  --out data_eval/mathvista_testmini_100/eval.jsonl \
  --image-dir data_eval/mathvista_testmini_100/images \
  --limit 100
```

Validate:

```bash
python examples/eval/validate_eval_data.py \
  --data data_eval/mathvista_testmini_100/eval.jsonl \
  --data-root data_eval/mathvista_testmini_100
```

Run:

```bash
DATA_FILE=data_eval/mathvista_testmini_100/eval.jsonl \
DATA_ROOT=data_eval/mathvista_testmini_100 \
RUN_DIR=runs/qwen_vs_metis_mathvista_100 \
LIMIT=100 \
bash examples/eval/run_qwen_vs_metis_eval.sh
```

### Option B: Use a Downloaded Benchmark File

If you download a benchmark from Hugging Face and its field names differ, use
the same converter with explicit field mapping.

Example:

```bash
python examples/eval/prepare_hf_vlm_jsonl.py \
  --dataset DATASET_ID_ON_HF \
  --split SPLIT_NAME \
  --out data_eval/my_benchmark/eval.jsonl \
  --image-dir data_eval/my_benchmark/images \
  --id-field "id,uid,pid" \
  --question-field "question,query,prompt" \
  --answer-field "answer,gt,ground_truth,label" \
  --image-field "image,decoded_image,images" \
  --choices-field "choices,options" \
  --limit 100
```

Fill these placeholders:

- `DATASET_ID_ON_HF`: the Hugging Face dataset id.
- `SPLIT_NAME`: for example `test`, `validation`, `testmini`, or the dataset's available split.
- field mappings if the dataset uses unusual column names.

If the dataset requires custom loading code, add:

```bash
--trust-remote-code
```

### Option C: Manually Build JSONL

For HRBench8K, CharXiv(RQ), or WeMath, the fastest reliable path is often:

1. Download the official benchmark from its project/Hugging Face page.
2. Inspect one row and identify: id, question, answer, image path.
3. Convert to:

```json
{"id":"...","question":"...","answer":"...","images":["relative/or/absolute/path.png"]}
```

Then validate with:

```bash
python examples/eval/validate_eval_data.py \
  --data /path/to/eval.jsonl \
  --data-root /path/to/image_root
```

### Which Dataset Should You Start With?

For debugging the pipeline:

1. `MathVista testmini`, 20-100 samples.

For the actual research question:

1. `HRBench8K`, if you want to test whether visual crop/zoom tools help.
2. `CharXiv(RQ)`, if you want chart-reasoning and code/crop-related behavior.
3. `WeMath`, if you want harder multimodal math where no-tool failures are common.

The important thing is not the benchmark name alone. After the run, inspect
`compare/model_compare_records.csv` and count how many samples are
`metis_only`, `base_only`, `both_correct`, and `both_wrong`.
