#!/bin/bash
# Download + adapt math datasets to layout expected by OPSD prepare_grpo_data.py.
# Source repos (all public, no auth):
#   open-r1/DAPO-Math-17k-Processed   -> data/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet
#   Maxwell-Jia/AIME_2024             -> data/AIME_2024/aime_2024_problems.parquet
#   yentinglin/aime_2025              -> data/AIME_2025/train.jsonl
#   HuggingFaceH4/MATH-500            -> data/MATH-500/test.jsonl
set -e

HF=/home/ubuntu/skypilot-runtime/bin/hf
PY=/home/ubuntu/skypilot-runtime/bin/python
DATA_DIR=/home/ubuntu/OPSD_OnPolicyDistillation/data
mkdir -p "$DATA_DIR/DAPO-Math-17k-dedup" "$DATA_DIR/AIME_2024" "$DATA_DIR/AIME_2025" "$DATA_DIR/MATH-500"

STAGING=/tmp/opsd_data_dl
mkdir -p "$STAGING"

# ---- 1. DAPO-Math-17k (English split) ----
echo "[1/4] DAPO-Math-17k-Processed"
"$HF" download "open-r1/DAPO-Math-17k-Processed" --repo-type dataset \
    --local-dir "$STAGING/DAPO" --include "en/*.parquet" --max-workers 4 2>&1 | tail -2
cp "$STAGING/DAPO/en/train-00000-of-00001.parquet" \
   "$DATA_DIR/DAPO-Math-17k-dedup/distinct-prompts-with-rewards.parquet"

# ---- 2. AIME 2024 ----
echo "[2/4] AIME_2024"
"$HF" download "Maxwell-Jia/AIME_2024" --repo-type dataset \
    --local-dir "$STAGING/AIME24" --max-workers 4 2>&1 | tail -2
"$PY" - <<'EOF'
import pandas as pd, glob, os
files = sorted(glob.glob('/tmp/opsd_data_dl/AIME24/*.parquet') + glob.glob('/tmp/opsd_data_dl/AIME24/**/*.parquet', recursive=True))
print('AIME24 candidates:', files)
df = pd.read_parquet(files[0]) if files else None
if df is None:
    raise SystemExit('no AIME24 parquet')
print('AIME24 cols:', list(df.columns), 'rows:', len(df))
# normalize: prepare_grpo_data expects cols ['problem','solution']
# Maxwell-Jia/AIME_2024 has Problem, Answer, Solution -- normalize.
rename_map = {}
if 'Problem' in df.columns: rename_map['Problem'] = 'problem'
if 'Solution' in df.columns: rename_map['Solution'] = 'solution'
if 'Answer' in df.columns and 'solution' not in df.columns and 'Solution' not in df.columns:
    rename_map['Answer'] = 'solution'
df = df.rename(columns=rename_map)
if 'solution' not in df.columns and 'Answer' in df.columns:
    df['solution'] = df['Answer'].astype(str)
df.to_parquet('/home/ubuntu/OPSD_OnPolicyDistillation/data/AIME_2024/aime_2024_problems.parquet')
print('AIME24 saved:', len(df), 'rows; cols:', list(df.columns))
EOF

# ---- 3. AIME 2025 ----
echo "[3/4] AIME_2025"
"$HF" download "yentinglin/aime_2025" --repo-type dataset \
    --local-dir "$STAGING/AIME25" --max-workers 4 2>&1 | tail -2
"$PY" - <<'EOF'
import pandas as pd, glob, json, os
files = sorted(glob.glob('/tmp/opsd_data_dl/AIME25/**/*.parquet', recursive=True))
print('AIME25 candidates:', files)
if not files:
    raise SystemExit('no AIME25 parquet')
df = pd.read_parquet(files[0])
print('AIME25 cols:', list(df.columns), 'rows:', len(df))
rename_map = {}
for k in df.columns:
    if k.lower() in ('problem','question'): rename_map[k] = 'problem'
    if k.lower() == 'answer': rename_map[k] = 'answer'
df = df.rename(columns=rename_map)
df = df[['problem','answer']]
out = '/home/ubuntu/OPSD_OnPolicyDistillation/data/AIME_2025/train.jsonl'
with open(out, 'w') as f:
    for _, row in df.iterrows():
        f.write(json.dumps({'problem': row['problem'], 'answer': str(row['answer'])}) + '\n')
print('AIME25 saved:', len(df), 'rows ->', out)
EOF

# ---- 4. MATH-500 ----
echo "[4/4] MATH-500"
"$HF" download "HuggingFaceH4/MATH-500" --repo-type dataset \
    --local-dir "$STAGING/MATH500" --max-workers 4 2>&1 | tail -2
"$PY" - <<'EOF'
import pandas as pd, glob, json, os
# MATH-500 ships as JSONL or parquet. Try both.
files = sorted(glob.glob('/tmp/opsd_data_dl/MATH500/**/*.jsonl', recursive=True)
              + glob.glob('/tmp/opsd_data_dl/MATH500/**/*.parquet', recursive=True))
print('MATH-500 candidates:', files)
if not files:
    raise SystemExit('no MATH-500 file')
src = files[0]
if src.endswith('.parquet'):
    df = pd.read_parquet(src)
    rename_map = {}
    for k in df.columns:
        if k.lower() in ('problem','question'): rename_map[k] = 'problem'
        if k.lower() == 'answer': rename_map[k] = 'answer'
    df = df.rename(columns=rename_map)
    rows = df.to_dict(orient='records')
else:
    rows = []
    with open(src) as f:
        for line in f:
            rows.append(json.loads(line))
out = '/home/ubuntu/OPSD_OnPolicyDistillation/data/MATH-500/test.jsonl'
with open(out, 'w') as f:
    for r in rows:
        # ensure 'answer' key exists
        if 'answer' not in r and 'Answer' in r: r['answer'] = r.pop('Answer')
        f.write(json.dumps({'problem': r.get('problem') or r.get('Problem'), 'answer': str(r.get('answer'))}) + '\n')
print('MATH-500 saved:', len(rows), 'rows ->', out)
EOF

echo "=== ALL DATA READY ==="
ls -la /home/ubuntu/OPSD_OnPolicyDistillation/data/*/  | head -40
