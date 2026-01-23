## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. Please run `pip install qwen-tts` first, then run the command below:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Then follow the steps below to complete the entire fine-tuning workflow. Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases.

### 0) Convert metadata.csv to JSONL format (for datasets with metadata.csv)

If your dataset is in the format with a `metadata.csv` file and `wavs/` folder, you can use the conversion script to generate the required JSONL format.

The `metadata.csv` file should have the format:
```
filename.wav|transcription text
filename2.wav|transcription text 2
```

Example:
```csv
Maxim Gorky - Zindigi ki Shahrah par - part - 4 - میکسم گورکی زندگی کی شاھراہ پر- حصہ_0000.wav|میکسم گورکی کیا بیٹی زندگی کی شہرہ پر باب نمبر پانچ
Maxim Gorky - Zindigi ki Shahrah par - part - 4 - میکسم گورکی زندگی کی شاھراہ پر- حصہ_0001.wav|میں موسم بہار میں آخر بہاگ ہی نکلا
```

Convert your dataset to JSONL format:

```bash
python convert_metadata_to_jsonl.py \
  --dataset_dir /home/proxima/PROXIMA-AI/qwen\ tts\ train \
  --metadata_file metadata.csv \
  --wavs_folder wavs \
  --output_jsonl train_raw.jsonl \
  --ref_audio /path/to/reference/audio.wav
```

Arguments:
- `--dataset_dir`: Root directory containing `metadata.csv` and `wavs/` folder
- `--metadata_file`: Name of metadata file (default: `metadata.csv`)
- `--wavs_folder`: Name of folder containing audio files (default: `wavs`)
- `--output_jsonl`: Output JSONL file path (default: `train_raw.jsonl`)
- `--ref_audio`: Optional path to reference audio file. If not provided, uses the first audio file in the dataset

Note: The reference audio (`ref_audio`) should be a high-quality sample of the target speaker. It is recommended to use the same reference audio for all training samples to ensure speaker consistency.

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 2 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name speaker_test
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

For datasets already in JSONL format:

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```

For datasets with metadata.csv format (e.g., Urdu dataset):

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

DATASET_DIR="/home/proxima/PROXIMA-AI/qwen tts train"
RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="urdu_speaker"

# Step 1: Convert metadata.csv to JSONL format
python convert_metadata_to_jsonl.py \
  --dataset_dir "${DATASET_DIR}" \
  --metadata_file metadata.csv \
  --wavs_folder wavs \
  --output_jsonl ${RAW_JSONL}

# Step 2: Prepare data (extract audio_codes)
python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

# Step 3: Fine-tune
python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```

### Dataset Requirements

- Audio files should be in WAV format
- Recommended audio sample rate: 24kHz (the model will handle resampling if needed)
- Audio files should be mono channel
- Ensure transcriptions match the audio content accurately
- For best results, use consistent audio quality across all samples
- Reference audio should be a clear, high-quality sample of the target speaker (recommended: 3-10 seconds)