# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import json
from pathlib import Path


def convert_metadata_to_jsonl(
    dataset_dir: str,
    metadata_file: str = "metadata.csv",
    wavs_folder: str = "wavs",
    output_jsonl: str = "train_raw.jsonl",
    ref_audio: str = None,
):
    """
    Convert metadata.csv format (filename.wav|transcription) to JSONL format.
    
    Args:
        dataset_dir: Root directory containing metadata.csv and wavs folder
        metadata_file: Name of metadata file (default: metadata.csv)
        wavs_folder: Name of folder containing audio files (default: wavs)
        output_jsonl: Output JSONL file path
        ref_audio: Path to reference audio file. If None, uses first audio file in dataset
    """
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / metadata_file
    wavs_path = dataset_path / wavs_folder
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    if not wavs_path.exists():
        raise FileNotFoundError(f"WAVs folder not found: {wavs_path}")
    
    output_path = Path(output_jsonl)
    
    jsonl_data = []
    audio_files = []
    
    print(f"Reading metadata from: {metadata_path}")
    print(f"Looking for audio files in: {wavs_path}")
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row_num, row in enumerate(reader, 1):
            if len(row) < 2:
                print(f"Warning: Skipping row {row_num} - insufficient columns: {row}")
                continue
            
            filename = row[0].strip()
            text = row[1].strip()
            
            if not filename or not text:
                print(f"Warning: Skipping row {row_num} - empty filename or text")
                continue
            
            audio_path = wavs_path / filename
            
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            jsonl_data.append({
                "audio": str(audio_path.absolute()),
                "text": text,
            })
            audio_files.append(str(audio_path.absolute()))
    
    if not jsonl_data:
        raise ValueError("No valid data found in metadata file")
    
    if ref_audio is None:
        ref_audio = audio_files[0]
        print(f"Using first audio file as reference: {ref_audio}")
    else:
        ref_audio_path = Path(ref_audio)
        if not ref_audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
        ref_audio = str(ref_audio_path.absolute())
    
    for item in jsonl_data:
        item["ref_audio"] = ref_audio
    
    print(f"Writing {len(jsonl_data)} entries to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Successfully converted {len(jsonl_data)} entries to JSONL format")
    print(f"Output file: {output_path.absolute()}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert metadata.csv format to JSONL format for Qwen3-TTS fine-tuning"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Root directory containing metadata.csv and wavs folder"
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.csv",
        help="Name of metadata file (default: metadata.csv)"
    )
    parser.add_argument(
        "--wavs_folder",
        type=str,
        default="wavs",
        help="Name of folder containing audio files (default: wavs)"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="train_raw.jsonl",
        help="Output JSONL file path (default: train_raw.jsonl)"
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Path to reference audio file. If not provided, uses first audio file in dataset"
    )
    
    args = parser.parse_args()
    
    convert_metadata_to_jsonl(
        dataset_dir=args.dataset_dir,
        metadata_file=args.metadata_file,
        wavs_folder=args.wavs_folder,
        output_jsonl=args.output_jsonl,
        ref_audio=args.ref_audio,
    )


if __name__ == "__main__":
    main()
