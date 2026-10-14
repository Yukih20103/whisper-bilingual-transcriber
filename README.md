# Bilingual Transcription (Chinese & English)

A transcription tool with a user-friendly interface that primarily aims to transcribe bilingual (Chinese/English) church sermons into readable subtitles. It also supports other single-language transcription and bilingual (Chinese & English only) transcription tasks. This tool is Intel XPU and CPU version.


This is **Version 1**, which uses **pretrained Whisper models**.

---

## Motivation

Existing Whisper models detect only one language per video, show low accuracy on Chinese and rare domain(eg. Christian) topics, and does not directly support the control of transcript clip length. This project breaks the whole transcription task into small steps, assigns specific tools, and provides a friendly GUI for non-technical users.

---

## Features

- **Supports single-speaker and 2-speaker / bilingual (Chinese & English) videos/audios.**
- **Speaker diarization** using diarize package to separate speakers reliably.
- **Context-preserving segmentation**: merges same-speaker clips into long segments before transcription.
- **Pretrained Whisper transcription** with auto language detection (no forced language/translation).
- **Domain-specific term correction** corrects Chinese terms based on Levenshtein distance between pinyin pronunciation.
- **Language-aware subtitle splitting**: separates mixed Chinese/English text, then merges adjacent same-language pieces.
- **Readable clause splitting** cuts subtitles results into 3–5 second clips.
- **Exports** `.srt` and timestamped `.txt` files. Optional conversion to Traditional Chinese.
- **Simple GUI** built an friendly interface with Flet.

---

## Get started

### Prerequisites

- Python 3.9 or higher
- `ffmpeg` installed and available in your system PATH
- (Optional but recommended) Intel XPU for faster transcription. CPU works too but takes much more time.

---

### Install Dependencies

Clone the repository and install the required packages:

```bash
git clone https://github.com/Yukih20103/whisper-bilingual-transcriber.git
cd whisper-bilingual-transcriber
pip install -r requirements.txt
```

Note: diarize may require additional system libraries. Please refer to its documentation if installation fails.

---

### Usage
Running the GUI
```bash
flet run main.py
```
The application window will open.

---

### GUI Selections

![GUI Screenshot](images/whisper-transcriber.png)

#### 1. Audio or Video File 
The target audio / video for transcription. Support .wav files for audio and .mp4 files for video.

#### 2. Term File (term.txt)
Optional for transcriptions having Chinese. Support .txt files only. One term per line. Lines starting with # are ignored. The writing format (Simplified / Traditional Chinese) must match the desired output writing format. These terms are used to correct Chinese transcription errors based on pronunciation similarity.

Example:

```bash
text
# Christian terms
耶穌
基督
聖靈
阿們
```
#### 3. Output Directory
The output directory for transcript output. The `.srt` and timestamped `.txt` result file will appear in this directory.

#### 4. Whisper Model Selection
tiny, base, small, medium, large, large-v3
Larger models are more accurate but slower. large-v3 is recommended for best quality.

#### 5. Single-Speaker Mode
True: The whole audio is treated as one speaker/language; no diarization is performed.
False: Diarization splits the audio into two speakers, then transcribe segments one by one.

#### 6. Traditional Chinese Mode
True: The model will output Chinese subtitles in Traditional Chinese format (if any).
False: The model will output Chinese subtitles in Simplified Chinese format (if any).

---

### Output
Two files are generated in the output directory. Both contain the transcribed text, split into readable pieces.

filename.srt – SubRip subtitle file.

filename_with_timestamps.txt – Plain text with timestamps.

---

## Limitations & Future Work
Version 1 uses only pretrained Whisper models. There is no fine-tuning or custom model included. A Chinese fine-tuned model may be supported in the future.

CPU transcription can be slow. Future versions may include optimizations or a pre-built CPU executable.

The tool is currently focused on Chinese/English audios. Other language pairs may be supported in the future.

---

## License
This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** license.

**You are free to use this project for educational, non-profit, and personal purposes.**

See the [LICENSE](LICENSE) file for details.
