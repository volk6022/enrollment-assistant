# test_data — STT Benchmark Audio Samples

## Expected Format

Each test sample consists of two paired files with the same base name:

| File | Description |
|------|-------------|
| `sample_01.wav` | Audio file: **16 kHz, mono, PCM 16-bit WAV** |
| `sample_01.txt` | Ground-truth transcript (see format below) |

## Transcript Format

- **Lowercase only** — no capital letters
- **No punctuation** — strip all commas, periods, question marks, etc.
- **Plain space-separated words** — no markdown, no line breaks within one transcript
- **UTF-8 encoding**

### Example

```
test_data/
  sample_01.wav        →  audio of someone saying "Привет, как дела?"
  sample_01.txt        →  привет как дела

  sample_02.wav        →  audio of "Я хочу поступить на юридический факультет."
  sample_02.txt        →  я хочу поступить на юридический факультет

  sample_03.wav        →  audio of a longer utterance
  sample_03.txt        →  меня зовут иван и я хочу узнать о правилах поступления в институт
```

## Audio Requirements

| Parameter | Required value |
|-----------|----------------|
| Sample rate | **16 000 Hz** |
| Channels | **1 (mono)** |
| Bit depth | **16-bit PCM** |
| Format | `.wav` |

If you have audio in a different format, convert it with `ffmpeg`:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Recommended Dataset

For Russian speech benchmarking, consider using samples from:

- **OpenSTT** (Russian open-source speech corpus): https://github.com/snakers4/open_stt
- **SOVA Dataset** (Russian conversational speech): https://github.com/sova-tm/dataset_speech
- **CommonVoice Russian**: https://commonvoice.mozilla.org/ru/datasets
- Record custom samples specific to the admissions office domain (names, dates, faculty names, etc.)

## WER Scoring Notes

The benchmark normalises both reference and hypothesis before WER calculation:
1. Convert to lowercase
2. Strip all punctuation
3. Collapse extra whitespace

This means punctuation differences do not affect the WER score — only word-level errors count.
