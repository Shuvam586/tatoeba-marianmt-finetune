# MarianMT Fine-tuned on Tatoeba (en → fr)

This repository contains a Transformer-based Neural Machine Translation (NMT) pipeline where a pretrained MarianMT model was fine-tuned on the Tatoeba English–French parallel corpus.

## Languages
- English → French

## Base Model
- **MarianMT**: `Helsinki-NLP/opus-mt-en-fr`

## Dataset
- **Tatoeba EN–FR parallel corpus**: `Helsinki-NLP/tatoeba`

## Metrics
| Model | BLEU Score |
|------|------------|
| Base OPUS-MT (MarianMT) | **50.5** |
| Fine-tuned OPUS-MT | **55.44** |

## Repository Structure
- `finetuned-opus-en2fr.ipynb` — training + evaluation pipeline
- `demo.ipynb` — sample translations
- `demo.py` — streamlit demo