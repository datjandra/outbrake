# outbrake NER – training scripts

Modular Python version of `outbrake.ipynb`. The scripts read `train.txt`
(CoNLL-style `<token> <tag>` lines with BIO tags for `DISEASE`, `LOC`, `DATE`
and `CASE`), train the notebook's GRU encoder/decoder with Bahdanau attention
and save the resulting model to local disk.

As in the notebook, every *word* is treated as a tiny "sentence" that the
seq2seq model translates into its tag, e.g. `Ebola` → `b disease`.

## Layout

```
train.py                  # train:      python train.py [options]
infer_rss.py              # inference:  python infer_rss.py --artifacts artifacts
requirements.txt
outbrake_ner/
    config.py             # TrainingConfig dataclass (notebook defaults)
    preprocessing.py      # unicode_to_ascii / preprocess_sentence
    data.py               # read train.txt, Tokenizer, padding, train/val split, tf.data pipeline
    model.py              # Encoder, BahdanauAttention, Decoder
    training.py           # Seq2SeqTrainer: teacher-forced train step + early stopping loop
    persistence.py        # save_artifacts / save_checkpoint / load_artifacts
    inference.py          # WordTagger (greedy decoding) + BIO entity grouping
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python train.py                                  # notebook defaults, writes ./artifacts
python train.py --data-path train.txt --output-dir artifacts --epochs 32 --patience 16
python train.py --validation-split 0 --checkpoint-every 5   # train on all data, checkpoint every 5 epochs
python train.py --help                           # all options
```

Defaults (identical to the notebook): batch size 8, embedding dim 1024,
1024 GRU units, dropout 0.2, Adam (lr 1e-3), up to 32 epochs, early stopping
after 16 epochs without improvement of the total training loss, 80/20 split
(the 20 % held-out part is not used for training, as in the notebook).

## Saved artifacts

`<output-dir>/` contains everything needed to rebuild the model:

| file                  | content                                                        |
|-----------------------|----------------------------------------------------------------|
| `encoder.weights.h5`  | encoder weights                                                |
| `decoder.weights.h5`  | decoder (+ attention) weights                                  |
| `tokenizers.json`     | word → index maps for the input words and the target tags      |
| `config.json`         | hyper-parameters plus `max_length_inp`, `max_length_targ`, vocab sizes |
| `checkpoints/`        | per-epoch weights (only when `--checkpoint-every N` is used)   |

Reload with:

```python
from outbrake_ner.persistence import load_artifacts
config, encoder, decoder, inp_tokenizer, targ_tokenizer = load_artifacts("artifacts")
```

## Inference on the WHO RSS feed

```bash
python infer_rss.py --artifacts artifacts              # all entries of the WHO DON feed
python infer_rss.py --limit 5 --quiet                  # first 5 entries, no summary text
python infer_rss.py --feed-url path/to/feed.xml        # a locally saved feed
```

For every entry the script prints the title, link and summary followed by the
extracted entities, e.g. `Disease: Ebola virus`, `Location: Democratic Republic
of the Congo`, `Date: 17 February 2020`, `Case: no new cases`.

## Notes

* `tf.keras.preprocessing.text.Tokenizer` used by the notebook no longer exists
  in Keras 3 (TensorFlow ≥ 2.16); `outbrake_ner.data.Tokenizer` is a
  small, JSON-serialisable drop-in with the same behaviour.
* Inference requires internet access to fetch the RSS feed (or pass a local
  file with `--feed-url`). Tagging is done one word at a time, exactly as in
  the notebook; results are cached per word so repeated tokens are decoded once.
