# Suggested improvements to the model training code

Suggestions are ordered by expected impact. The numbers quoted below were
measured on `train.txt` and the notebook's default hyper-parameters:

| measurement                                         | value                    |
|-----------------------------------------------------|--------------------------|
| tokens / sentences / unique words                   | 5,018 / 136 / 937        |
| words that appear with more than one distinct tag   | 91, covering 39% of tokens |
| share of `O` tags                                   | 76.5%                    |
| model parameters (1024 emb / 1024 units)            | 18.8M (7.2M enc + 11.6M dec) |
| training examples (80% split)                       | 4,014                    |

Examples of context-dependent words:

```
'the'     : {'O': 226, 'I-LOC': 35, 'I-CASE': 1}
'of'      : {'O': 155, 'I-LOC': 42, 'I-CASE': 1}
'cases'   : {'I-CASE': 43, 'O': 38, 'I-DISEASE': 1}
'new'     : {'I-CASE': 19, 'O': 16, 'I-DISEASE': 1}
'disease' : {'I-DISEASE': 37, 'O': 6}
```

---

## 1. Fix the modelling approach (biggest win by far)

**The model tags each word in isolation, with no sentence context.** 39% of
the training tokens are words that carry *different* tags depending on
context (`the` is `O` 226x but `I-LOC` 35x in "Republic of **the** Congo";
`cases` is a coin-flip between `I-CASE` and `O`). No amount of training can
resolve these - the model can only memorise the majority tag per word. It also
discards the sentence boundaries that `train.txt` already provides (136
sentences).

Recommended replacement, in order of effort:

- **Sentence-level BiGRU/BiLSTM tagger** (+ optional CRF layer): input is the
  whole sentence, output is one tag per token. Same TensorFlow stack, ~50
  lines of model code, and it reuses the data module almost unchanged (split
  on blank lines instead of flattening).
- **Fine-tune a pretrained transformer** (`bert-base-cased`, BioBERT or
  PubMedBERT via HuggingFace `transformers` +
  `AutoModelForTokenClassification`). With only 136 sentences this will
  dramatically outperform anything trained from scratch, and it handles
  unseen words in the live RSS feed gracefully - which is the real use case.

**Stop decoding the tag as a sequence.** Generating `b` then `disease` with an
attention decoder is a 9-way classification problem dressed up as
translation. It can emit invalid tags (`b` alone, `o loc`), and Bahdanau
attention over a 3-token input does nothing useful. A single softmax over the
9 labels is simpler, faster, and cannot produce malformed output.

## 2. Right-size the model

18.8M parameters (1024-dim embeddings + 1024 GRU units) for 4,014 training
examples and an 876-word vocabulary is ~4,700 parameters per example - the
model will memorise the training set. `embedding_dim=64-128` and
`units=64-128` would train ~50x faster and generalise better. Add a
hyper-parameter sweep over these once a validation metric exists (see 3).

## 3. Use the validation set, and stop on the right signal

- The 20% split is held out but never looked at; early stopping uses
  **training** loss, which (almost) always decreases, so with `patience=16`
  and `epochs=32` it never fires. Compute validation loss and **entity-level
  F1** (via `seqeval`) each epoch, stop on F1, and **restore the best
  weights** rather than saving the last epoch.
- **Split by sentence, not by token.** The current random split puts the same
  word in both halves (937 unique words over 5,018 tokens), so any validation
  score is inflated by leakage.

## 4. Input representation

- **Keep capitalisation.** `preprocess_sentence` lower-cases everything,
  throwing away the single strongest surface cue for entities in English
  (`Congo`, `Ebola`, `February`). At minimum add an "is-capitalised" feature
  or use a cased vocabulary.
- **Add a real `<unk>` token.** Unknown words currently map to id 0, which is
  the *padding* id the model was trained to ignore - so every unseen word in a
  live feed is processed as "nothing". Reserve a separate `<unk>` index and
  randomly replace rare training words with it (word dropout) so the model
  learns how to handle it.
- **Character-level or subword features** (char-CNN or a pretrained
  tokenizer) cover spelling variants (`Ebola`/`EVD`), numbers and dates
  (`17`, `2020`) without needing each form in the vocabulary.
- Consider normalising digits (`2020 -> 0000`) so dates and case counts
  generalise.

## 5. Training-loop hygiene

- **Class imbalance**: 76.5% of tokens are `O`. Use class weights or monitor
  F1, otherwise "predict `O`" looks like 77% accuracy.
- **Gradient clipping** (`clipnorm=1.0`) - standard for GRUs; cheap insurance.
- **LR schedule**: `ReduceLROnPlateau` or cosine decay; Adam at a fixed 1e-3
  for 32 epochs is rarely optimal.
- **Larger batches** (32-64) with the padding mask - batch 8 on a CPU spends
  most of its time on Python overhead.
- Only average the loss over non-pad positions (the current
  `loss / targ.shape[1]` divides by the padded length, so the loss scale
  depends on padding).

## 6. Engineering improvements

- **Move to `model.fit` with a custom `train_step`** (Keras 3 supports this
  for subclassed models). You then get `EarlyStopping`,
  `ModelCheckpoint(save_best_only=True)`, `ReduceLROnPlateau`, `TensorBoard`
  and `CSVLogger` for free instead of the hand-written loop in `training.py`.
- **Save a single `.keras` file** (or a SavedModel with a `tf.function`
  signature) so inference does not have to rebuild the architecture from
  `config.json` and call `build_variables()` first.
- **Unit tests** for `preprocessing`, `Tokenizer` round-tripping,
  `group_entities`, and a one-batch training smoke test - fast, and they catch
  most regressions.
- **Config from YAML/JSON** (`--config train.yaml`) in addition to CLI flags,
  so experiments are reproducible and diffable; log the resolved config
  alongside the artifacts (already done) plus the git commit hash.
- **Write the held-out split to disk** (`artifacts/val.txt`) so a later
  evaluation step uses exactly the sentences the model never saw.
- Add `seqeval` to `requirements.txt` once entity-level F1 is adopted.

---

## If you only do one thing

Switch to a sentence-level tagger (BiLSTM-CRF or fine-tuned BERT) with
validation-F1 early stopping on a sentence-level split. That addresses items
1, 3 and most of 4 at once, and should matter more than everything else on
this list combined. The change fits the existing package structure:
`data.py` reads sentences, `model.py` holds the tagger, `train.py` stays the
entry point.
