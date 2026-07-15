# Study

BERT / mini BERT pretraining study notes.

## Mindmap

Open: https://jsk900210-oss.github.io/Study/

Click a branch to fold or unfold. Drag to move, scroll to zoom.

## Files

| File | Description |
|---|---|
| index.html | Interactive mindmap (markmap) |
| bert_project.md | Full study notes converted from the Jupyter notebook |

## Contents

1. Word embedding limits and why contextual embedding was needed
2. Transfer learning and language modeling
3. ELMo vs GPT vs BERT: the difference is which direction they look
4. BERT internals: three embeddings, MLM, NSP, fine-tuning
5. mini BERT project: building a 1M parameter BERT from scratch
6. After BERT: ALBERT, Transformer-XL, XLNet, T5, Switch Transformer, ERNIE

## mini BERT project

Built a BERT with about 1M parameters and pretrained it on the Korean Wikipedia corpus.
BERT-large has 340M parameters, so this is roughly 1/300 the size, but every step is the same.

Key points:

- SentencePiece BPE tokenizer, vocab 8,000 plus 7 special tokens
- Whole-word masking so the model learns context, not spelling
- MLM 80:10:10 rule and why not 100 percent MASK
- NSP pair generation with 50:50 true and false
- np.memmap to handle a 1.4GB dataset in limited memory
- Config: d_model 96, n_head 3, d_head 32, d_ff 256, n_layer 3
- About 73 percent of the parameters live in the embedding matrix alone
