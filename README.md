# mGPT

I built this little character-level GPT in PyTorch to really understand how GPT models work, especially self-attention and what a transformer is doing behind the scenes.

It's a simplified **decoder-only transformer**: you give it some text, it learns the patterns, and then it writes new text in the same style. It's tiny, but the architecture is the same idea as the big models. Scale up the hyperparameters, train on a much larger corpus from the internet, and train for far longer, and you'd get something like a base-version chatbot.

## Project structure

| File | Description |
|------|-------------|
| [v1.py](v1.py) | Baseline bigram language model (token embedding table only, no attention). |
| [v2.py](v2.py) | Full decoder-only transformer: multi-head causal self-attention, feed-forward blocks, residual connections, LayerNorm, dropout. |
| [input.txt](input.txt) | Training corpus (~1.1 MB of Shakespeare text). |
| [outputs](outputs) | Sample generations before and after training. |

## Models

### v1 – Bigram model
The simplest starting point. It guesses the next character by looking only at the current one, with no attention at all.

| Hyperparameter | Value |
|---|---|
| batch_size | 32 |
| block_size | 8 |
| max_iters | 3000 |
| learning_rate | 1e-2 |

### v2 – Transformer
| Hyperparameter | Value |
|---|---|
| batch_size | 32 |
| block_size (context) | 64 |
| n_embd | 100 |
| n_head | 4 |
| n_layer | 4 |
| dropout | 0.2 |
| learning_rate | 3e-4 (AdamW) |
| max_iters | 5000 |
| eval_interval | 500 |

A few other details: it works on individual characters, uses a 90/10 train/validation split, and learns its positional embeddings. A causal mask keeps each position from peeking at future characters.


## Results

Here's what the model writes, starting from the same blank context (token 0) each time.

**Before training** (random weights), it's pure noise:

```text
OerIzs;zjxTYYEouOs,L&R:hwpIlXzEK3 yDv'nx3$:WisOItCmQs:X;Fr'.tCkVeOnQ:LJPCV FdgjRMgqgyOGN!NNOBi,H.sJ'GfUEFumaVxCFUQNdVJtU TxNl:SiagM:VyKXGoGXb ympO:Mgis3l.HuwNfGwUV&gF$'VsFsJs CH:JipedEefdetVD&FNJoOi&bnAVpwN:BfIHnvxDPAXg-e&.cdAlJWx$VnZihKC JgjD&KXPEiJcHdxJJ& VSiXW&HeG-E3D&xGjqZEZUREOB&iL$fs'iQEYF
njG.-idZGsoy3'lH .EM:JCFw'zOgcxn'3iyp'-lGBc3msihV ViaEJfpejp;eADWtIpwgngTUypJfiMdE$IXZ&FwisayOs;eG&ldPgbwHXUy.pVV Ma?LOZsZDSSpoMMMqKV-.GTsstzfpNWGArpeEweseMig,. .b  ZOLvpULBXMzp&!Z?wNoWJcVMM;xVN-jvjahpS:
```

**After training** (iteration 5000, loss ≈ 1.85): it has picked up the shape of a play: speaker names followed by colons, line breaks, capital letters and punctuation. It also produces English-looking words. Most of them are still misspelled and the text doesn't make much sense, but that's a big step up from noise.

```text
AANTES:
But fancouse Share to sufford be go to-glater
dide!
What! befow he powder.

Firce:
In'll surn hast's frewwere of that;
You me his uneels, bady your wide,
Thon apking latorur fant him!
I have one thy horgholk speod; you lett for bethe theies ans fair more of the on,
You have hust loope of deamlyour my says!
'Clet prade in of this in you, to-lan abby forthed, fat sir,
make I faor I than, punn'er'd;
And what reweove yehalow beatter our forth mir
Now, me God bard drewand I kill my king,
Thi
```

## Possible improvements

- Bigger `n_embd`, `n_layer` and `block_size`, plus longer training.
- A subword tokenizer (like BPE) instead of single characters.
- A larger, more varied training corpus.
- Fine-tuning on conversational data to get closer to a chatbot.
