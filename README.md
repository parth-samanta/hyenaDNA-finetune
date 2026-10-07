# HyenaDNA pooling

Comparison of seven sequence aggregation methods for splice-acceptor classification using **partial fine-tuning of a pretrained HyenaDNA encoder**. The experiment asks whether retaining regional information and learning which nucleotide states to emphasize improves classification while adding few pooling parameters.

The study uses the original HyenaDNA paper's splice-acceptor dataset and contains **seven aggregation methods × three seeds = 21 completed training runs**, with a saved holdout evaluation for every run.

## Model and partial fine-tuning

The starting checkpoint is [`LongSafari/hyenadna-tiny-1k-seqlen-d256`](https://huggingface.co/LongSafari/hyenadna-tiny-1k-seqlen-d256). Its architecture is recorded locally in `checkpoints/tiny/config.json`; each run also saves its resolved `architecture.json`.

| Architecture component | Setting |
| --- | --- |
| Hyena blocks | **2**, indexed `0` and `1` in the code |
| Hidden width | **256** features per nucleotide |
| Feed-forward network in each block | `256 → 1,024 → 256`, with GELU |
| Sequence mixer in each block | Hyena operator, order 2; depthwise short convolution with kernel size 3 and an implicit long filter |
| Implicit-filter network | Width 64; positional input dimension 5 |
| Checkpoint filter context | 1,026 positions; these experiments use **600 nucleotides** |
| Embedding table | 16 × 256; the configured vocabulary of 12 is padded to a multiple of 8 |
| Task output | Two logits: non-acceptor (`0`) and acceptor (`1`) |

One nucleotide is one token: `A/C/G/T/N` map to IDs `7/8/9/10/11`. Padding uses ID `4`. No BOS or EOS token is appended. Aggregation excludes padding; `N` is a valid nucleotide token. The encoder produces a sequence of 256-dimensional states.

```text
600 nucleotide tokens
    ↓
Token embeddings                 frozen
    ↓
Hyena block 0                    frozen
    ↓
Hyena block 1                    trainable: mixer + MLP + both block norms
    ↓
Final LayerNorm                  trainable
    ↓
Selected aggregation             trainable when it has learned parameters
    ↓
Linear projection → GELU         trainable; output width 256
    ↓
Dropout(0.1) → Linear(256, 2)     trainable
    ↓
Two-class cross-entropy loss
```

**We trained the entire second Hyena block plus the final LayerNorm**, starting at epoch 1. The embeddings and first block stay frozen throughout training. Pooling parameters, the projection and the classifier are jointly optimized with the second block. The frozen encoder modules run in evaluation mode, while the second block runs in training mode.


### Exact encoder parameter counts

Counts below count unique scalar `nn.Parameter` entries, including biases and normalization parameters. Fixed buffers, such as filter positional coordinates and modulation decay values, are excluded. The pretrained language-model output head is not used.

| Encoder component | Total parameters | Trainable parameters |
| --- | ---: | ---: |
| Token embeddings | 4,096 | 0 |
| Hyena block 0 | 818,240 | 0 |
| Hyena block 1 | 818,240 | 818,240 |
| Final LayerNorm | 512 | 512 |
| **Encoder total** | **1,641,088** | **818,752** |

Each block's 818,240 parameters comprise:

| Component inside one block | Parameters |
| --- | ---: |
| Hyena input projection, `256 → 768` | 197,376 |
| Depthwise short convolution | 3,072 |
| Implicit-filter parameters, including learned frequencies and filter bias | 25,408 |
| Hyena output projection, `256 → 256` | 65,792 |
| **Hyena mixer subtotal** | **291,648** |
| Feed-forward MLP, `256 → 1,024 → 256` | 525,568 |
| Two LayerNorms | 1,024 |
| **Block total** | **818,240** |

Thus **822,336 encoder parameters remain frozen**. This approach updates approximately half the encoder; it is not a method that updates only a small pooling head.

## Aggregation methods

For a sequence with valid nucleotide states $h_1,\ldots,h_L\in\mathbb{R}^{256}$, the seven methods are:

| Method / config name | How it creates the sequence representation | Output width |
| --- | --- | ---: |
| Mean / `mean` | Average all valid nucleotide states | 256 |
| Max / `max` | Take the maximum over positions independently for each feature | 256 |
| Last token / `last` | Use the final valid nucleotide's state | 256 |
| Global attention / `attention` | Learn a linear score per state, normalize scores across all valid positions, then take their weighted average | 256 |
| Regional means / `regional` | Split valid positions into four ordered bins, average within each, concatenate the four summaries | 1,024 |
| Regional attention / `regional_attention` | Use one shared learned query to score states; normalize separately within each of four bins, then concatenate weighted summaries | 1,024 |
| Residual regional attention / `residual_regional_attention` | Learn one gate per bin to blend its ordinary mean with its attention-weighted summary; concatenate the four blends | 1,024 |

The four bins preserve sequence order. For the 600-nucleotide inputs they contain positions **1–150, 151–300, 301–450 and 451–600**. For other lengths, bin assignment uses the valid-token rank, so padding never moves a nucleotide into a different bin.

Global attention uses a learned weight vector $w$ and bias $b$:

$$
s_i=w^\top h_i+b,\qquad
\alpha_i=\frac{\exp(s_i)}{\sum_{j=1}^{L}\exp(s_j)},\qquad
z=\sum_{i=1}^{L}\alpha_i h_i.
$$

Regional attention uses a **single shared 256-parameter query** $q$, with a separate softmax within each region $R_r$:

$$
a_r=\sum_{i\in R_r}
\frac{\exp(q^\top h_i)}{\sum_{j\in R_r}\exp(q^\top h_j)}h_i,
\qquad z=[a_1;a_2;a_3;a_4].
$$

There is no query/key/value matrix or pairwise self-attention in this pooling layer. The query starts at zero, so the initial summaries equal regional means. In the residual variant, with regional mean $m_r$, the summary is

$$
z_r=(1-g_r)m_r+g_ra_r,\qquad g_r=\operatorname{sigmoid}(\gamma_r).
$$

Four learned gate logits $\gamma_r$ start at zero, giving $g_r=0.5$. Pooling work scales linearly with sequence length and hidden width for a fixed four bins; it does not construct an $L\times L$ attention matrix.

These bins describe **position within the supplied sequence**, not verified upstream/site-centered/downstream biological regions. Coordinate-based `site_regions` is supported separately but was not included in either completed study, because verified site indices were not supplied for every example.

### Exact parameters for the complete classification model

All methods feed a trainable linear projection into the same 256-dimensional classifier input. The projection has $256D+256$ parameters for aggregation width $D$; GELU has none. The final classifier has $256\times2+2=514$ parameters.

| Aggregation | Pooling parameters | Projection parameters | Classifier parameters | Total model parameters | Trainable parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean | 0 | 65,792 | 514 | 1,707,394 | **885,058** |
| Max | 0 | 65,792 | 514 | 1,707,394 | **885,058** |
| Last token | 0 | 65,792 | 514 | 1,707,394 | **885,058** |
| Global attention | 257 | 65,792 | 514 | 1,707,651 | **885,315** |
| Regional means | 0 | 262,400 | 514 | 1,904,002 | **1,081,666** |
| Regional attention | 256 | 262,400 | 514 | 1,904,258 | **1,081,922** |
| Residual regional attention | 260 | 262,400 | 514 | 1,904,262 | **1,081,926** |

For example, mean pooling trains `818,752 + 65,792 + 514 = 885,058` parameters. Regional attention trains `818,752 + 256 + 262,400 + 514 = 1,081,922`.

Regional methods use **196,608 more projection parameters** because their concatenated representation is wider. The comparison therefore holds encoder architecture, trainable encoder blocks and classifier width constant, but does not hold total parameter count constant. Comparing regional means with regional attention isolates a much smaller change: **256 additional pooling parameters**; the residual version adds another **four**. The trainable fraction of the complete model ranges from approximately 51.84% to 56.82%.

These counts were checked against instantiated models and the saved `complete.json` metadata for all 21 completed runs.

## Training and checkpoint selection

Settings are defined in [`configs/base.json`](configs/base.json); each run saves the fully resolved settings in its own `config.json`.

| Control | Setting used in the study |
| --- | --- |
| Fine-tuning | Partial from the first epoch; one independent model per aggregation and seed |
| Seeds | `0, 1, 2` for model initialization and training randomness |
| Training data fraction | 100% of the prepared training partition |
| Optimizer | AdamW; default betas `(0.9, 0.999)` and epsilon `1e-8` |
| Trainable encoder learning rate | `1e-4` |
| Pooling, projection and classifier learning rate | `1e-3` |
| Weight decay | `0.1` on eligible weight matrices; `0` on parameters with fewer than two dimensions and all Hyena mixer parameters |
| Loss | Ordinary two-class cross-entropy; no class weighting or label smoothing |
| Microbatch / effective batch | 4 / 256 examples; accumulate 64 microbatches per full optimizer update |
| Gradient handling | Normalize accumulated gradients by actual example count; clip global gradient norm to `1.0` |
| Learning-rate schedule | 1% linear warmup, then cosine decay toward 10% of the initial LR over the planned 30-epoch budget |
| Maximum epochs | **30** |
| Early stopping | **7 consecutive epochs without a strictly higher validation macro F1** |
| Checkpoint selection | Save the checkpoint with the highest validation macro F1, rather than using the final epoch automatically |
| Classifier dropout | `0.1`, before the final two-logit linear layer |
| Backbone dropout during partial training | Inactive: frozen modules run in evaluation mode; the trained block's residual dropout is `0.0` |
| Sequence handling | Single forward orientation; no reverse-complement augmentation or bidirectional pass |
| Additional pretraining | None; `tapt_epochs=0` |
| Feature caching | Disabled; the trained block changes throughout training |
| LR trials | One fixed head LR per method; no LR search in this study |
| Execution | FP32; Python 3.11.16, PyTorch 2.6.0+cu124; NVIDIA GeForce RTX 4060 Laptop GPU |

The configured embedding dropout is `0.1`, but it is disabled by evaluation mode in the frozen encoder. The loss is summed within microbatches, then gradients are divided by the number of examples accumulated. The final, smaller accumulation group is retained: all training examples are used every epoch.

An epoch contains **78 optimizer updates**: 77 updates of 256 examples and one update of 248 examples. Every epoch uses all 19,960 training records.

**30 epochs is a ceiling, not a fixed duration.** Actual training lasted 10–30 epochs. The schedule retains its planned 30-epoch horizon if early stopping occurs. `history.jsonl` contains every trained epoch; `validation.json` describes the best epoch. For example, original-data regional attention with seed 0 trained for 10 epochs and selected epoch 3. Different stopping epochs follow the same selection rule; they are not different configured epoch budgets.

Evaluation uses the class-1 softmax probability. Predictions use `probability > 0.5`; macro F1 averages the F1 of both classes. Reported F1 is the **mean ± sample standard deviation** across three separate seed runs, not an ensemble prediction. AUROC below is the mean across those runs. The CSV also includes accuracy, binary F1, MCC and average precision; its `auprc` field contains sklearn average precision rather than trapezoidal PR area.

## Datasets and splits

The task is binary splice-acceptor classification using 600-nucleotide sequences.

| Partition | Examples | Negative (`0`) | Positive (`1`) |
| --- | ---: | ---: | ---: |
| Training | 19,960 | 9,899 | 10,061 |
| Validation | 2,218 | 1,100 | 1,118 |
| Evaluation | 2,218 | 1,100 | 1,118 |

Validation and evaluation reference the **same** holdout sequences. The original training and holdout FASTA partitions are retained without repartitioning.

## Completed results

All seven methods have three completed training runs and three saved holdout evaluations.

| Aggregation | Holdout macro F1 | Holdout AUROC |
| --- | ---: | ---: |
| Mean | 0.9073 ± 0.0031 | 0.9626 |
| Max | 0.9023 ± 0.0098 | 0.9609 |
| Last token | 0.8622 ± 0.0042 | 0.9318 |
| Global attention | 0.9000 ± 0.0041 | 0.9594 |
| Regional means | 0.9280 ± 0.0039 | 0.9756 |
| Regional attention | 0.9390 ± 0.0052 | 0.9838 |
| Residual regional attention | 0.9358 ± 0.0036 | 0.9833 |

Regional attention has the highest mean macro F1: **0.9390**. Its increase over regional means is approximately **1.10 percentage points**, for 256 additional pooling parameters. Three seeds describe optimization variation; they do not establish statistical significance. The holdout is also used for checkpoint selection, as described above.



