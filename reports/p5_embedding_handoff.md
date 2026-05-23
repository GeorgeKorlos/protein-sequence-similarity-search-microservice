# P5 Embedding Handoff

## Purpose
ESM-2 650M embeddings of the full SwissProt corpus, produced by P4, consumed by P5
for retrieval-augmented DTI prediction and embedding ablation.

## Artifacts

| File | Format | Shape/Size | Description |
|---|---|---|---|
| swissprot_embeddings.npy | NumPy float32 | (547205, 1280) | L2-normalized mean-pooled ESM-2 embeddings |
| swissprot_ids.txt | Plain text | 547205 lines | UniProt accession IDs, row-aligned with npy |
| swissprot_clean.csv | CSV | 547205 rows | Corpus metadata: id, sequence, organism, keywords, go_terms |
| embedding_config.json | JSON | — | Embedding provenance: model, pooling, normalization, corpus hash |

## Loading

```python
import numpy as np

embeddings = np.load("data/swissprot_embeddings.npy", mmap_mode="r")  # (547205, 1280)
with open("data/swissprot_ids.txt") as f:
    ids = [line.strip() for line in f if line.strip()]
id_to_idx = {uid: i for i, uid in enumerate(ids)}
```

## Embedding Provenance

- Model: facebook/esm2_t33_650M_UR50D
- Pooling: mean, mask-aware, excluding BOS/EOS tokens
- Normalization: L2 post-pooling
- Inference precision: fp16 (cast to float32 before storage)
- Corpus: SwissProt 2026_01, 547,205 sequences (≤1024 aa, non-fragment)
- Corpus SHA256: 815b9c416dba10200840df8ba925c0d104a99c4ba72004839ee5a0bf04b202e4
- Build date: 2026-05-02
- Hardware: Vast.ai RTX 3090, 88 min, $0.25

## Quality Evidence

GO-term functional similarity proxy task (see reports/embedding_comparison.md):
- ESM-2 650M: MRR=0.991, AUROC=0.706, Hit@10=0.998
- Physicochemical (4-dim): MRR=0.426, AUROC=0.515
- Random (1280-dim): MRR=0.299, AUROC=0.485

AUROC for ESM-2 is computed on 62/500 queries only — 437/438 skipped queries had
all-positive top-10 neighborhoods, reducing AUROC discriminability. MRR is the
reliable primary metric.

- ESM-2 650M AUROC: 0.706 vs physicochemical 0.515 vs random 0.485

## P5 Usage Notes

- Row order in npy matches line order in swissprot_ids.txt exactly
- Vectors are unit-norm — inner product equals cosine similarity
- For P5 ablation: 150M and 3B variants should use identical pooling and normalization
  for fair comparison
- Norms are unit to within 1e-3 due to fp16→float32 cast at storage time