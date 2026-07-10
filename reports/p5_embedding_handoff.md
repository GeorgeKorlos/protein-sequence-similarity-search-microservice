# P5 Embedding Handoff

## Purpose
ESM-2 650M embeddings of the full SwissProt corpus, produced by P4, consumed by P5
for retrieval-augmented DTI prediction and embedding ablation.

## Artifacts

| File | Format | Shape/Size | Description |
|---|---|---|---|
| swissprot_embeddings.npy | NumPy float32 | (547205, 1280) | L2-normalized mean-pooled ESM-2 embeddings |
| swissprot_ids.txt | Plain text | 547205 lines | UniProt accessions, row-aligned with npy |
| swissprot_clean.csv | CSV | 547205 rows | Corpus metadata: id, sequence, organism, keywords, go_terms |
| embedding_config.json | JSON | - | Embedding provenance |


## Provenance
- Model: facebook/esm2_t33_650M_UR50D
- Pooling: mean over residue tokens (BOS/EOS/PAD masked by token id)
- Normalization: L2 post-pooling
- Precision: fp16 inference, cast to float32 at storage
- Corpus: SwissProt 2026_01, 547,205 sequences (<=1024 aa, non-fragment)
- Corpus SHA256: 815b9c416dba10200840df8ba925c0d104a99c4ba72004839ee5a0bf04b202e4
- Build date: 2026-07-07
- Hardware: Vast.ai RTX 3090, ~82 min, $0.25

## Quality Evidence
GO-term functional similarity proxy (see reports/embedding_comparison.md):

| Embedder | MRR | AUROC | Hit@10 |
|---|---|---|---|
| ESM-2 650M | 0.9907 | 0.7178 | 0.998 |
| Physicochemical | 0.4204 | 0.5231 | 0.728 |
| Random | 0.2991 | 0.4853 | 0.694 |

AUROC for ESM-2 is computed on 64/500 queries only (435/436 skipped queries had
all-positive neighborhoods). MRR is the reliable primary metric.

## P5 Usage Notes
- Row order in npy matches line order in swissprot_ids.txt exactly.
- Vectors are unit-norm — inner product equals cosine similarity.
- Ablation (150M, 3B) must use identical pooling and normalization for fair comparison.
- Norms unit to within 1e-3 (fp16->float32 cast).