# Embedding Comparison: GO-term Functional Similarity Proxy Task

## Task
Evaluates whether embedding proximity reflects functional similarity between
proteins, using GO-term overlap as the relevance signal. Serves as an indirect
proxy for downstream DTI retrieval (P5).

## Setup
- Corpus: SwissProt 2026_01, 547,205 sequences with ≥1 GO annotation
- Queries: 500 randomly sampled (seed=42)
- Top-K: 10
- Metrics: AUROC, Mean Reciprocal Rank (MRR), Hit@10

## Results

| Embedder | Dim | AUROC | MRR | Hit@10 | Scored | Skipped | Pure-positive |
|---|---|---|---|---|---|---|---|
| ESM-2 650M | 1280 | 0.7178 | 0.9907 | 0.998 | 64 | 436 | 87.0% |
| Physicochemical | 4 | 0.5231 | 0.4204 | 0.728 | 343 | 157 | 4.2% |
| Random | 1280 | 0.4853 | 0.2991 | 0.694 | 347 | 153 | 0.0% |

## Interpretation
ESM-2 dominates both baselines. MRR 0.9907 and Hit@10 0.998 mean relevant
proteins almost always sit at the top of the retrieved set.

AUROC (0.7178) is computed on only 64 of 500 queries. Of the 436 skipped, 435
had all-positive top-10 neighborhoods — ESM-2 retrieves functional matches so
consistently that no negatives remain for AUROC to score. The baselines invert
this: their skips are almost entirely all-negative (physico 136/157, random
153/153), meaning their neighborhoods rarely contain a functional match. So MRR,
not AUROC, is the reliable metric here.

Physicochemical barely beats random (AUROC 0.5231 vs 0.4853): bulk amino-acid
properties carry weak functional signal but don't organize proteins by role.

For P5, ESM-2 provides a strong retrieval substrate — neighbors are enriched for
shared functional context.

## Limitations
- GO-term overlap is a coarse proxy for function.
- The >=1-shared-GO-term relevance rule is permissive, which inflates MRR/Hit@10
  and collapses AUROC coverage. A stricter criterion (GO semantic similarity,
  Resnik/Lin, or a Jaccard threshold) would restore AUROC; planned for P5.
- 500 queries may underrepresent rare functional categories.
- Functional similarity does not guarantee shared drug-binding behavior.