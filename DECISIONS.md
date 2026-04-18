## ESM-2 Model Size

## FAISS Index Type

## Distance Metric

## Pooling Strategy

## Sequence Length Cap

## Batch Size

## Corpus Choice

**Choice:** SwissProt reviewed subset (UniProtKB/Swiss-Prot), release 2026_01

**Source files:**
- `uniprot_sprot.fasta.gz` — sequence pipeline
- `uniprot_sprot.dat.gz` — functional annotations (UMAP coloring, DTI baseline stratification in Week 6)

**Provenance:** See `data/data_source.md` (MD5 verified for both files)

**Corpus statistics (post-filtering):**
- Raw sequences: 574627 
- Post-fragment-filter: 565361
- Post-length-cap (≤1024 aa): 547205
- Final: 547205 sequences
- Length: min=2, mean=324.3, median=289, max=1024, p95=729
- Keyword coverage: 98.84%
- GO term coverage: 96.39%

**Rationale:**
SwissProt is manually reviewed — every entry has experimentally supported functional annotation. This matters for two reasons: (1) embedding quality evaluation in Week 6 requires reliable ground-truth functional labels; random or automated annotations would make the DTI baseline comparison uninformative. (2) SwissProt is the standard benchmark corpus in protein representation learning literature, making results directly comparable to published work. Alternatives (TrEMBL, PDB sequences) were rejected: TrEMBL is computationally predicted and annotation quality is uneven; PDB is structurally biased and ~10x smaller.

**Citation:** The UniProt Consortium, *Nucleic Acids Research* 2023. DOI: 10.1093/nar/gkac1052

## Embedding Normalization

## Cloud Platform
