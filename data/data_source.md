## Dataset Identity

* **Dataset name**: UniProt SwissProt (reviewed subset)
* **Release**: 2026_01
* **Source**: https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/

## File 

* **Filename**: uniprot_sprot.fasta.gz
* **Format**: FASTA (compressed)
* **Size**: 93,457,057 bytes
* **Last Modified (source)**: 2026-01-28 23:00
* **Download date**: 2026-04-15 

## File Integrity Verification

* **Method**: MD5 Checksum (provided by Uniprot)
* **Expected MD5**: 5245b19456d9a063b13c46602269bc5f 
* **Computed MD5**: 5245b19456d9a063b13c46602269bc5f 
* **Status**: Match (integrity verified)
* **Secondary checksum (local, SHA256)**: 5ba5cb332fc7794ab1c02075a79c8b3d95b573f9b244a38bb53558172e1f9b7b  

## Dataset Characteristics

* **Type**: protein sequences (UniProt SwissProt reviewed subset)  
* **Scale**: 574,627 sequences  
* **Taxonomic scope**: heterogeneous (bacteria, viruses, archaea, eukaryotes)  

* **Annotation fields**:
- UniProt accession IDs
- Entry names
- Protein names and descriptions
- Organism source (OS)
- Taxonomy IDs (OX)
- Gene names (GN)
- Protein existence evidence (PE levels)
- Sequence version (SV)

* **Sequence properties**:
- Amino acid sequences (20 standard residues + occasional ambiguous symbols)
- Variable lengths (observed range: short peptides to >1000 residues)
- Mean pooling-compatible representation target

* **Preprocessing constraints**:
- Remove fragment sequences
- Cap sequence length at 1024 amino acids
- Preserve full annotation metadata for downstream retrieval