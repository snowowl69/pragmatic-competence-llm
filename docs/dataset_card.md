# Pragmatic Competence Benchmark v2

Built from the committed raw files with seed 42. Manifest: `data/benchmark/v2/manifest.json`.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| sarcasm | 517 | 68 | 104 |
| implicature | 2000 | 3251 | 4726 |
| indirect_speech_act | 108 | 15 | 22 |

Circa questions and MUStARD dialogue contexts are grouped before splitting. Exact normalized replies shared across splits are excluded, and training is capped at 2,000 examples per category. These are within-source grouped splits, not source-dataset-disjoint splits. Training and validation instruction files are derived from v2 only. The original v1 files remain solely for provenance and must not be used for new experiments.

The committed iSarcasm CSVs contain tweet IDs without text, so they are excluded. Circa is parsed as literal eight-column TSV with quoting disabled. Conditional yes is mapped to uncertain because it is not an unconditional yes; uncertain also includes NA and ambiguity. This mapping sacrifices nuance and must be reported in all comparisons.

MUStARD meanings are generic label-derived explanations, not human-authored meaning gold standards. Circa meanings similarly restate the classification target. Do not use these as evidence of explanation quality. Indirect speech-act examples are author-provided and small; no new independent human annotation has been performed. Exact leakage checks pass; near-duplicate and template-family leakage, pretraining contamination, and cultural bias remain limitations. Automated splits do not establish human validity.

Source references: [MUStARD](https://github.com/soujanyaporia/MUStARD), [Circa](https://github.com/google-research-datasets/circa), [iSarcasm](https://github.com/silviu-oprea/iSarcasm). The Circa README describes Attribution 4.0 but links to BY-SA 4.0; resolve this discrepancy with the authors before redistribution beyond the existing repository. Source license placeholders are not verified grants.
