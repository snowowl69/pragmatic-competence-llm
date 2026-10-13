# Evaluating Pragmatic Classification Across Sarcasm, Implicature, and Indirect Speech Acts

Research manuscript draft — CPU reference experiment completed; LLM experiments pending.

## Abstract
We construct a grouped English benchmark of three pragmatic phenomena and provide a reproducible evaluation and QLoRA training pipeline. An audit of the original repository revealed tweet IDs used as text, malformed quote handling, and unreliable illustrative model comparisons. We correct these issues and freeze benchmark v2. A supervised TF-IDF logistic regression reference achieves accuracy of 0.625 for sarcasm, 0.573 for implicature, and 0.500 for indirect speech acts with context. These measurements are not LLM results and establish neither QLoRA improvement nor cross-category transfer. GPU experiments and independent annotation remain necessary to answer the project's main research questions.

## Motivation and questions
Pragmatic interpretation depends on conversational context and speaker intent rather than surface wording alone. We study classification accuracy across phenomena (RQ1–2), transfer from phenomenon-specific versus joint fine-tuning (RQ3), context removal (RQ4), and failure types (RQ5). This draft reports a CPU reference and the experimental protocol; RQ1–3 for LLMs remain unanswered.

## Data and methodology
The benchmark uses MUStARD dialogue sarcasm, Circa indirect answers, and 145 author-provided indirect speech-act examples. iSarcasm is excluded because available files contain identifiers only. Related Circa questions and MUStARD dialogue contexts are held within one split; exact reply collisions are excluded across splits. Training is capped per category. The benchmark is not source-dataset-disjoint. The frozen manifest records SHA-256 hashes.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| sarcasm | 517 | 68 | 104 |
| implicature | 2000 | 3251 | 4726 |
| indirect_speech_act | 108 | 15 | 22 |

Labels are sincere/sarcastic, yes/no/uncertain, and statement/request/question/suggestion/complaint. Circa conditional affirmative answers are conservatively placed in uncertain. Accuracy and macro-F1 are reported per category; pooled overall accuracy is dominated by Circa and should not be the headline metric.

## Completed reference experiment
We fit per-category TF-IDF word unigram/bigram vectorizers (30,000 feature maximum, sublinear term frequency) and class-balanced logistic regression (seed 42, 1,000 iteration maximum). Hyperparameters were fixed before testing. The majority baseline uses the most common training label. Separate models are trained with context and with utterance alone. This is a supervised lexical reference, not instruction tuning.

| Category | Test N | Majority accuracy | Context accuracy | Utterance-only accuracy |
|---|---:|---:|---:|---:|
| sarcasm | 104 | 0.423 | 0.625 | 0.596 |
| implicature | 4726 | 0.452 | 0.573 | 0.612 |
| indirect_speech_act | 22 | 0.545 | 0.500 | 0.409 |

Predictions, confusion matrices, macro-F1, and paired context-ablation statistics are in `results/local/`. Context is beneficial in sarcasm and indirect speech acts for this reference, but reduces implicature accuracy. Small indirect test size (22 items) limits conclusions. No result was altered to force improvement.

## Planned LLM protocol
Evaluate Llama-3.1-8B-Instruct zero-shot/few-shot with two prompt variants and context removal. Train joint and three category-specific QLoRA adapters using completion-only loss, seed 42, three epochs, rank 16, effective batch size eight, and validation-loss checkpoint selection. Evaluate all adapters on the identical frozen test split. Build the 4-by-3 transfer matrix and compare matched zero-shot experiments with paired McNemar tests. Report raw p-values and apply a multiple-comparison correction when making confirmatory claims. Optional Mistral evaluation provides an independent lineage comparison. API upper-bound evaluation requires a separately authorized account.

## LLM results
Pending execution on a GPU. No baseline LLM predictions, trained 8B adapters, transfer gains, or model significance conclusions have been produced here. The Colab notebook runs the implemented experiment and exports artifacts. GPU availability, gated Llama access, and timeouts may affect execution. A successful tiny training smoke test, if available, verifies plumbing only.

## Validity and limitations
No new double annotation or inter-annotator agreement measurement is available. Existing source label-derived explanations cannot measure generated explanation quality. Public-source pretraining contamination cannot be ruled out. Grouping addresses exact replies and questions but not all semantic paraphrases or author-template families. The indirect dataset is small and not an independent population sample. The three-label Circa mapping loses conditional nuance. Resolve source-license ambiguity before broader redistribution.

## Reproducibility
Run `python src/run_pipeline.py prepare`, then run the GPU stage using `notebooks/colab_pipeline.ipynb`, and `python src/run_pipeline.py report`. CPU inference is available through `python app/demo.py --local`. The illustrative mode is explicitly marked and cannot serve as research evidence.

## References
- Castro et al. (2019), Towards Multimodal Sarcasm Detection. https://github.com/soujanyaporia/MUStARD
- Louis, Roth, and Radlinski (2020), “I'd rather just go to bed”: Understanding Indirect Answers. https://github.com/google-research-datasets/circa
- Hugging Face TRL SFT documentation. https://huggingface.co/docs/trl/sft_trainer
