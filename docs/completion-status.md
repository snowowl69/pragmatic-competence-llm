# Evidence required for completion

The software supports the workflow below. The research is not complete merely because the workflow is implemented or tests pass.

| Requirement | Current evidence | Work still required |
|---|---|---|
| Unified benchmark | Frozen v2 splits, hashes, labels, exact leakage checks | Review near duplicates and annotation provenance |
| CPU reference | Saved models and all 4,852 test predictions in both context conditions | These results do not establish Llama performance |
| Llama baseline | Evaluation code and GPU notebook | Execute the full frozen test set and retain model identity and benchmark hashes |
| QLoRA training | Four configurations, trainer, live telemetry, adapter export | Train joint and category-specific adapters on an adequate GPU; retain logs and hardware records |
| Fair comparisons | Matching setting/template/context/run checks and paired test-ID checks | Execute baseline and adapter comparisons; verify full coverage |
| Context and transfer | Orchestrated context ablation and four-condition transfer | Run the experiments; missing cells remain unmeasured |
| Interpretation quality | Separate explanation generation and blinded human-review page | Generate actual explanations; recruit independent reviewers, collect ratings and adjudicate disagreements |
| Failure analysis | Confusion matrices and per-label scores | Review LLM errors and unsupported interpretations |
| Demonstration | Real CPU predictions, separate graph page, numeric matrices and review page | Load the base model and trained adapter for the intended LLM comparison |
| Licences and report | Dataset card and evidence-qualified report | Resolve source-licence/provenance questions; incorporate actual GPU and human results |

## GPU execution

Use `notebooks/colab_pipeline.ipynb` with an adequate CUDA GPU and authorized Hugging Face access to the chosen Llama weights. The configured entry point is:

```text
python src/run_pipeline.py llm --conditions joint,sarcasm,implicature,indirect
python src/run_pipeline.py report
```

The entry point evaluates the baseline and all selected adapters with context, without context, few-shot prompts and the second template. It also generates a separate explanation sample using the same greedy decoding settings for each condition. The sample uses seed 42 and up to 100 held-out items per category, so all 22 speech-act test items are included. Classification prompts still request labels only. Explanation generation is a separate experiment and does not alter classification scores.

The local machine has an NVIDIA MX330 with 2 GB VRAM and cannot execute the configured 8B QLoRA experiment. The project entry point checks for at least 14 GiB; this is a minimum guard, not a guarantee that every GPU/configuration will fit. No cloud account is connected. No real 8B training has been performed by adding these features.

## Human review

Copy the generated `results/interpretations` folder back with the matching benchmark, or run the app beside those files. Open `/review`, enter a reviewer ID, and select Next item. Model identity and other reviewers' scores are hidden from the rating view. Score meaning, use of context and grounding on the displayed 0–2 rubric, add a rationale, attest independence, and save. The local append-only file `results/human/ratings.jsonl` retains ratings; the latest rating for each reviewer/item is used in agreement summaries.

Use at least two genuine independent reviewers. Pseudonymous IDs are not an identity-verification system. The coverage panel reports paired counts and quadratic weighted Cohen's kappa for each dimension; constant ratings produce an undefined kappa. Review disagreements separately and retain an adjudication record before reporting final interpretation quality. Do not replace independent human scores with AI-generated ratings.

## Navigation and live graphs

The Analyze page links to `/graphs` and `/review`. All graphs, confusion matrices and numeric test/per-label matrices are on `/graphs`. Browser-local label history survives navigation, contains no input text, and can be cleared from the graph page. It is not a measure of held-out accuracy. Research plots poll saved files every five seconds; they do not start a training job or observe a disconnected cloud runtime.
