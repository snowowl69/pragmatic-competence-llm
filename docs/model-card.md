# Model card

## Model roles and current status

The available measured reference is TF-IDF with word unigrams/bigrams and logistic regression, trained separately by task and input condition. Its comparison baseline predicts the majority training label. It classifies examples and does not generate explanations.

The planned main language model is Llama-3.1-8B-Instruct with project-specific QLoRA adapters. Configuration and execution code are present. No completed full 8B baseline evaluation or trained project adapter is established by the committed evidence. Tiny synthetic training checks are software tests, not research models.

## Intended use

Academic experiments on context-dependent English meaning and an educational demonstration of pragmatic classification. The user supplies the category. The benchmark combines MUStARD text, Circa indirect answers and a small curated speech-act set; see the [dataset card](dataset_card.md).

## Limitations

- Lexical shortcuts can drive CPU predictions; accuracy does not prove pragmatic understanding.
- Context, culture and plausible ambiguity can change the intended meaning.
- Circa labels are collapsed, and speech-act test support is only 22 examples.
- Public training-data contamination and semantic near duplicates remain unresolved.
- Preset meanings and label-derived meanings are not independently reviewed generated explanations.
- Browser history counts are not evaluation scores; a changing plot does not indicate updated model weights.

Use the system for research exploration, not consequential judgments about a person's intentions, honesty or character. Report failure cases and uncertainty alongside aggregate scores. Model access and model-weight terms are independent of this repository's code license.

Measured scores are in [results](results.md); the research-completion boundary is in [completion status](completion-status.md).
