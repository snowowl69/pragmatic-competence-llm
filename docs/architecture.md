# Architecture

The implemented software separates benchmark construction, measured CPU inference, GPU experiment execution and human scoring. The diagrams describe supported workflows, not proof that all experiments have run.

## Overall workflow

```mermaid
flowchart LR
  Sources[Source records] --> Prepare[Normalize and group]
  Prepare --> Benchmark[Frozen v2 splits]
  Benchmark --> CPU[CPU reference]
  Benchmark --> Base[Llama baseline evaluation]
  Benchmark --> Train[QLoRA training]
  Train --> Adapt[Adapter evaluation]
  CPU --> Saved[Saved predictions and metrics]
  Base --> Saved
  Adapt --> Saved
  Saved --> Graphs[Graphs and matrices]
```

## Preprocessing

```mermaid
flowchart TD
  Raw[MUStARD text, Circa and curated examples] --> Normalize[Normalize fields and labels]
  Normalize --> Groups[Group dialogues and questions]
  Groups --> Split[Create train, validation and test splits]
  Split --> Dedup[Exclude exact reply collisions]
  Dedup --> Audit[Schema and exact leakage audit]
  Audit --> Manifest[Save splits and SHA256 manifest]
  Manifest --> Instructions[Derive training instructions from train and validation]
```

Near duplicates and pretraining contamination are not eliminated by exact-match checks.

## Application

```mermaid
flowchart LR
  Home[Analyze page] --> Input[Context, utterance and supplied category]
  Input --> Predict[Configured inference backend]
  Predict --> Output[Labels and available reference information]
  Home --> GraphPage[Graphs page]
  Files[Saved results and training telemetry] --> GraphPage
  Home --> ReviewPage[Human review page]
  Interpretations[Generated interpretation sample] --> ReviewPage
  ReviewPage --> Ratings[Local ratings and agreement]
```

## QLoRA

```mermaid
flowchart LR
  Base[Base model] --> Quant[4 bit NF4 frozen weights]
  Examples[Training instructions] --> Forward[Forward pass]
  Quant --> Forward
  LoRA[Trainable attention adapters] --> Forward
  Forward --> Loss[Supervised completion loss]
  Loss --> Update[Update adapter weights]
  Update --> LoRA
  LoRA --> Saved[Saved adapter]
  Loss --> Logs[Training and validation logs]
```

The joint configuration uses rank 16, alpha 32, dropout 0.05, learning rate 0.00015, three epochs, sequence length 512 and effective batch size eight on one device. These are configured settings, not completed-run observations.

## Experiments

```mermaid
flowchart TD
  Test[Same held out benchmark] --> Baseline[Unadapted baseline]
  Test --> Joint[Joint adapter]
  Test --> Single[Three category adapters]
  Baseline --> Compare[Compare equivalent settings and test IDs]
  Joint --> Compare
  Single --> Compare
  Compare --> Metrics[Accuracy, macro F1 and confusion matrices]
  Compare --> Transfer[Cross category transfer]
  Compare --> Context[Context present versus removed]
  Explanation[Separate explanation generation] --> Humans[Blinded independent ratings]
  Humans --> Agreement[Coverage, agreement and adjudication]
```
