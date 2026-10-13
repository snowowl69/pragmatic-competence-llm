"""Read-only plots of real artifacts. Missing measurements are never zero scores."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('sarcasm', 'implicature', 'indirect_speech_act')


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def read_records(path):
    """A writer may be appending the last JSONL line during a refresh."""
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            continue
    return rows


def empty_plot(title, message):
    fig = Figure(figsize=(8, 3.4), layout='constrained')
    ax = fig.subplots(); ax.set_title(title)
    ax.text(.5, .5, message, ha='center', va='center', transform=ax.transAxes, wrap=True)
    ax.set_axis_off()
    return fig


def collect(root=ROOT):
    root = Path(root)
    manifest = read_json(root / 'data/benchmark/v2/manifest.json', {})
    tests = {r['id']: r for r in read_records(root / 'data/benchmark/v2/test.jsonl') if 'id' in r}
    groups = defaultdict(dict)
    # Group by identity, prompt and context; never pool incompatible experiments.
    files = [root / 'results/local/predictions.jsonl', *sorted((root / 'results/llm').rglob('*.jsonl'))]
    for path in files:
        for r in read_records(path):
            item = tests.get(r.get('item_id'))
            if not item or r.get('gold_label') != item['label'] or r.get('pragmatic_category') != item['pragmatic_category']:
                continue
            key = (path.relative_to(root).as_posix(), str(r.get('model', 'unknown')), str(r.get('setting', 'unknown')),
                   str(r.get('prompt_template', 'unknown')), str(r.get('context_mode', 'unknown')), str(r.get('run_index', 0)))
            groups[key][item['id']] = r
    rows = []
    for key, records in groups.items():
        for category in CATEGORIES:
            subset = [r for r in records.values() if r['pragmatic_category'] == category]
            if not subset:
                continue
            expected = sum(i['pragmatic_category'] == category for i in tests.values())
            from src.evaluation.metrics import compute_metrics
            from src.evaluation.prompts import get_valid_labels
            score = compute_metrics([r['gold_label'] for r in subset], [r.get('predicted_label', 'INVALID') for r in subset], get_valid_labels(category))
            rows.append({'key': key, 'category': category, 'count': len(subset), 'expected': expected, 'metrics': score})
    runs = []
    for path in sorted((root / 'experiments').glob('*/live_status.json')):
        state = read_json(path)
        if isinstance(state, dict):
            state['condition'] = path.parent.name
            state['history'] = read_json(path.parent / 'training_metrics.json', [])
            runs.append(state)
    checks = []
    try:
        from src.preprocessing.build_benchmark import audit_splits
        splits = {s: read_records(root / f'data/benchmark/v2/{s}.jsonl') for s in ('train', 'val', 'test')}
        hashes = manifest.get('sha256', {})
        assert all(splits.values()) and len(hashes) == 3
        assert all(hashlib.sha256((root / 'data/benchmark/v2' / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
        audit_splits(splits)
        valid = True
    except (OSError, ValueError, AssertionError, KeyError, TypeError):
        valid = False
    checks.append(['Benchmark and exact leakage checks', 'Verified' if valid else 'Missing or failed', 'Schema, split hashes, unique IDs, exact replies and dialogue groups. Near duplicates still need review.'])
    cpu = [r for r in rows if r['key'][0].startswith('results/local/')]
    complete_cpu = all(any(r['category'] == c and r['key'][4] == mode and r['count'] == r['expected'] for r in cpu) for c in CATEGORIES for mode in ('present', 'removed'))
    checks.append(['CPU baseline and context comparison', 'Verified' if complete_cpu and valid else 'Incomplete', 'Recalculated from saved predictions; not LLM results.'])
    llm = [r for r in rows if r['key'][0].startswith('results/llm/')]
    checks.append(['Full LLM baseline evaluation', 'Review artifacts' if llm else 'Missing', 'Coverage shown below. Model identity, benchmark provenance and equivalent settings require verification.'])
    adapters = [p for p in (root / 'experiments').glob('*/adapter/adapter_config.json') if any(p.parent.glob('adapter_model.*'))]
    checks.append(['QLoRA adapters and training evidence', 'Review artifacts' if adapters else 'Missing', f'{len(adapters)} adapter(s) found; configs alone do not establish completed training.'])
    checks.extend([
        ['LLM context ablation and cross-category transfer', 'Review artifacts' if llm else 'Missing', 'Requires paired complete runs on identical test IDs; CPU task classifiers are not a transfer experiment.'],
        ['Independent interpretation scoring', 'Manual review required', 'Need blinded human ratings, adjudication and reviewer agreement. Classification accuracy does not prove explanation quality.'],
        ['Source licences and annotation provenance', 'Manual review required', 'Circa licence-link discrepancy and curated-example permissions remain unresolved.'],
        ['Failure analysis and limitations', 'Partial', 'Confusion matrices available; reviewed LLM errors and explanation analysis still required.'],
        ['Gradio demonstration and live monitoring', 'Implemented', 'CPU inference and real artifact plots; genuine baseline/adapter demo requires both models.'],
        ['Report and diagrams', 'Manual review required', 'Five design diagrams are in the report; confirm the final report matches the current run artifacts.'],
    ])
    return {'rows': rows, 'runs': runs, 'checks': checks, 'manifest': manifest}


def dashboard(category='sarcasm', root=ROOT):
    data = collect(root)
    selected = [r for r in data['rows'] if r['category'] == category]
    fig = empty_plot('Held-out accuracy and macro-F1', 'No saved predictions for this category.')
    if selected:
        fig = Figure(figsize=(10, max(3.5, len(selected)*.6)), layout='constrained'); ax = fig.subplots()
        labels = []
        for i, r in enumerate(selected):
            key = r['key']; labels.append(f'{key[0]} | {key[4]} | {key[3]} | run {key[5]}\n{r["count"]}/{r["expected"]} evaluated')
            ax.barh(i-.18, r['metrics']['accuracy'], .34, label='Accuracy' if i == 0 else None, color='#3366aa')
            ax.barh(i+.18, r['metrics']['macro_f1'], .34, label='Macro-F1' if i == 0 else None, color='#df8737')
        ax.set_yticks(range(len(labels)), labels, fontsize=8); ax.set_xlim(0, 1); ax.legend(); ax.set_title(f'{category}: held-out results (partial coverage is provisional)')
    confusion = empty_plot('Confusion matrices', 'No saved predictions.')
    if selected:
        confusion = Figure(figsize=(7, max(4, 3.8*len(selected))), layout='constrained')
        axes = confusion.subplots(len(selected), 1, squeeze=False).flatten()
        for ax, r in zip(axes, selected):
            score = r['metrics']; matrix = score['confusion_matrix']; labels = score['label_names']
            ax.imshow(matrix, cmap='Blues'); ax.set_xticks(range(len(labels)), labels, rotation=20); ax.set_yticks(range(len(labels)), labels)
            for i, row in enumerate(matrix):
                for j, v in enumerate(row): ax.text(j, i, str(v), ha='center', va='center', color='black', bbox={'facecolor':'white','alpha':.65,'edgecolor':'none'})
            ax.set_title(f'{r["key"][0]} | {r["key"][4]} | {r["count"]}/{r["expected"]}', fontsize=9)
            ax.set_xlabel('Predicted'); ax.set_ylabel('Gold')
    training = empty_plot('Training loss', 'No training measurements yet. Start a real GPU training run to populate this plot.')
    series = []
    run_lines = []
    now = datetime.now(timezone.utc)
    for run in data['runs']:
        age = None
        try: age = (now-datetime.fromisoformat(run['updated_at'])).total_seconds()
        except (KeyError, ValueError, TypeError): pass
        status = run.get('status', 'unknown')
        if status in ('training', 'loading', 'saving') and (age is None or age > 120): status += ' — no recent update; process may be stopped or a step may be slow'
        run_lines.append(f"- **{run['condition']}**: {status}; step {run.get('step', 0)}/{run.get('max_steps', '?')}; updated {run.get('updated_at', 'unknown')}")
        for metric in ('loss', 'eval_loss'):
            points = [(x.get('step', 0), x[metric]) for x in run.get('history', []) if isinstance(x, dict) and isinstance(x.get(metric), (int, float)) and math.isfinite(x[metric])]
            if points: series.append((f'{run["condition"]} {metric}', points))
    if series:
        training = Figure(figsize=(9, 4), layout='constrained'); ax = training.subplots()
        for label, points in series: ax.plot(*zip(*points), marker='.', label=label)
        ax.set_xlabel('Optimizer step'); ax.set_ylabel('Loss'); ax.set_title('Actual logged training and validation loss'); ax.legend()
    status = f'Last read: {now:%Y-%m-%d %H:%M:%S UTC}. Refreshes every 5 seconds while this page is open.\n\n**The full research project is not yet evidenced as complete.**\n\n' + ('\n'.join(run_lines) or 'No active training telemetry found. A YAML configuration is not a training run.')
    dataset = empty_plot('Benchmark composition', 'No benchmark manifest available.')
    splits = data['manifest'].get('splits', {})
    if all(s in splits for s in ('train', 'val', 'test')):
        dataset = Figure(figsize=(8, 4), layout='constrained'); ax = dataset.subplots()
        for i, c in enumerate(CATEGORIES):
            ax.bar([x+(i-1)*.25 for x in range(3)], [splits[s].get('categories', {}).get(c, 0) for s in ('train', 'val', 'test')], width=.25, label=c)
        ax.set_xticks(range(3), ['Train', 'Validation', 'Test']); ax.set_ylabel('Examples'); ax.set_title('Saved benchmark composition'); ax.legend()
    transfer = empty_plot('LLM baseline and cross-category transfer', 'No complete LLM evaluation cells yet. Missing results are not zero.')
    cells = defaultdict(dict)
    for r in data['rows']:
        parts = r['key'][0].split('/')
        if parts[:2] != ['results', 'llm'] or r['key'][4] != 'present' or r['count'] != r['expected']:
            continue
        condition = 'baseline' if parts[2] == 'baseline' else parts[3] if parts[2] == 'post_ft' and len(parts)>4 else None
        if condition:
            identity = (condition, *r['key'][1:])
            cells[identity][r['category']] = r['metrics']['accuracy']
    if cells:
        import numpy as np
        transfer = Figure(figsize=(10, max(4, len(cells)*.5)), layout='constrained'); ax = transfer.subplots()
        matrix = [[values.get(c, float('nan')) for c in CATEGORIES] for values in cells.values()]
        im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='Blues', aspect='auto')
        for i, row in enumerate(matrix):
            for j, value in enumerate(row): ax.text(j, i, f'{value:.1%}' if math.isfinite(value) else 'Missing', ha='center', va='center', bbox={'facecolor':'white','alpha':.7,'edgecolor':'none'})
        ax.set_xticks(range(3), CATEGORIES); ax.set_yticks(range(len(cells)), [' | '.join(k) for k in cells], fontsize=8)
        ax.set_title('Full-coverage LLM test accuracy by training condition — compare matching settings only')
        transfer.colorbar(im, ax=ax, label='Accuracy')
    score_rows = [[r['key'][1], r['key'][0], r['category'], r['key'][4],
                   f"{r['count']}/{r['expected']}", round(r['metrics']['accuracy'], 4),
                   round(r['metrics']['macro_f1'], 4)] for r in data['rows']]
    class_rows = []
    for r in selected:
        for label, values in r['metrics']['classification_report'].items():
            if isinstance(values, dict):
                class_rows.append([r['key'][0], r['key'][4], label,
                                   round(values['precision'],4), round(values['recall'],4),
                                   round(values['f1-score'],4), values['support']])
    return status, fig, confusion, training, dataset, transfer, score_rows, class_rows


def session_plot(history):
    if not history: return empty_plot('This session’s predictions', 'Analyze an utterance to start a live prediction history.')
    counts = Counter(f'{r["category"]}: {r["label"]}' for r in history)
    fig = Figure(figsize=(8, 3.5), layout='constrained'); ax = fig.subplots()
    ax.barh(list(counts), list(counts.values()), color='#3366aa'); ax.set_xlabel('Predictions in the last 100 analyses'); ax.set_title('Live inputs — label counts, not held-out accuracy')
    return fig


def mount(gr):
    gr.Markdown('## Research graphs\nCharts read real saved predictions and training logs. A changing input does not change benchmark accuracy. Training in Colab must run this dashboard there, or sync its experiment/results files here.')
    category = gr.Dropdown(list(CATEGORIES), value='sarcasm', label='Chart category')
    refresh = gr.Button('Refresh research evidence')
    status = gr.Markdown()
    with gr.Tabs():
        with gr.Tab('Test results'):
            results = gr.Plot()
            scores = gr.Dataframe(headers=['Model','Artifact','Category','Context','Coverage','Accuracy','Macro-F1'], interactive=False, wrap=True)
        with gr.Tab('Per-label scores'):
            classes = gr.Dataframe(headers=['Artifact','Context','Label','Precision','Recall','F1','Support'], interactive=False, wrap=True)
        with gr.Tab('Confusion matrices'): confusion = gr.Plot()
        with gr.Tab('Live training'): training = gr.Plot()
        with gr.Tab('Dataset'): dataset = gr.Plot()
        with gr.Tab('LLM transfer'): transfer = gr.Plot()
    targets = [status, results, confusion, training, dataset, transfer, scores, classes]
    refresh.click(dashboard, category, targets)
    category.change(dashboard, category, targets)
    timer = gr.Timer(5)
    timer.tick(dashboard, category, targets, show_progress='hidden', trigger_mode='always_last')
    return category, targets
