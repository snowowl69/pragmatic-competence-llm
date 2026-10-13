"""Generate a separate, reproducible explanation sample for human review.

Does not reuse label-only classification outputs as explanations.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def select_sample(items, per_category=100, seed=42):
    if per_category < 1:
        raise ValueError('Sample size must be positive')
    selected = []
    for category in sorted({r['pragmatic_category'] for r in items}):
        group = sorted((r for r in items if r['pragmatic_category'] == category), key=lambda r: r['id'])
        random.Random(seed).shuffle(group)
        selected.extend(group[:per_category])
    return selected


def generate(model_name, adapter=None, per_category=100):
    from src.evaluation.harness import load_hf_model, generate_hf
    test = ROOT / 'data/benchmark/v2/test.jsonl'
    digest = hashlib.sha256(test.read_bytes()).hexdigest()
    items = select_sample([json.loads(line) for line in test.read_text(encoding='utf-8').splitlines()], per_category)
    condition = Path(adapter).parent.name if adapter else 'baseline'
    output = ROOT / 'results/interpretations' / f'{condition}.jsonl'
    adapter_hash = None
    if adapter:
        hasher = hashlib.sha256()
        for p in sorted(Path(adapter).glob('adapter*')):
            if p.is_file():
                with p.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''): hasher.update(chunk)
        adapter_hash = hasher.hexdigest()
    existing = []
    if output.exists():
        existing = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines() if line.strip()]
        expected_ids = {r['id'] for r in items}
        if len({r['item_id'] for r in existing}) != len(existing):
            raise ValueError('Repeated interpretation item IDs')
        for row in existing:
            if (row.get('model'), row.get('benchmark_sha256'), row.get('adapter_sha256'), row.get('protocol'), row.get('condition')) != (model_name, digest, adapter_hash, 'meaning-v1', condition) or row['item_id'] not in expected_ids:
                raise ValueError('Existing interpretations use different model, adapter, benchmark or protocol; archive them before a new run.')
    completed = {r['item_id'] for r in existing}
    if len(completed) == len(items): return output
    model, tokenizer = load_hf_model(model_name, adapter)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf-8') as stream:
        for item in items:
            if item['id'] in completed: continue
            prompt = ('Explain what the speaker most likely means in this context in two sentences. '
                      'Ground the explanation in the given context; acknowledge ambiguity and do not invent facts.\n'
                      f"Category: {item['pragmatic_category']}\nContext: {item['context']}\nUtterance: {item['utterance']}")
            explanation = generate_hf(model, tokenizer, prompt, max_new_tokens=160)
            row = {'item_id': item['id'], 'condition': condition, 'model': model_name,
                   'benchmark_sha256': digest, 'adapter_sha256': adapter_hash,
                   'prompt': prompt, 'protocol': 'meaning-v1', 'seed': 42,
                   'max_new_tokens': 160, 'decoding': 'greedy', 'explanation': explanation}
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--adapter')
    parser.add_argument('--per-category', type=int, default=100)
    args = parser.parse_args()
    print(generate(args.model, args.adapter, args.per_category))
