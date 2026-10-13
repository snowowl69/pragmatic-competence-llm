"""Blinded human ratings of real generated interpretations; never auto-rate."""
import hashlib
import json
import math
import threading
from collections import defaultdict
from datetime import datetime, timezone

from app.monitor import ROOT, read_records

LOCK = threading.Lock()
DIMENSIONS = ('meaning', 'context', 'grounding')


def review_items(root=ROOT):
    test_path = root / 'data/benchmark/v2/test.jsonl'
    if not test_path.exists(): return {}
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    test = {r['id']: r for r in read_records(test_path)}
    items = {}
    for path in sorted((root / 'results/interpretations').glob('*.jsonl')):
        for row in read_records(path):
            if row.get('benchmark_sha256') != digest or row.get('item_id') not in test: continue
            if not isinstance(row.get('explanation'), str) or not row['explanation'].strip(): continue
            if not row.get('condition') or not row.get('model'): continue
            identity = json.dumps(row, sort_keys=True, ensure_ascii=False)
            sample_id = hashlib.sha256(identity.encode()).hexdigest()
            items[sample_id] = {**row, 'item': test[row['item_id']]}
    return items


def latest_ratings(root=ROOT):
    return {(r['sample_id'], r['reviewer']): r for r in read_records(root/'results/human/ratings.jsonl')
            if all(k in r for k in ('sample_id', 'reviewer', *DIMENSIONS))}


def next_item(reviewer, root=ROOT):
    reviewer = str(reviewer).strip()
    if not reviewer: return '', 'Enter your reviewer ID to begin.', '', '', ''
    items = review_items(root); ratings = latest_ratings(root)
    # Reviewer-dependent order; neither model identity nor other ratings shown.
    order = sorted(items, key=lambda key: hashlib.sha256((reviewer+key).encode()).hexdigest())
    for key in order:
        if (key, reviewer) not in ratings:
            row = items[key]; item = row['item']
            return key, f"Category: {item['pragmatic_category']}", item['context'], item['utterance'], row['explanation']
    return '', ('All available items rated by this reviewer.' if items else 'No generated interpretations available. Run the GPU interpretation experiment first.'), '', '', ''


def save_rating(reviewer, sample_id, meaning, context, grounding, notes, attested, root=ROOT):
    reviewer = str(reviewer).strip()
    if not reviewer or len(reviewer) > 80: raise ValueError('Use a reviewer ID of 1–80 characters.')
    if not attested: raise ValueError('Confirm that you independently reviewed this example.')
    if sample_id not in review_items(root): raise ValueError('No current review item is selected.')
    scores = [meaning, context, grounding]
    if any(v not in (0, 1, 2) for v in scores): raise ValueError('Select all three rubric scores.')
    row = dict(zip(DIMENSIONS, map(int, scores)))
    row.update(reviewer=reviewer, sample_id=sample_id, notes=str(notes)[:4000],
               independent_attestation=True, updated_at=datetime.now(timezone.utc).isoformat())
    path = root/'results/human/ratings.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
    with LOCK, path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    return 'Rating saved. Use Next item to continue.'


def rating_summary(root=ROOT):
    from sklearn.metrics import cohen_kappa_score
    items = review_items(root)
    ratings = [r for r in latest_ratings(root).values() if r['sample_id'] in items]
    grouped = defaultdict(list)
    for r in ratings: grouped[r['sample_id']].append(r)
    pairs = defaultdict(list)
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda r:r['reviewer'])
        for i, a in enumerate(rows):
            for b in rows[i+1:]: pairs[(a['reviewer'], b['reviewer'])].append((a,b))
    table = []
    for reviewers, paired in pairs.items():
        for metric in DIMENSIONS:
            a, b = [x[metric] for x,y in paired], [y[metric] for x,y in paired]
            kappa = cohen_kappa_score(a,b,labels=[0,1,2],weights='quadratic') if len(set(a+b))>1 else float('nan')
            table.append([' / '.join(reviewers), metric, len(paired), sum(x==y for x,y in zip(a,b))/len(a), round(kappa,4) if math.isfinite(kappa) else 'Undefined'])
    status = (f'{len(items)} generated interpretations; {len(grouped)} rated; '
              f'{sum(len(r)>=2 for r in grouped.values())} rated by at least two reviewer IDs. '
              'IDs and attestations do not independently verify reviewer identity. Disagreements still require adjudication.')
    return status, table


def mount_review(gr):
    gr.Markdown('# Human interpretation review\nModel identity and other reviewers’ scores are hidden. Use a distinct pseudonymous ID for each real reviewer. Read the context and score the generated meaning independently. This workflow does not manufacture human ratings.')
    gr.Markdown('**Rubric (0–2 for each dimension)**\n- Meaning: 0 incorrect, 1 partly correct or misses ambiguity, 2 correct and appropriately qualified.\n- Context: 0 ignores or contradicts context, 1 partially uses it, 2 uses the relevant contextual evidence.\n- Grounding: 0 substantial invented facts, 1 minor unsupported assumptions, 2 no unsupported assumptions.')
    reviewer = gr.Textbox(label='Reviewer ID')
    current = gr.State('')
    next_button = gr.Button('Next item')
    status = gr.Textbox(label='Review status', interactive=False)
    context = gr.Textbox(label='Context', interactive=False)
    utterance = gr.Textbox(label='Utterance', interactive=False)
    explanation = gr.Textbox(label='Generated interpretation', interactive=False)
    scores = [gr.Radio([0,1,2], label=name.capitalize()) for name in DIMENSIONS]
    notes = gr.Textbox(label='Reason for scores and any ambiguity')
    attested = gr.Checkbox(label='I independently reviewed this example without seeing another reviewer’s ratings')
    save = gr.Button('Save human rating')
    receipt = gr.Textbox(label='Save status', interactive=False)
    def advance(identity): return (*next_item(identity), None, None, None, '', False, '')
    next_button.click(advance, reviewer, [current,status,context,utterance,explanation,*scores,notes,attested,receipt])
    def save_checked(*values):
        try: return save_rating(*values)
        except ValueError as error: return str(error)
    save.click(save_checked, [reviewer,current,*scores,notes,attested], receipt)
    gr.Markdown('## Review coverage and agreement\nRefresh after saving ratings. Quadratic weighted Cohen’s kappa is undefined for constant ratings; it is not replaced with a perfect score.')
    summary = gr.Textbox(label='Coverage', interactive=False)
    table = gr.Dataframe(headers=['Reviewer pair','Dimension','Paired items','Exact agreement','Weighted kappa'], interactive=False)
    gr.Button('Refresh review coverage').click(rating_summary, outputs=[summary,table])
