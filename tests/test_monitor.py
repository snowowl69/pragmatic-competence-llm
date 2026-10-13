import json
from pathlib import Path
from types import SimpleNamespace

from app.monitor import collect, dashboard, read_records, session_plot


def test_missing_evidence_is_not_zero_or_complete(tmp_path):
    state = collect(tmp_path)
    assert not state['rows']
    assert state['checks'][0][1] == 'Missing or failed'
    assert state['checks'][2][1] == 'Missing'
    output = dashboard(root=tmp_path)
    assert 'not yet' in output[0]
    assert len(output) == 8


def test_partial_jsonl_is_safe(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_text('{"item_id":"a"}\n{"item_', encoding='utf-8')
    assert read_records(path) == [{'item_id':'a'}]


def test_real_artifacts_have_expected_counts_and_scores():
    root = Path(__file__).resolve().parents[1]
    state = collect(root)
    cpu = [r for r in state['rows'] if r['key'][0].startswith('results/local/')]
    assert len(cpu) == 6
    sarcasm = next(r for r in cpu if r['category']=='sarcasm' and r['key'][4]=='present')
    assert sarcasm['count'] == sarcasm['expected'] == 104
    assert sarcasm['metrics']['accuracy'] == .625
    assert state['checks'][0][1] == 'Verified'


def test_sessions_do_not_share_history():
    assert 'not held-out' in session_plot([{'category':'sarcasm','label':'sincere'}]).axes[0].get_title()
    assert not session_plot([]).axes[0].patches


def test_live_callback_publishes_logs_and_failure(tmp_path):
    import pytest
    pytest.importorskip('transformers', reason='Training callback requires optional training dependencies')
    from src.training.live_monitor import LiveMonitor
    monitor = LiveMonitor(tmp_path, {'base_model':'test-fixture'})
    state = SimpleNamespace(is_world_process_zero=True, global_step=2, epoch=.5,
                            log_history=[{'step':2,'loss':1.25}])
    monitor.on_log(None, state, None)
    assert json.loads((tmp_path/'training_metrics.json').read_text())[0]['loss'] == 1.25
    monitor.publish('failed', error_type='RuntimeError')
    assert json.loads((tmp_path/'live_status.json').read_text())['status'] == 'failed'
