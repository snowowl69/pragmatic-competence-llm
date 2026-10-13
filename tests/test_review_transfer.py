import hashlib
import json
import pytest

from app.review import next_item, review_items, save_rating, rating_summary
from src.evaluation.interpretations import select_sample
from src.analysis.transfer import build_transfer_matrix, _compute_per_category_accuracy, save_transfer_matrix_csv


def fixture_review(root):
    path = root/'data/benchmark/v2/test.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'id':'x','pragmatic_category':'sarcasm','context':'Failed exam','utterance':'Great!'})+'\n')
    source = root/'results/interpretations/baseline.jsonl'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'item_id':'x','condition':'baseline','model':'test-model',
        'benchmark_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'explanation':'The speaker is dissatisfied.'})+'\n')


def test_review_is_blind_and_requires_actual_human_action(tmp_path):
    fixture_review(tmp_path)
    item = next_item('reviewer-a', tmp_path)
    assert 'test-model' not in str(item) and 'baseline' not in str(item)
    with pytest.raises(ValueError, match='independently'):
        save_rating('a',item[0],2,2,2,'',False,tmp_path)
    save_rating('a',item[0],2,2,2,'Evidence fits',True,tmp_path)
    assert next_item('a',tmp_path)[0] == ''
    save_rating('b',item[0],2,2,2,'Evidence fits',True,tmp_path)
    status, agreement = rating_summary(tmp_path)
    assert '1 rated by at least two' in status
    assert all(row[-1]=='Undefined' for row in agreement)


def test_stale_interpretations_not_rated(tmp_path):
    fixture_review(tmp_path)
    with (tmp_path/'data/benchmark/v2/test.jsonl').open('a') as f: f.write('\n')
    assert not review_items(tmp_path)


def test_stratified_sample_reproducible():
    rows=[{'id':str(i),'pragmatic_category':str(i%3)} for i in range(30)]
    assert select_sample(rows,2)==select_sample(list(reversed(rows)),2)
    assert len(select_sample(rows,2))==6


def prediction(item='x'):
    return {'item_id':item,'model':'fixture','model_id':'fixture','pragmatic_category':'sarcasm',
            'gold_label':'sincere','predicted_label':'sarcastic','correct':True}


def test_transfer_recomputes_and_missing_is_not_zero(tmp_path):
    row=prediction()
    score=_compute_per_category_accuracy([row],['sarcasm','implicature'])
    assert score=={'sarcasm':0,'implicature':None}
    folder=tmp_path/'post_ft/joint'; folder.mkdir(parents=True)
    (folder/'predictions.jsonl').write_text(json.dumps(row)+'\n')
    result=build_transfer_matrix(tmp_path)
    assert result['transfer_gains']['joint']['sarcasm'] is None
    save_transfer_matrix_csv(result,tmp_path/'matrix.csv')
    assert 'Not measured' in (tmp_path/'matrix.csv').read_text()


def test_transfer_rejects_duplicates_and_unpaired_runs(tmp_path):
    with pytest.raises(ValueError, match='repeated'):
        _compute_per_category_accuracy([prediction(),prediction()],['sarcasm'])
    for folder, item in [('baseline','x'),('post_ft/joint','y')]:
        path=tmp_path/folder; path.mkdir(parents=True)
        (path/'predictions.jsonl').write_text(json.dumps(prediction(item))+'\n')
    with pytest.raises(ValueError, match='test IDs'):
        build_transfer_matrix(tmp_path)


def test_navigation_routes():
    from app.demo import build_demo
    demo=build_demo()
    assert {'graphs','review'} <= {p[0] for p in demo.pages}
    graph_ids={i for i,b in demo.blocks.items() if getattr(b,'page',None)=='graphs'}
    assert graph_ids


def test_explanation_generation_resumes_without_loading_again(tmp_path, monkeypatch):
    from src.evaluation import interpretations, harness
    monkeypatch.setattr(interpretations, 'ROOT', tmp_path)
    path=tmp_path/'data/benchmark/v2/test.jsonl'; path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'id':'x','pragmatic_category':'sarcasm','context':'Failed exam','utterance':'Great!'})+'\n')
    loaded=[]
    monkeypatch.setattr(harness,'load_hf_model',lambda *args: (loaded.append(args), None))
    monkeypatch.setattr(harness,'generate_hf',lambda *args,**kw:'Test fixture explanation, not an experiment result.')
    output=interpretations.generate('fixture-model',per_category=1)
    row=json.loads(output.read_text())
    assert row['protocol']=='meaning-v1'
    assert 'gold' not in row['prompt'].lower()
    assert interpretations.generate('fixture-model',per_category=1)==output
    assert len(loaded)==1
    with pytest.raises(ValueError,match='different model'):
        interpretations.generate('different-model',per_category=1)
