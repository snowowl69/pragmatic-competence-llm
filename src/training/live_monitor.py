"""Trainer callback that publishes atomic, inspectable live evidence."""
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from transformers import TrainerCallback


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class LiveMonitor(TrainerCallback):
    def __init__(self, directory, config):
        self.directory = Path(directory)
        if (self.directory / 'live_status.json').exists():
            archive = self.directory / 'monitor_archive' / uuid4().hex
            archive.mkdir(parents=True, exist_ok=True)
            for name in ('live_status.json', 'training_metrics.json'):
                if (self.directory / name).exists(): shutil.copy2(self.directory / name, archive / name)
        self.data = {'run_id': uuid4().hex, 'model': config['base_model'], 'config': config,
                     'python': platform.python_version(), 'status': 'loading', 'step': 0}
        atomic_json(self.directory / 'training_metrics.json', [])
        self.publish('loading')

    def publish(self, status=None, **fields):
        self.data.update(fields)
        if status: self.data['status'] = status
        self.data['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.directory / 'live_status.json', self.data)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero: return
        import torch
        self.publish('training', max_steps=state.max_steps, torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                     gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero: self.publish(step=state.global_step, epoch=state.epoch)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero: return
        atomic_json(self.directory / 'training_metrics.json', state.log_history)
        self.publish(step=state.global_step, epoch=state.epoch)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero: self.publish('saving', step=state.global_step)
