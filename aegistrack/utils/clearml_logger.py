from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional


def resolve_clearml_dataset(dataset_id: str, fallback_data: str = "") -> str:
    if not dataset_id:
        return fallback_data
    cached_path = Path.home() / '.clearml' / 'cache' / 'storage_manager' / 'datasets' / f'ds_{dataset_id}'
    if cached_path.exists():
        print(f'ClearML dataset resolved: id={dataset_id} path={cached_path}', flush=True)
        return str(cached_path)
    try:
        from clearml import Dataset
    except Exception as exc:
        if fallback_data:
            print(f'WARNING: ClearML dataset is disabled: {exc}', file=sys.stderr)
            return fallback_data
        raise RuntimeError(f'ClearML is required to resolve dataset id {dataset_id}: {exc}') from exc

    dataset = Dataset.get(dataset_id=str(dataset_id))
    local_path = dataset.get_local_copy()
    print(f'ClearML dataset resolved: id={dataset_id} path={local_path}', flush=True)
    return str(local_path)


class ClearMLLogger:
    def __init__(
        self,
        enabled: bool,
        project_name: str,
        task_name: str,
        queue_name: Optional[str] = None,
        remote: bool = False,
        args: Optional[Any] = None,
    ):
        self.enabled = False
        self.task = None
        self.logger = None
        if not enabled:
            return
        try:
            from clearml import Task
        except Exception as exc:
            print(f'WARNING: ClearML is disabled: {exc}', file=sys.stderr)
            return

        self.task = Task.init(project_name=project_name, task_name=task_name)
        if args is not None:
            self.task.connect(args if isinstance(args, dict) else vars(args))
        if remote:
            self.task.execute_remotely(queue_name=queue_name, exit_process=True)
        self.logger = self.task.get_logger()
        self.enabled = True

    def report(self, name: str, value: float, iteration: int):
        if not self.enabled or self.logger is None:
            return
        if '/' in name:
            title, series = name.split('/', 1)
        else:
            title, series = 'metrics', name
        self.logger.report_scalar(title=title, series=series, value=float(value), iteration=int(iteration))

    def report_many(self, metrics: dict, iteration: int):
        if not self.enabled:
            return
        for name, value in metrics.items():
            if value is not None:
                self.report(name, value, iteration)

    def upload_artifact(self, name: str, path: str):
        if not self.enabled or self.task is None:
            return
        artifact_path = Path(path)
        if not artifact_path.exists():
            print(f'WARNING: ClearML artifact not found: {artifact_path}', file=sys.stderr)
            return
        self.task.upload_artifact(name, str(artifact_path))

    def close(self):
        if not self.enabled or self.task is None:
            return
        try:
            self.task.close()
        except AttributeError:
            pass
