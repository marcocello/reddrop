from __future__ import annotations

import json
from pathlib import Path

from backend.cli import reddrop as cli
from backend.services.job_config_store import JobConfigStore


def test_cli_add_accepts_max_runtime_minutes() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([
        'add',
        '--name',
        'growth',
        '--topic',
        'AI launch strategy',
        '--max-runtime-minutes',
        '30',
    ])

    assert args.max_runtime_minutes == 30


def test_job_store_converts_legacy_hours_to_minutes(tmp_path: Path) -> None:
    config_path = tmp_path / 'jobs_config.json'
    config_path.write_text(
        json.dumps(
            {
                'jobs': [
                    {
                        'id': 'b3fd3495-d2fe-4e38-b05f-376c93be6b83',
                        'name': 'growth',
                        'topic': 'AI launch strategy',
                        'max_runtime_hours': 2,
                    }
                ]
            }
        ),
        encoding='utf-8',
    )

    store = JobConfigStore(config_path)
    jobs = store.list_jobs()

    assert len(jobs) == 1
    assert jobs[0].max_runtime_minutes == 120
