from pathlib import Path


def test_architecture_doc_matches_current_runtime_and_ui_behavior() -> None:
    source = Path('docs/ARCHITECTURE.md').read_text(encoding='utf-8')

    assert '`backend/reddrop`' in source
    assert '`./reddrop`: single CLI entrypoint.' not in source

    assert 'settings.yaml' in source
    assert 'load/save `.env` runtime settings' not in source

    assert 'invoke in-process command handlers' in source
    assert 'spawn CLI `search` subprocesses' not in source

    assert 'default `30s`' in source
    assert 'with default `off`' not in source
