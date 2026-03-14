from pathlib import Path


def test_architecture_doc_coherence_claims_are_updated() -> None:
    source = Path('docs/ARCHITECTURE.md').read_text(encoding='utf-8')

    assert '`backend/reddrop`' in source
    assert 'settings.yaml' in source
    assert 'invoke in-process command handlers' in source
    assert 'default `30s`' in source
