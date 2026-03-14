from pathlib import Path


def test_job_config_store_uses_atomic_persistence_primitives() -> None:
    source = Path('backend/services/job_config_store.py').read_text(encoding='utf-8')
    assert 'import fcntl' in source
    assert 'import tempfile' in source
    assert 'def _exclusive_lock' in source
    assert 'def _write_atomic' in source
    assert 'os.replace(' in source


def test_threads_api_exposes_reply_linkage_and_cache_layer() -> None:
    source = Path('backend/api/main.py').read_text(encoding='utf-8')
    assert 'def _cached_thread_rows' in source
    assert 'def _runs_snapshot_signature' in source
    assert 'reply_job_id' in source
    assert 'reply_job_name' in source
    assert 'reply_personas' in source


def test_llm_service_no_longer_uses_demo_fallback_responses() -> None:
    source = Path('backend/services/llm_service.py').read_text(encoding='utf-8')
    assert 'class LLMUnavailableError' in source
    assert 'class LLMRequestError' in source
    assert 'OpenRouter is not configured. Set OPENROUTER_API_KEY.' in source
    assert '_get_demo_response' not in source
