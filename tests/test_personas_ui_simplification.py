from pathlib import Path


def test_personas_form_uses_single_combined_objective_field() -> None:
    source = Path('frontend/src/features/personas/index.tsx').read_text(encoding='utf-8')
    assert "Label htmlFor={`${prefix}-persona-objective`}>Objective, tone & style</Label>" in source
    assert "## Objectives" in source
    assert "......" in source
    assert "# Tone & Style" in source
    assert "....." in source
    assert "Label htmlFor={`${prefix}-persona-tone-style`}>Tone & style</Label>" not in source
    assert "Label htmlFor={`${prefix}-persona-description`}>Description</Label>" not in source
    assert "max-h-[80vh] space-y-4 overflow-y-auto pr-1" in source
