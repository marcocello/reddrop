from pathlib import Path


def test_sidebar_is_split_into_workflows_and_configuration_groups() -> None:
    source = Path('frontend/src/components/layout/data/sidebar-data.ts').read_text(encoding='utf-8')
    workflows_start = source.find("title: 'Workflows'")
    configuration_start = source.find("title: 'Configuration'")

    assert workflows_start != -1
    assert configuration_start != -1
    assert workflows_start < configuration_start

    workflows_section = source[workflows_start:configuration_start]
    configuration_section = source[configuration_start:]

    assert "title: 'Jobs'" in workflows_section
    assert "title: 'Threads'" in workflows_section
    assert "title: 'Personas'" in workflows_section
    assert "title: 'Auths'" not in workflows_section
    assert "title: 'Settings'" not in workflows_section

    assert "title: 'Auths'" in configuration_section
    assert "title: 'Settings'" in configuration_section
    assert "title: 'Personas'" not in configuration_section
