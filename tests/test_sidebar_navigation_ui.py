from pathlib import Path


def test_sidebar_is_split_into_workflows_results_and_configuration_groups() -> None:
    source = Path('frontend/src/components/layout/data/sidebar-data.ts').read_text(encoding='utf-8')
    workflows_start = source.find("title: 'Workflows'")
    results_start = source.find("title: 'Results'")
    configuration_start = source.find("title: 'Configuration'")

    assert workflows_start != -1
    assert results_start != -1
    assert configuration_start != -1
    assert workflows_start < results_start < configuration_start

    workflows_section = source[workflows_start:results_start]
    results_section = source[results_start:configuration_start]
    configuration_section = source[configuration_start:]

    assert "title: 'Personas'" in workflows_section
    assert "title: 'Search Queries'" in workflows_section
    assert "title: 'Jobs'" in workflows_section
    assert workflows_section.index("title: 'Personas'") < workflows_section.index("title: 'Search Queries'") < workflows_section.index("title: 'Jobs'")
    assert "title: 'Conversations'" not in workflows_section

    assert "title: 'Conversations'" in results_section
    assert "title: 'Jobs'" not in results_section
    assert "title: 'Personas'" not in results_section

    assert "title: 'Auths'" in configuration_section
    assert "title: 'Settings'" in configuration_section
    assert "title: 'Personas'" not in configuration_section


def test_top_left_sidebar_has_no_teams_context_menu() -> None:
    source = Path('frontend/src/components/layout/team-switcher.tsx').read_text(encoding='utf-8')
    assert 'Teams' not in source
    assert 'Add team' not in source
    assert 'DropdownMenuContent' not in source
