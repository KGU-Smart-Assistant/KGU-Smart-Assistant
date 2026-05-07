from pathlib import Path

from app.crawlers.parsing.parser_registry import load_registry


def test_load_registry_reads_yaml_templates() -> None:
    registry = load_registry(Path("app/crawlers/parsing/templates"))

    names = [entry["name"] for entry in registry]

    assert "open_major_notice_detail" in names
    assert "artificial_intelligence_notice_detail" in names
    assert "kyonggi_notice_detail" in names
    assert "kyonggi_faq_list" in names
    assert "kyonggi_schedule" in names
    assert "generic_markdown" in names


def test_load_registry_normalizes_patterns_to_tuples() -> None:
    templates_dir = Path(".tmp/parser-registry-test")
    templates_dir.mkdir(parents=True, exist_ok=True)
    template_path = templates_dir / "custom.yaml"
    try:
        template_path.write_text(
            """
registry:
  - name: custom_notice
    priority: 5
    url_patterns:
      - "notice"
    categories:
      - "notice"
    parser: notice_detail
""".strip(),
            encoding="utf-8",
        )

        registry = load_registry(templates_dir)

        assert registry == [
            {
                "name": "custom_notice",
                "url_patterns": ("notice",),
                "categories": ("notice",),
                "parser": "notice_detail",
                "priority": 5,
                "options": {},
            }
        ]
    finally:
        template_path.unlink(missing_ok=True)
        templates_dir.rmdir()


def test_load_registry_sorts_by_priority() -> None:
    templates_dir = Path(".tmp/parser-registry-priority-test")
    templates_dir.mkdir(parents=True, exist_ok=True)
    first = templates_dir / "b.yaml"
    second = templates_dir / "a.yaml"
    try:
        first.write_text(
            """
registry:
  - name: later_parser
    priority: 50
    parser: generic_markdown
""".strip(),
            encoding="utf-8",
        )
        second.write_text(
            """
registry:
  - name: earlier_parser
    priority: 10
    parser: notice_detail
""".strip(),
            encoding="utf-8",
        )

        registry = load_registry(templates_dir)

        assert [entry["name"] for entry in registry] == ["earlier_parser", "later_parser"]
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)
        templates_dir.rmdir()


def test_site_specific_templates_rank_ahead_of_generic_kyonggi_entries() -> None:
    registry = load_registry(Path("app/crawlers/parsing/templates"))
    names = [entry["name"] for entry in registry]

    assert names.index("open_major_notice_detail") < names.index("kyonggi_notice_detail")
    assert names.index("artificial_intelligence_faq") < names.index("kyonggi_faq_list")
    assert names.index("social_college_schedule") < names.index("kyonggi_schedule")


def test_site_specific_template_options_are_loaded() -> None:
    registry = load_registry(Path("app/crawlers/parsing/templates"))
    tourism_notice = next(
        entry for entry in registry if entry["name"] == "tourism_culture_college_notice_detail"
    )

    assert tourism_notice["options"] == {"blocked_title_prefixes": ["* 경기대학교"]}
