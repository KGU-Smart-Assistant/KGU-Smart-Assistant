from app.services import map_service


def test_pick_best_match_strips_topic_particle_from_building_name() -> None:
    rows = [
        ("경기대학교 수원캠퍼스 제8강의동(육영관)", "8강의동", 37.3001, 127.0351),
    ]

    result = map_service._pick_best_match("8강의동은 어디야?", rows)

    assert result == ("경기대학교 수원캠퍼스 제8강의동(육영관)", 37.3001, 127.0351)
