from scripts.validate_migration import gate


def good():
    return {'reference_seconds': 1082, 'object_micro_f1': 1.0, 'people_exact': 1.0,
            'people_within_one': 1.0, 'missing_seconds': 0, 'extra_seconds': 0,
            'reference_errors': 0, 'candidate_errors': 0}


def test_migration_rejects_degradation_even_above_grader_floor():
    result = good()
    assert gate(result)
    result['people_exact'] = .98
    assert not gate(result)
    result = good()
    result['object_micro_f1'] = .98
    assert not gate(result)


def test_migration_requires_complete_error_free_coverage():
    for key in ('missing_seconds', 'extra_seconds', 'reference_errors', 'candidate_errors'):
        result = good()
        result[key] = 1
        assert not gate(result)
