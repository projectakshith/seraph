from seraph.evaluation.repo import aggregate, is_relevant, score_case


def test_repo_eval_matches_path_and_optional_symbol_labels():
    expected = [{"path": "src/search.py", "symbols": ["search_code"]}]
    assert is_relevant({"path": "src/search.py", "symbol": "search_code"}, expected)
    assert not is_relevant({"path": "src/search.py", "symbol": "other"}, expected)
    assert is_relevant({"path": "README.md", "symbol": None}, [{"path": "README.md"}])


def test_repo_eval_scores_rank_and_noise():
    expected = [{"path": "src/search.py", "symbols": ["search_code"]}]
    rows = [
        {"path": "tests/test_search.py", "symbol": "test_search"},
        {"path": "src/search.py", "symbol": "search_code"},
        {"path": "README.md", "symbol": None},
    ]
    metrics = score_case(rows, expected, k=3)

    assert metrics["hit@1"] == 0
    assert metrics["hit@3"] == 1
    assert metrics["mrr"] == 0.5
    assert metrics["noise@3"] == 2 / 3
    assert aggregate({"q": metrics})["mrr"] == 0.5


def test_repo_eval_scores_empty_results_as_misses():
    metrics = score_case([], [{"path": "src/search.py"}], k=5)
    assert metrics["hit@1"] == metrics["hit@5"] == metrics["mrr"] == 0
    assert metrics["noise@5"] == 0
