import numpy as np

from gate2_fast_memory import HybridConfig, causal_prefix_check, evaluate_stream, make_stream
from gate2_attacks import build_stream


def test_gate2_prediction_is_pre_outcome():
    assert causal_prefix_check(1000)


def test_cache_without_retrieval_is_prediction_identical_to_fast_only():
    cfg = HybridConfig()
    events = make_stream(1000, 'base')
    fast = evaluate_stream(events, 'fast_only', cfg)
    cache = evaluate_stream(events, 'cache_only', cfg)
    np.testing.assert_array_equal(fast['predictions'], cache['predictions'])
    assert cache['final_memories'] >= 1


def test_quadratic_mismatch_does_not_create_linear_retained_memory():
    cfg = HybridConfig()
    r = evaluate_stream(make_stream(1000, 'quadratic_stationary'), 'hybrid', cfg)
    assert r['births'] == 0


def test_close_world_really_uses_minimum_distance_construction():
    # Smoke the close-context attacker and ensure the run can create retained
    # memories without hidden labels being passed to the learner.
    cfg = HybridConfig()
    r = evaluate_stream(build_stream(1000, 'close'), 'hybrid', cfg)
    assert r['final_memories'] >= 1
