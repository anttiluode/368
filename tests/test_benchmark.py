import math
from benchmark import evaluate_one, SwitchingWorld, WorldConfig, ACTIONS


def test_prediction_is_pre_outcome_and_deterministic():
    a=evaluate_one(1000,'base','growing')
    b=evaluate_one(1000,'base','growing')
    assert a['mse'] == b['mse']
    assert a['births'] == b['births']


def test_prospective_growth_rejects_noise_burst_birth():
    r=evaluate_one(1000,'noise_burst','growing')
    attacker=evaluate_one(1000,'noise_burst','error_only')
    assert r['births'] == 0
    assert attacker['births'] >= 1


def test_switching_world_can_earn_structure():
    r=evaluate_one(1000,'base','growing')
    single=evaluate_one(1000,'base','single')
    assert r['births'] >= 1
    assert r['mse'] < single['mse']


def test_missing_feature_not_solved_by_more_linear_experts():
    grow=evaluate_one(1000,'quadratic_stationary','growing')
    rich=evaluate_one(1000,'quadratic_stationary','single_quadratic')
    assert grow['births'] == 0
    assert rich['mse'] < 0.5*grow['mse']


def test_world_target_is_independent_of_learner():
    w=SwitchingWorld(7,WorldConfig(),family='stationary_005')
    u=ACTIONS[0]
    y0=w.target(10,u,0)
    y1=w.target(10,u,0)
    assert math.isclose(y0,y1)
