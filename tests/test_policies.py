from models.policies.rl_policy import RLPolicy

def test_policy():
    policy = RLPolicy(10)
    assert policy is not None