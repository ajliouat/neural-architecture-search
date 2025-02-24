from models.architectures.candidate import CandidateArchitecture

def test_candidate():
    layers = [nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10)]
    candidate = CandidateArchitecture(layers)
    assert candidate is not None