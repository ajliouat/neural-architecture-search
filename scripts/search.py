import yaml
from src.algorithms.nas import NAS
from src.envs.search_env import SearchEnv

def search(config):
    # Initialize search environment
    env = SearchEnv(config["env"])

    # Initialize NAS algorithm
    nas = NAS(env, config["policy"])

    # Run search
    nas.search(config["search"])

    # Save best candidate
    torch.save(nas.best_candidate, "models/architectures/best_candidate.pth")