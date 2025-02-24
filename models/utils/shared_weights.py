import torch

class SharedWeights:
    def __init__(self, model):
        self.model = model
        self.shared_weights = model.state_dict()

    def update(self, new_weights):
        self.shared_weights.update(new_weights)

    def get(self):
        return self.shared_weights