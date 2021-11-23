from ray import tune
import torch
import os


class MultiAgentTrainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.config = config

    def step(self):  # This is called iteratively.
        score = 0.0
        return {"score": score}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))