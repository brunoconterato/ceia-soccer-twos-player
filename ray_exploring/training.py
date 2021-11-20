from ray import tune

def objective(x, a, b):
    return x


class Trainable(tune.Trainable):
    def setup(self, config):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.a = config["a"]
        self.b = config["b"]

    def step(self):  # This is called iteratively.
        score = objective(self.x, self.a, self.b)
        self.x += 1
        print(self.x)
        return {"score": score}

analysis = tune.run(
    Trainable,
    stop={"training_iteration": 20},
    config={
        "a": 2,
        "b": 4
    })

print('best config: ', analysis.get_best_config(metric="score", mode="max"))