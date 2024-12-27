import yaml
import os

class ParamLoader:
    def __init__(self):
        self.param_file = os.path.join(os.path.dirname(__file__), "..", "config", "hyperparameters.yaml")
        self.__params = self.load_params()

    def load_params(self):
        with open(self.param_file) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def __getitem__(self, key):
        if key in self.__params:
            return self.__params[key]
        else:
            raise KeyError(f"Key {key} not found in hyperparameters")
