from abc import ABCMeta, abstractmethod


class BaseKD(object, metaclass=ABCMeta):
    @abstractmethod
    def tensorflow(self):
        pass

    @abstractmethod
    def torch(self):
        pass

    @abstractmethod
    def _check_config(self, config):
        pass


class ST(BaseKD):
    def __init__(self, alpha, T):
        self.config = {"alpha": alpha, "T": T}

    def _check_config(self, config):
        assert type(self.config["alpha"]) == float
        assert self.config["alpha"] <= 1.0 and self.config["alpha"] >= 0

        assert type(self.config["T"]) == float or type(self.config["T"]) == int

    def tensorflow(self):
        from .st_tensorflow import ST as TENSORFLOW_ST

        self._check_config(self.config)
        return TENSORFLOW_ST(self.config)

    def torch(self):
        from .st_torch import ST as TORCH_ST

        self._check_config(self.config)
        return TORCH_ST(self.config)
