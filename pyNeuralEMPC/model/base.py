
class Model:
    def __init__(self, output_dim: int):
        self.output_dim = output_dim

    def forward(self, x):
        raise NotImplementedError("")

    def jacobian(self, x):
        raise NotImplementedError("")

    def hessian(self, x):
        raise NotImplementedError("")


CONSTANT_VAR = 1
CONTROL_VAR  = 2
STATE_VAR    = 3

class ReOrderProxyModel(Model):
    def __init__(self, model, order_list: list):
        raise NotImplementedError("")