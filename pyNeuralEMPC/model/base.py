
class Model:
    def __init__(self, x_dim: int, u_dim: int, p_dim=None, tvp_dim=None):
        self.x_dim = x_dim
        self.u_dim = u_dim

        self.p_dim = p_dim
        self.tvp_dim = tvp_dim

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