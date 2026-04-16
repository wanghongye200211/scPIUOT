import torch
from collections import OrderedDict
from torch import nn
import src.sde as sde






class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)

class MLP(torch.nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(input_dim, hidden_dim), LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(hidden_dim, hidden_dim))
            model.append(LipSwish())
        model.append(torch.nn.Linear(hidden_dim, out_dim))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)

class AutoGenerator(nn.Module):

    def __init__(self, config):
        super(AutoGenerator, self).__init__()

        self.dim = config.x_dim
        self.k_dims = config.k_dims
        self.layers = config.layers
        self.sigma_type = config.sigma_type
        self.sigma_const = config.sigma_const
        self.use_growth = bool(getattr(config, "use_growth", False))
        self.growth_mode = str(getattr(config, "growth_mode", "free"))
        self.growth_scale = float(getattr(config, "growth_scale", 0.05))
        self.hjb_growth_coeff = float(getattr(config, "hjb_growth_coeff", 2.0))

        self.activation = config.activation
        if self.activation == 'relu':
            self.act = nn.LeakyReLU
        elif self.activation == 'softplus':
            self.act = nn.Softplus
        elif self.activation == 'tanh':
            self.act = nn.Tanh
        elif self.activation == 'none':
            self.act = None
        else:
            raise NotImplementedError
        
        self.net_ = []
        for i in range(self.layers): 
            if i == 0: 
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.dim+1, self.k_dims[i]))) 
            else: 
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.k_dims[i-1], self.k_dims[i]))) 
            if self.activation == 'none': 
                pass
            else:
                self.net_.append(('{}{}'.format(self.activation, i+1), self.act()))
        self.net_.append(('linear', nn.Linear(self.k_dims[-1], 1, bias = False)))
        self.net_ = OrderedDict(self.net_)
        self.net = nn.Sequential(self.net_)

        net_params = list(self.net.parameters())
        net_params[-1].data = torch.zeros(net_params[-1].data.shape) 

        self.noise_type = 'diagonal'
        self.sde_type = "ito"

        cfg = dict(
            input_dim=self.dim + 1,
            out_dim=self.dim,
            hidden_dim=128,
            num_layers=2,
            tanh=True
        )
        if self.sigma_type == 'Mlp':
            self.sigma = MLP(**cfg)
        elif self.sigma_type == "const":
            self.register_buffer('sigma', torch.as_tensor(self.sigma_const))
            self.sigma = self.sigma.repeat(self.dim).unsqueeze(0)
        elif self.sigma_type == "const_param":
            self.sigma = nn.Parameter(torch.randn(1,self.dim), requires_grad=True)

        if self.use_growth:
            growth_layers = []
            for i in range(self.layers):
                if i == 0:
                    growth_layers.append((f"linear{i+1}", nn.Linear(self.dim + 1, self.k_dims[i])))
                else:
                    growth_layers.append((f"linear{i+1}", nn.Linear(self.k_dims[i - 1], self.k_dims[i])))
                if self.activation != 'none':
                    growth_layers.append((f"{self.activation}{i+1}", self.act()))
            growth_layers.append(("linear", nn.Linear(self.k_dims[-1], 1)))
            self.growth_net = nn.Sequential(OrderedDict(growth_layers))
            growth_params = list(self.growth_net.parameters())
            growth_params[-2].data.zero_()
            growth_params[-1].data.zero_()

    def _pot(self, xt):
        xt = xt.requires_grad_()
        pot = self.net(xt)
        return pot

    def _state_x(self, state):
        if self.use_growth:
            return state[:, 0:-2]
        return state[:, 0:-1]

    def _growth(self, xt):
        if not self.use_growth:
            return xt.new_zeros((xt.shape[0], 1))
        growth = self.growth_net(xt)
        if self.growth_mode == "bounded":
            growth = self.growth_scale * torch.tanh(growth)
        return growth

    def f(self, t, x_r):
        x = self._state_x(x_r)
        t = x.new_full((x.shape[0], 1), float(t))
        xt = torch.cat([x, t], dim=1)
        pot = self._pot(xt)                    # batch * 1
        drift = torch.autograd.grad(pot, xt, torch.ones_like(pot),create_graph=True)[0]

        drift_x = -drift[:,0:-1]               # batch * N
        drift_t = drift[:,-1].unsqueeze(1)     # batch *1

        delta_hjb = drift_t - 0.5 * torch.sum(torch.pow(drift_x, 2), 1, keepdims=True)
        if self.use_growth:
            growth = self._growth(xt)
            delta_hjb = delta_hjb + 0.5 * self.hjb_growth_coeff * growth.pow(2)
            delta_hjb = torch.abs(delta_hjb)
            new_drift = torch.cat([drift_x, delta_hjb, growth], dim=1)  # batch * (N+2)
        else:
            delta_hjb = torch.abs(delta_hjb)
            new_drift = torch.cat([drift_x, delta_hjb], dim=1)  # batch * (N+1)
        return new_drift
    
    def _drift(self, xt):
        pot = self._pot(xt)                    # batch * 1
        drift = torch.autograd.grad(pot, xt, torch.ones_like(pot),create_graph=True)[0]

        drift_x = -drift[:,0:-1]                # batch * N
        return drift_x

    def g(self, t, x_r):
        x = self._state_x(x_r)
        if self.sigma_type == "Mlp": 
            t = x.new_full((x.shape[0], 1), float(t))
            xt = torch.cat([x, t], dim=1)
            g = self.sigma(xt).view(-1, self.dim)
        elif self.sigma_type == "const":
            g = self.sigma.repeat(x.shape[0], 1)
        elif self.sigma_type == "const_param":
            g = self.sigma.repeat(x.shape[0], 1)
        extra_dim = 2 if self.use_growth else 1
        extra_g = x.new_zeros((x.shape[0], extra_dim))
        g = torch.cat([g, extra_g], dim=1)
        return g





class ForwardSDE(torch.nn.Module):
    def __init__(self, config):
        super(ForwardSDE, self).__init__()

        self._func = AutoGenerator(config)
        self.solver_dt = float(getattr(config, "solver_dt", 0.1))

    def forward(self, ts, x_r_0):
        x_r_s = sde.sdeint_adjoint(self._func, x_r_0, ts, method='euler', dt=self.solver_dt, dt_min=0.0001,
                                     adjoint_method='euler', names={'drift': 'f', 'diffusion': 'g'} )
        return x_r_s
