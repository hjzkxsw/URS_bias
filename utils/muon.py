from torch.optim import AdamW, Muon


class HybridMuonAdamW:
    """Use Muon for 2D parameters and AdamW for scalar/vector parameters."""

    def __init__(self, named_parameters, lr=1e-4, weight_decay=1e-6,
                 muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5,
                 muon_eps=1e-7, muon_ns_coefficients=(3.4445, -4.7750, 2.0315),
                 muon_adjust_lr_fn="original"):
        self.muon_param_names = []
        self.adamw_param_names = []
        matrix_params = []
        fallback_params = []

        for name, param in named_parameters:
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                if param.ndim != 2:
                    raise ValueError(
                        "torch.optim.Muon only supports 2D parameters, but {} has shape {}.".format(
                            name, tuple(param.shape)
                        )
                    )
                matrix_params.append(param)
                self.muon_param_names.append(name)
            else:
                fallback_params.append(param)
                self.adamw_param_names.append(name)

        if not matrix_params:
            raise ValueError("HybridMuonAdamW requires at least one 2D trainable parameter.")

        self.muon = Muon(
            matrix_params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            eps=muon_eps,
            ns_coefficients=muon_ns_coefficients,
            adjust_lr_fn=muon_adjust_lr_fn,
        )
        self.adamw = AdamW(fallback_params, lr=lr, weight_decay=weight_decay) if fallback_params else None

    @property
    def param_groups(self):
        groups = list(self.muon.param_groups)
        if self.adamw is not None:
            groups.extend(self.adamw.param_groups)
        return groups

    def zero_grad(self, *args, **kwargs):
        self.muon.zero_grad(*args, **kwargs)
        if self.adamw is not None:
            self.adamw.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        loss = self.muon.step(closure=closure)
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict() if self.adamw is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        if self.adamw is not None and state_dict.get("adamw") is not None:
            self.adamw.load_state_dict(state_dict["adamw"])
