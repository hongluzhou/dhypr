import torch.optim
from manifolds import Euclidean, ManifoldParameter

_default_manifold = Euclidean()


class OptimMixin(object):
    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        for group in self.param_groups:
            self.stabilize_group(group)


def copy_or_set_(dest, source):
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter)):
                        manifold = point.manifold
                        c = point.c
                    else:
                        manifold = _default_manifold
                        c = None
                    if grad.is_sparse:
                        raise RuntimeError(
                                "Riemannian Adam does not support sparse gradients yet (PR is welcome)"
                        )

                    state = self.state[point]

                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(point)
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    
                    grad.add_(weight_decay, point)
                    grad = manifold.egrad2rgrad(point, grad, c)
                    exp_avg.mul_(betas[0]).add_(1 - betas[0], grad)
                    exp_avg_sq.mul_(betas[1]).add_(
                            1 - betas[1], manifold.inner(point, c, grad, keepdim=True)
                    )
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        denom = max_exp_avg_sq.sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)
                    group["step"] += 1
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    step_size = (
                        learning_rate * bias_correction2 ** 0.5 / bias_correction1
                    )

                    direction = exp_avg / denom
                    new_point = manifold.proj(manifold.expmap(-step_size * direction, point, c), c)
                    exp_avg_new = manifold.ptransp(point, new_point, exp_avg, c)
                    copy_or_set_(point, new_point)
                    exp_avg.set_(exp_avg_new)

                    group["step"] += 1
                if self._stabilize is not None and group["step"] % self._stabilize == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, ManifoldParameter):
                continue
            state = self.state[p]
            if not state:   
                continue
            manifold = p.manifold
            c = p.c
            exp_avg = state["exp_avg"]
            copy_or_set_(p, manifold.proj(p, c))
            exp_avg.set_(manifold.proj_tan(exp_avg, u, c))
