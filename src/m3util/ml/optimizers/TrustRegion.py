import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable

# extended from Zheng Shi zhs310@lehigh.edu and Majid Jahani maj316@lehigh.edu
# https://github.com/Optimization-and-Machine-Learning-Lab/TRCG
# BSD 3-Clause License
# Copyright (c) 2023, Optimization-and-Machine-Learning-Lab

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# [License details omitted for brevity]


class TRCG(Optimizer):
    """
    Implements the Trust-Region Conjugate Gradient (TRCG) optimization algorithm.

    TRCG is a second-order optimization method that combines trust-region strategies
    with conjugate gradient methods to solve optimization problems efficiently, especially
    for large-scale deep learning models.

    Attributes:
        model (torch.nn.Module): The model to be optimized.
        radius (float): Initial trust-region radius.
        device (str or torch.device): The device on which to perform the computations.
        closure_size (int): The number of parts the closure function evaluates in.
        cgopttol (float): Tolerance for CG optimization convergence.
        c0tr, c1tr, c2tr (float): Trust-region parameters for accepting or rejecting a step.
        t1tr, t2tr (float): Trust-region parameters for adjusting the radius.
        radius_max (float): Maximum allowable radius.
        radius_initial (float): Initial radius value.
        differentiable (bool): Whether the optimization is differentiable.
    """

    def __init__(
        self,
        model,
        radius,
        device,
        closure_size=1,
        cgopttol=1e-7,
        c0tr=0.0001,
        c1tr=0.1,
        c2tr=0.75,
        t1tr=0.25,
        t2tr=2.0,
        radius_max=2.0,
        radius_initial=0.1,
        differentiable: bool = False,
    ):
        """
        Initializes the TRCG optimizer.

        Args:
            model (torch.nn.Module): The model to optimize.
            radius (float): Initial trust-region radius.
            device (str or torch.device): The device to perform computations on.
            closure_size (int, optional): Number of parts the closure function evaluates in. Defaults to 1.
            cgopttol (float, optional): Tolerance for CG optimization convergence. Defaults to 1e-7.
            c0tr (float, optional): Trust-region parameter for rejecting a step. Defaults to 0.0001.
            c1tr (float, optional): Trust-region parameter for shrinking the radius. Defaults to 0.1.
            c2tr (float, optional): Trust-region parameter for accepting a step and potentially enlarging the radius. Defaults to 0.75.
            t1tr (float, optional): Trust-region parameter for radius shrinking. Defaults to 0.25.
            t2tr (float, optional): Trust-region parameter for radius enlargement. Defaults to 2.0.
            radius_max (float, optional): Maximum allowable radius. Defaults to 2.0.
            radius_initial (float, optional): Initial radius value. Defaults to 0.1.
            differentiable (bool, optional): Whether the optimization is differentiable. Defaults to False.
        """
        self.gradient_cache = None
        self.model = model
        self.device = device
        self.cgopttol = cgopttol
        self.c0tr = c0tr
        self.c1tr = c1tr
        self.c2tr = c2tr
        self.t1tr = t1tr
        self.t2tr = t2tr
        self.radius_max = radius_max
        self.radius_initial = radius_initial
        self.radius = radius
        self.cgmaxiter = sum(p.numel() for p in model.parameters())
        self.cgmaxiter = min(120, self.cgmaxiter)

        self.closure_size = closure_size

        defaults = dict(differentiable=differentiable)
        self.params = list(model.parameters())

        super(TRCG, self).__init__(self.params, defaults)

    def findroot(self, x, p):
        """
        Finds the root of the quadratic equation to adjust the trust-region radius.

        Args:
            x (list): Current parameter values.
            p (list): Direction vector.

        Returns:
            float: Step size satisfying the trust-region constraint.
        """
        aa, bb, cc = 0.0, 0.0, 0.0
        for pi, xi in zip(p, x):
            aa += (pi * pi).sum()
            bb += (pi * xi).sum()
            cc += (xi * xi).sum()
        bb = bb * 2.0
        cc = cc - self.radius**2
        alpha = (-2.0 * cc) / (bb + (bb**2 - (4.0 * aa * cc)).sqrt())
        return alpha.data.item()

    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        """
        Initializes the parameter group for the optimizer.

        Args:
            group (dict): Parameter group.
            params_with_grad (list): List of parameters with gradients.
            d_p_list (list): List of parameter gradients.
            momentum_buffer_list (list): List of momentum buffers.
        """
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if "pk" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["ph"])

    def CGSolver(self, loss_grad, cnt_compute, closure):
        """
        Conjugate Gradient (CG) solver for finding the optimization direction.

        Args:
            loss_grad (list): Gradient of the loss function with respect to model parameters.
            cnt_compute (int): Counter for the number of computations.
            closure (callable): Closure function that reevaluates the model and returns the loss.

        Returns:
            tuple: Direction vector, number of CG iterations, CG termination condition, updated computation count.
        """
        cg_iter = 0  # Iteration counter
        x0 = [
            torch.zeros(p.shape).to(self.device) for p in self.model.parameters()
        ]  # Initialize x0 as zeros

        r0 = [g.data.clone() for g in loss_grad]  # Set initial residual to gradient
        p0 = [
            -g.data.clone() for g in loss_grad
        ]  # Set initial conjugate direction to -r0
        self.cgopttol = torch.sqrt(
            sum(torch.norm(g.data) ** 2 for g in loss_grad)
        ).item()
        self.cgopttol = min(0.5, self.cgopttol**0.5) * self.cgopttol

        cg_term = 0
        j = 0

        while True:
            j += 1

            if j > self.cgmaxiter:  # If CG exceeds maximum iterations
                j -= 1
                p1 = x0
                print("\n\nCG has issues !!!\n\n")
                break

            Hp = self.computeHessianVector(
                closure, p0
            )  # Compute Hessian-vector product
            cnt_compute += 1
            pHp = self.computeDotProduct(Hp, p0)  # Quadratic term

            if pHp.item() <= 0:  # Nonpositive curvature detected
                tau = self.findroot(x0, p0)
                p1 = [xi + tau * pi for xi, pi in zip(x0, p0)]
                cg_term = 1
                break

            rr0 = sum((r**2).sum() for r in r0)

            alpha = (rr0 / pHp).item()  # Update alpha

            x1 = [xi + alpha * pi for xi, pi in zip(x0, p0)]
            norm_x1 = torch.sqrt(sum(torch.norm(xi) ** 2 for xi in x1))

            if norm_x1.item() >= self.radius:  # Check trust-region constraint
                tau = self.findroot(x0, p0)
                p1 = [xi + tau * pi for xi, pi in zip(x0, p0)]
                cg_term = 2
                break

            r1 = [ri + alpha * Hpi for ri, Hpi in zip(r0, Hp)]
            norm_r1 = torch.sqrt(sum(torch.norm(ri) ** 2 for ri in r1))

            if norm_r1.item() < self.cgopttol:  # Check for convergence
                p1 = x1
                cg_term = 3
                break

            rr1 = sum((ri**2).sum() for ri in r1)
            beta = (rr1 / rr0).item()

            p1 = [
                -ri + beta * pi for ri, pi in zip(r1, p0)
            ]  # Update conjugate direction

            p0, x0, r0 = p1, x1, r1

        cg_iter = j
        return p1, cg_iter, cg_term, cnt_compute

    def computeHessianVector(self, closure, p):
        """
        Computes the product of the Hessian matrix with a vector `p`.

        Args:
            closure (callable): Closure function that reevaluates the model and returns the loss.
            p (list): Direction vector.

        Returns:
            list: Hessian-vector product.
        """
        with torch.enable_grad():
            if self.closure_size == 1 and self.gradient_cache is not None:
                Hpp = torch.autograd.grad(
                    self.gradient_cache, self.params, grad_outputs=p, retain_graph=True
                )
                Hp = [Hpi.data.clone() for Hpi in Hpp]
            else:
                for part in range(self.closure_size):
                    loss = closure(part, self.closure_size, self.device)
                    loss_grad_v = torch.autograd.grad(
                        loss, self.params, create_graph=True
                    )
                    Hpp = torch.autograd.grad(
                        loss_grad_v, self.params, grad_outputs=p, retain_graph=False
                    )
                    if part == 0:
                        Hp = [Hpi.data.clone() for Hpi in Hpp]
                    else:
                        for Hpi, Hppi in zip(Hp, Hpp):
                            Hpi.add_(Hppi)

        return Hp

    def computeLoss(self, closure):
        """
        Computes the total loss value across the closure parts.

        Args:
            closure (callable): Closure function that reevaluates the model and returns the loss.

        Returns:
            float: Total loss value.
        """
        lossVal = 0.0
        with torch.no_grad():
            for part in range(self.closure_size):
                loss = closure(part, self.closure_size, self.device)
                lossVal += loss.item()

        return lossVal

    def computeGradientAndLoss(self, closure):
        """
        Computes the gradients and the total loss value across the closure parts.

        Args:
            closure (callable): Closure function that reevaluates the model and returns the loss.

        Returns:
            tuple: Total loss value and the gradients.
        """
        lossVal = 0.0
        with torch.enable_grad():
            for part in range(self.closure_size):
                loss = closure(part, self.closure_size, self.device)
                lossVal += loss.item()
                if self.closure_size == 1 and self.gradient_cache is None:
                    loss_grad = torch.autograd.grad(
                        loss, self.params, retain_graph=True, create_graph=True
                    )
                    self.gradient_cache = loss_grad
                else:
                    loss_grad = torch.autograd.grad(
                        loss, self.params, create_graph=False
                    )

                if part == 0:
                    grad = [p.data.clone() for p in loss_grad]
                else:
                    for gi, gip in zip(grad, loss_grad):
                        gi.add_(gip)

        return lossVal, grad

    def computeGradient(self, closure):
        """
        Computes the gradients using the closure function.

        Args:
            closure (callable): Closure function that reevaluates the model and returns the loss.

        Returns:
            list: Gradients of the loss with respect to the model parameters.
        """
        return self.computeGradientAndLoss(closure)[1]

    def computeDotProduct(self, v, z):
        """
        Computes the dot product between two lists of vectors.

        Args:
            v (list): First list of vectors.
            z (list): Second list of vectors.

        Returns:
            torch.Tensor: Dot product result.
        """
        return torch.sum(torch.stack([(vi * zi).sum() for vi, zi in zip(v, z)]))

    def computeNorm(self, v):
        """
        Computes the L2 norm of a list of vectors.

        Args:
            v (list): List of vectors.

        Returns:
            torch.Tensor: L2 norm result.
        """
        return torch.sqrt(torch.sum(torch.stack([(p**2).sum() for p in v])))

    @_use_grad_for_differentiable
    def step(self, closure):
        """
        Performs a single optimization step using the TRCG algorithm.

        Args:
            closure (callable): A closure that reevaluates the model and returns the loss.

        Returns:
            tuple: Final loss value, updated radius, computation count, and CG iteration count.
        """
        self.gradient_cache = None

        wInit = [w.clone() for w in self.params]  # Store the initial weights

        lossInit, loss_grad = self.computeGradientAndLoss(closure)
        NormG = self.computeNorm(loss_grad)

        cnt_compute = 1

        # Conjugate Gradient Method
        d, cg_iter, cg_term, cnt_compute = self.CGSolver(
            loss_grad, cnt_compute, closure
        )

        Hd = self.computeHessianVector(closure, d)
        dHd = self.computeDotProduct(Hd, d)

        # Update solution
        for wi, di in zip(self.params, d):
            with torch.no_grad():
                wi.add_(di)

        loss_new = self.computeLoss(closure)
        numerator = lossInit - loss_new

        gd = self.computeDotProduct(loss_grad, d)

        norm_d = self.computeNorm(d)

        denominator = -gd.item() - 0.5 * (dHd.item())

        # Compute ratio for trust region adjustment
        rho = numerator / denominator

        outFval = loss_new
        if rho < self.c1tr:  # Shrink radius
            self.radius = self.t1tr * self.radius
            update = 0
        elif (
            rho > self.c2tr and abs(norm_d.item() - self.radius) < 1e-10
        ):  # Enlarge radius
            self.radius = min(self.t2tr * self.radius, self.radius_max)
            update = 1
        if rho <= self.c0tr or numerator < 0:  # Reject step
            update = 3
            self.radius = self.t1tr * self.radius
            for wi, di in zip(self.params, d):
                with torch.no_grad():
                    wi.sub_(di)
            outFval = lossInit

        return outFval, self.radius, cnt_compute, cg_iter


class SimpleModel4(torch.nn.Module):
    def __init__(self):
        super(SimpleModel4, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def closure_fn3(model, device):
    def closure(part, closure_size, device):
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        targets = torch.tensor([[1.0], [2.0]], device=device)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, targets)
        return loss

    return closure


def test_trcg_step_shrink_radius():
    model = SimpleModel4()
    device = torch.device("cpu")
    optimizer = TRCG(model, radius=1.0, device=device, closure_size=1)

    closure = closure_fn3(model, device)

    # Perform a step
    outFval, radius, cnt_compute, cg_iter = optimizer.step(closure)

    # Check if the radius was shrunk
    assert radius < 1.0
    assert outFval is not None
    assert cnt_compute > 0
    assert cg_iter > 0
