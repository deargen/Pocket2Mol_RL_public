from math import pi, sqrt

import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment as min_cost_matching
from torch import Tensor

GAUSSIAN_COEF = 1.0 / sqrt(2 * pi)


def get_gaussian_entropy(sigma: Tensor, eps=1e-6):
    """
    Args:
        sigma (Tensor): (..., space_dim), std of gaussian
    Returns:
        entorpy (Tensor): (...)
    """
    space_dim = sigma.shape[-1]
    sigma = sigma + eps
    return 0.5 * (space_dim + torch.log(2 * pi * sigma**2).sum(-1))


def get_gaussian_pdf_inner_product(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps=1e-6
):
    """
    Args:
        mu1 (Tensor): (..., space_dim), mean of gaussian distribution 1
        sigma1 (Tensor): (..., space_dim), std of gaussian distribution 1
        mu2 (Tensor): (..., space_dim), mean of gaussian distribution 2
        sigma2 (Tensor): (..., space_dim), std of gaussian distribution 2
    """
    assert mu1.shape == mu2.shape == sigma1.shape == sigma2.shape
    sum_cov = sigma1**2 + sigma2**2 + eps
    diff_mu = mu2 - mu1
    log_x = -0.5 * (
        torch.log(2 * pi * sum_cov).sum(dim=-1) + (diff_mu**2 / sum_cov).sum(dim=-1)
    )
    x = torch.exp(log_x)
    return x


def get_gaussian_mixture_pdf(mu: Tensor, sigma: Tensor, w: Tensor, x: Tensor, eps=1e-6):
    """
    Args:
        mu (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution
        sigma (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution
        w (Tensor): (..., n_comp), weight of gaussian mixture distribution
        x (Tensor): (..., space_dim) or (..., n_samples, space_dim), samples
    Returns:
        pdf (Tensor): (...) or (..., n_samples), pdf of gaussian mixture distribution
    """
    *shape, n_comp, space_dim = mu.shape
    shape = tuple(shape)
    assert sigma.shape == shape + (n_comp, space_dim)
    assert w.shape == shape + (n_comp,)
    if x.ndim == len(shape) + 1:
        x = x.unsqueeze(-2)
        sample_unsqueezed = True
    else:
        sample_unsqueezed = False
    n_samples = x.shape[-2]
    assert x.shape == shape + (n_samples, space_dim)

    mu_expand = mu.unsqueeze(dim=-3)  # (..., 1, n_comp, space_dim)
    sigma_expand = sigma.unsqueeze(dim=-3)  # (..., 1, n_comp, space_dim)
    w_expand = w.unsqueeze(dim=-2)  # (..., 1, n_comp)
    x_expand = x.unsqueeze(dim=-2)  # (..., n_samples, 1, space_dim)

    sigma_expand = sigma_expand + eps
    gaussian_pdfs = (
        GAUSSIAN_COEF
        * torch.exp(-0.5 * ((x_expand - mu_expand) / sigma_expand) ** 2)
        / sigma_expand
    ).prod(
        dim=-1
    )  # (..., n_samples, n_comp)
    pdf = (w_expand * gaussian_pdfs).sum(dim=-1)  # (..., n_samples)

    if sample_unsqueezed:
        pdf = pdf.squeeze(dim=-1)
    return pdf


def get_gaussian_kld(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps=1e-6
):
    """
    Args:
        mu1 (Tensor): (..., space_dim), mean of gaussian distribution 1
        sigma1 (Tensor): (..., space_dim), std of gaussian distribution 1
        mu2 (Tensor): (..., space_dim), mean of gaussian distribution 2
        sigma2 (Tensor): (..., space_dim), std of gaussian distribution 2
    Returns:
        kld (Tensor): D_KL(N(mu1, sigma1) || N(mu2, sigma2)) (shape: (...))
    """
    assert mu1.shape == mu2.shape == sigma1.shape == sigma2.shape

    sigma1 = sigma1 + eps
    sigma2 = sigma2 + eps

    return (
        torch.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2 * sigma2**2)
        - 0.5
    ).sum(dim=-1)


def get_gaussian_mixture_kld_upper_bound(
    mu_a: Tensor,
    sigma_a: Tensor,
    w_a: Tensor,
    mu_b: Tensor,
    sigma_b: Tensor,
    w_b: Tensor,
    eps=1e-6,
):
    """
    Args:
        mu_a (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution 1
        sigma_a (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution 1
        w_a (Tensor): (..., n_comp), weight of gaussian mixture distribution 1
        mu_b (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution 2
        sigma_b (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution 2
        w_b (Tensor): (..., n_comp), weight of gaussian mixture distribution 2
    Returns:
        kld (Tensor): D_KL(GM(mu_a, sigma_a, w_a) || GM(mu_b, sigma_b, w_b)) (shape: (...))

    """
    *shape, n_comp, space_dim = mu_a.shape
    shape = tuple(shape)
    assert mu_a.shape == shape + (n_comp, space_dim)
    assert sigma_a.shape == shape + (n_comp, space_dim)
    assert w_a.shape == shape + (n_comp,)
    assert mu_b.shape == shape + (n_comp, space_dim)
    assert sigma_b.shape == shape + (n_comp, space_dim)
    assert w_b.shape == shape + (n_comp,)
    assert (
        mu_a.shape
        == mu_b.shape
        == sigma_a.shape
        == sigma_b.shape
        == w_a.shape + (space_dim,)
        == w_b.shape + (space_dim,)
    )

    mu_a_expand_1 = mu_a.unsqueeze(dim=-2).expand(*shape, n_comp, n_comp, space_dim)
    sigma_a_expand_1 = sigma_a.unsqueeze(dim=-2).expand(
        *shape, n_comp, n_comp, space_dim
    )
    mu_a_expand_2 = mu_a.unsqueeze(dim=-3).expand(*shape, n_comp, n_comp, space_dim)
    sigma_a_expand_2 = sigma_a.unsqueeze(dim=-3).expand(
        *shape, n_comp, n_comp, space_dim
    )
    mu_b_expand_2 = mu_b.unsqueeze(dim=-3).expand(*shape, n_comp, n_comp, space_dim)
    sigma_b_expand_2 = sigma_b.unsqueeze(dim=-3).expand(
        *shape, n_comp, n_comp, space_dim
    )

    z_a_alpha = get_gaussian_pdf_inner_product(
        mu_a_expand_1, sigma_a_expand_1, mu_a_expand_2, sigma_a_expand_2, eps=eps
    )  # (..., n_comp, n_comp)

    kld_a_b = get_gaussian_kld(
        mu_a_expand_1, sigma_a_expand_1, mu_b_expand_2, sigma_b_expand_2, eps=eps
    )  # (..., n_comp, n_comp)

    return (
        w_a
        * torch.log(
            ((w_a.unsqueeze(-2) * z_a_alpha).sum(dim=-1) + eps)
            / ((w_b.unsqueeze(-2) * torch.exp(-kld_a_b)).sum(dim=-1) + eps)
        )
        + w_a * get_gaussian_entropy(sigma_a, eps=eps)
    ).sum(dim=-1)


def get_gaussian_mixture_kld_matching_estimate(
    mu_a: Tensor,
    sigma_a: Tensor,
    w_a: Tensor,
    mu_b: Tensor,
    sigma_b: Tensor,
    w_b: Tensor,
    eps=1e-6,
):
    """
    Args:
        mu_a (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution 1
        sigma_a (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution 1
        w_a (Tensor): (..., n_comp), weight of gaussian mixture distribution 1
        mu_b (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution 2
        sigma_b (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution 2
        w_b (Tensor): (..., n_comp), weight of gaussian mixture distribution 2
    Returns:
        kld (Tensor): D_KL(GM(mu_a, sigma_a, w_a) || GM(mu_b, sigma_b, w_b)) (shape: (...))

    """
    *shape, n_comp, space_dim = mu_a.shape
    shape = tuple(shape)
    assert mu_a.shape == shape + (n_comp, space_dim)
    assert sigma_a.shape == shape + (n_comp, space_dim)
    assert w_a.shape == shape + (n_comp,)
    assert mu_b.shape == shape + (n_comp, space_dim)
    assert sigma_b.shape == shape + (n_comp, space_dim)
    assert w_b.shape == shape + (n_comp,)
    assert (
        mu_a.shape
        == mu_b.shape
        == sigma_a.shape
        == sigma_b.shape
        == w_a.shape + (space_dim,)
        == w_b.shape + (space_dim,)
    )

    mu_a = mu_a.reshape(-1, n_comp, space_dim)
    sigma_a = sigma_a.reshape(-1, n_comp, space_dim)
    w_a = w_a.reshape(-1, n_comp)
    mu_b = mu_b.reshape(-1, n_comp, space_dim)
    sigma_b = sigma_b.reshape(-1, n_comp, space_dim)
    w_b = w_b.reshape(-1, n_comp)
    n = mu_a.shape[0]

    mu_a_expand = mu_a.unsqueeze(dim=-2).expand(n, n_comp, n_comp, space_dim)
    sigma_a_expand = sigma_a.unsqueeze(dim=-2).expand(n, n_comp, n_comp, space_dim)
    w_a_expand = w_a.unsqueeze(dim=-1).expand(n, n_comp, n_comp)
    mu_b_expand = mu_b.unsqueeze(dim=-3).expand(n, n_comp, n_comp, space_dim)
    sigma_b_expand = sigma_b.unsqueeze(dim=-3).expand(n, n_comp, n_comp, space_dim)
    w_b_expand = w_b.unsqueeze(dim=-2).expand(n, n_comp, n_comp)

    cost_matrices_1 = w_a_expand * (torch.log((w_a_expand + eps) / (w_b_expand + eps)))
    cost_matrices_2 = w_a_expand * (
        get_gaussian_kld(
            mu_a_expand, sigma_a_expand, mu_b_expand, sigma_b_expand, eps=eps
        )
    )
    cost_matrices = w_a_expand * (
        torch.log((w_a_expand + eps) / (w_b_expand + eps))
        + get_gaussian_kld(
            mu_a_expand, sigma_a_expand, mu_b_expand, sigma_b_expand, eps=eps
        )
    )

    device = mu_a.device
    dtype = mu_a.dtype
    kld_estimate = torch.empty(n, dtype=dtype, device=device)
    for i in range(n):
        cost_matrix = cost_matrices[i].cpu().numpy()
        row_ind, col_ind = min_cost_matching(cost_matrix)
        total_cost = float(cost_matrix[row_ind, col_ind].sum())
        kld_estimate[i] = total_cost

    return kld_estimate.reshape(shape)


def get_gaussian_mixture_kld_lower_bound(
    mu_a: Tensor,
    sigma_a: Tensor,
    w_a: Tensor,
    mu_b: Tensor,
    sigma_b: Tensor,
    w_b: Tensor,
    eps=1e-6,
):
    """
    Args:
        mu_a (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution 1
        sigma_a (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution 1
        w_a (Tensor): (..., n_comp), weight of gaussian mixture distribution 1
        mu_b (Tensor): (..., n_comp, space_dim), mean of gaussian mixture distribution 2
        sigma_b (Tensor): (..., n_comp, space_dim), std of gaussian mixture distribution 2
        w_b (Tensor): (..., n_comp), weight of gaussian mixture distribution 2
    Returns:
        kld (Tensor): D_KL(GM(mu_a, sigma_a, w_a) || GM(mu_b, sigma_b, w_b)) (shape: (...))

    """
    *shape, n_comp, space_dim = mu_a.shape
    shape = tuple(shape)
    assert mu_a.shape == shape + (n_comp, space_dim)
    assert sigma_a.shape == shape + (n_comp, space_dim)
    assert w_a.shape == shape + (n_comp,)
    assert mu_b.shape == shape + (n_comp, space_dim)
    assert sigma_b.shape == shape + (n_comp, space_dim)
    assert w_b.shape == shape + (n_comp,)
    assert (
        mu_a.shape
        == mu_b.shape
        == sigma_a.shape
        == sigma_b.shape
        == w_a.shape + (space_dim,)
        == w_b.shape + (space_dim,)
    )

    mu_a_expand_1 = mu_a.unsqueeze(dim=-2).expand(*shape, n_comp, n_comp, space_dim)
    sigma_a_expand_1 = sigma_a.unsqueeze(dim=-2).expand(
        *shape, n_comp, n_comp, space_dim
    )
    mu_a_expand_2 = mu_a.unsqueeze(dim=-3).expand(*shape, n_comp, n_comp, space_dim)
    sigma_a_expand_2 = sigma_a.unsqueeze(dim=-3).expand(
        *shape, n_comp, n_comp, space_dim
    )
    mu_b_expand_2 = mu_b.unsqueeze(dim=-3).expand(*shape, n_comp, n_comp, space_dim)
    sigma_b_expand_2 = sigma_b.unsqueeze(dim=-3).expand(
        *shape, n_comp, n_comp, space_dim
    )

    kld_a_alpha = get_gaussian_kld(
        mu_a_expand_1, sigma_a_expand_1, mu_a_expand_2, sigma_a_expand_2, eps=eps
    )  # (..., n_comp, n_comp)
    t_a_b = get_gaussian_pdf_inner_product(
        mu_a_expand_1, sigma_a_expand_1, mu_b_expand_2, sigma_b_expand_2, eps=eps
    )  # (..., n_comp, n_comp)

    return (
        w_a
        * torch.log(
            (w_a.unsqueeze(-2) * torch.exp(-kld_a_alpha)).sum(dim=-1)
            / (w_b.unsqueeze(-2) * t_a_b).sum(dim=-1)
        )
        - w_a * get_gaussian_entropy(sigma_a)
    ).sum(dim=-1)


def test_random(
    n_comp=5,
    space_dim=3,
    n_instance=1000,
    mc_sample=1000,
    eps=1e-10,
    dists="independent",
    seed=42,
    small_kld=None,
    method="upper_bound",
):
    torch.manual_seed(seed)

    mu1 = torch.randn(n_instance, n_comp, space_dim)
    sigma1 = torch.rand(n_instance, n_comp, space_dim)
    w1 = torch.randn(n_instance, n_comp).softmax(dim=-1)
    if dists == "independent":
        mu2 = torch.randn(n_instance, n_comp, space_dim)
        sigma2 = torch.rand(n_instance, n_comp, space_dim)
        w2 = torch.randn(n_instance, n_comp).softmax(dim=-1)
    elif dists == "noise":
        mu2 = mu1 + 0.05 * torch.randn(n_instance, n_comp, space_dim)
        sigma2 = sigma1 * (0.95 + 0.1 * torch.rand(n_instance, n_comp, space_dim))
        w2 = w1
    else:
        raise ValueError(dists)

    comp_samples = torch.multinomial(
        w1, mc_sample, replacement=True
    )  # (n_instance, mc_sample)
    selected_mu = torch.gather(
        mu1, 1, comp_samples.unsqueeze(-1).expand(-1, -1, space_dim)
    )  # (n_instance, mc_sample, space_dim)
    selected_sigma = torch.gather(
        sigma1, 1, comp_samples.unsqueeze(-1).expand(-1, -1, space_dim)
    )  # (n_instance, mc_sample, space_dim)

    z_samples = torch.randn(n_instance, mc_sample, space_dim)
    x_samples = (
        selected_mu + selected_sigma * z_samples
    )  # (n_instance, mc_sample, space_dim)
    pdf1 = get_gaussian_mixture_pdf(mu1, sigma1, w1, x_samples, eps=eps)
    pdf2 = get_gaussian_mixture_pdf(mu2, sigma2, w2, x_samples, eps=eps)
    kld_samples = torch.log(pdf1 / (pdf2 + eps))  # (n_instance, mc_sample)
    kld_estimate = kld_samples.mean(dim=-1)  # (n_instance)
    kld_estimate_std = kld_samples.std(dim=-1)  # (n_instance)

    if method == "upper_bound":
        fast_estimate = get_gaussian_mixture_kld_upper_bound(
            mu1, sigma1, w1, mu2, sigma2, w2, eps=eps
        )
    elif method == "matching":
        fast_estimate = get_gaussian_mixture_kld_matching_estimate(
            mu1, sigma1, w1, mu2, sigma2, w2, eps=eps
        )
    else:
        pass

    normalized_error = (fast_estimate - kld_estimate) / kld_estimate_std

    print("average size of normalized error:", normalized_error.abs().mean())

    plt.figure(figsize=(20, 7))

    plt.subplot(1, 3, 1)
    plt.hist(kld_estimate.tolist(), bins=40, label="estimate", alpha=0.5, color="b")
    plt.hist(fast_estimate.tolist(), bins=40, label=method, alpha=0.5, color="r")
    plt.legend()

    plt.subplot(1, 3, 2)
    if small_kld is not None:
        assert small_kld > 0
        small_kld_idxs = kld_estimate < small_kld
        _kld_estimate = kld_estimate[small_kld_idxs]
        _fast_estimate = fast_estimate[small_kld_idxs]
    else:
        _kld_estimate = kld_estimate
        _fast_estimate = fast_estimate
    plt.scatter(_kld_estimate.tolist(), _fast_estimate.tolist(), s=1)
    plt.xlabel("estimate")
    plt.ylabel(method)

    plt.subplot(1, 3, 3)
    normalized_upper_errors = (fast_estimate - kld_estimate) / kld_estimate
    plt.hist(normalized_upper_errors.tolist(), bins=40)
    plt.xlabel("(upper_bound - estimate) / estimate")

    """
    plt.subplot(1, 3, 3)
    normalized_lower_errors = (kld_lower_bound - kld_estimate) / kld_estimate
    plt.hist(normalized_lower_errors.tolist(), bins=40)
    plt.xlabel("(lower_bound - estimate) / estimate")
    """

    plt.suptitle("KLD for 1000 random pairs of gaussian mixture distributions")
    plt.savefig(f"kld_estimates_{method}_{dists}.png")


if __name__ == "__main__":
    for method in ["matching", "upper_bound"]:
        test_random(dists="independent", method=method)
        test_random(
            dists="noise", mc_sample=10000, small_kld=1.0, method=method, eps=1e-6
        )
