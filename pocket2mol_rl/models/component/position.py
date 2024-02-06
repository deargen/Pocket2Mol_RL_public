import math

import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F

from .invariant import GVLinear, GVPerceptronVN

GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)


class PositionPredictor(Module):
    def __init__(self, in_sca, in_vec, num_filters, n_component):
        super().__init__()
        self.n_component = n_component
        self.gvp = Sequential(
            GVPerceptronVN(in_sca, in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )
        self.mu_net = GVLinear(num_filters[0], num_filters[1], n_component, n_component)
        self.logsigma_net = GVLinear(
            num_filters[0], num_filters[1], n_component, n_component
        )
        self.pi_net = GVLinear(num_filters[0], num_filters[1], n_component, 1)

    def freeze_mu(self):
        for param in self.gvp.parameters():
            param.requires_grad = False

        for param in self.mu_net.parameters():
            param.requires_grad = False

    def forward(self, h_compose, idx_focal, pos_compose):
        h_focal = [h[idx_focal] for h in h_compose]
        pos_focal = pos_compose[idx_focal]

        feat_focal = self.gvp(h_focal)
        relative_mu = self.mu_net(feat_focal)[1]  # (N_focal, n_component, space_dim)
        logsigma = self.logsigma_net(feat_focal)[1]  # (N_focal, n_component, space_dim)
        sigma = torch.exp(logsigma)
        pi = self.pi_net(feat_focal)[0]  # (N_focal, n_component)
        pi = F.softmax(pi, dim=1)

        abs_mu = relative_mu + pos_focal.unsqueeze(dim=1).expand_as(relative_mu)
        return relative_mu, abs_mu, sigma, pi

    def get_mdn_probability(self, mu, sigma, pi, pos_target):
        """TODO: Check if the probability needs eps to avoid nan"""
        prob_gauss = self._get_gaussian_probability(mu, sigma, pos_target)
        prob_mdn = pi * prob_gauss
        prob_mdn = torch.sum(prob_mdn, dim=1)
        return prob_mdn

    def _get_gaussian_probability(self, mu, sigma, pos_target):
        """
        mu - (N, n_component, space_dim)
        sigma - (N, n_component, space_dim)
        pos_target - (N, space_dim)
        """
        target = pos_target.unsqueeze(1).expand_as(mu)
        errors = target - mu
        sigma = sigma + 1e-16
        p = GAUSSIAN_COEF * torch.exp(-0.5 * (errors / sigma) ** 2) / sigma
        p = torch.prod(p, dim=2)
        return p  # (N, n_component)

    def sample_batch(self, mu, sigma, pi, num):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, space_dim)
            sigma - (N_batch, n_cat, space_dim)
            pi - (N_batch, n_cat)
        return
            (N_batch, num, space_dim)
        """
        index_cats = torch.multinomial(pi, num, replacement=True)  # (N_batch, num)
        # index_cats = index_cats.unsqueeze(-1)
        index_batch = (
            torch.arange(len(mu)).unsqueeze(-1).expand(-1, num)
        )  # (N_batch, num)
        mu_sample = mu[index_batch, index_cats]  # (N_batch, num, space_dim)
        sigma_sample = sigma[index_batch, index_cats]
        values = torch.normal(mu_sample, sigma_sample)  # (N_batch, num, space_dim)
        return values

    def get_maximum(self, mu, sigma, pi):
        """sample from multiple mix gaussian
            mu - (N_batch, n_cat, self.space_dim)
            sigma - (N_batch, n_cat, self.space_dim)
            pi - (N_batch, n_cat)
        return
            (N_batch, n_cat, self.space_dim)
        """
        return mu


class PositionEvaluator(Module):
    def __init__(self, in_sca, in_vec, num_filters):
        super().__init__()
        self.gvp = Sequential(
            GVPerceptronVN(in_sca, in_vec, num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
        )
        self.addi_gvp1 = GVPerceptronVN(
            num_filters[0], num_filters[1], num_filters[0], 2 * num_filters[1]
        )  # The second halves of the vector output will be used for inner product with the (relative) positions of interest
        self.addi_gvp2 = GVPerceptronVN(
            2 * num_filters[0], num_filters[1], num_filters[0], num_filters[1]
        )

        self.final_linear = GVLinear(num_filters[0], num_filters[1], 1, None)

    def load_params_from(self, model):
        if type(model) != PositionPredictor:
            raise NotImplementedError()

        self.gvp.load_state_dict(model.gvp.state_dict())

    def forward(self, h_compose, idx_focal, pos_compose, pos):
        """
        Args:
            pos: position to evaluate
        """
        n_focal = len(idx_focal)
        space_dim = pos.shape[-1]
        assert pos.shape == (n_focal, space_dim)
        pos_focal = pos_compose[idx_focal]
        relpos = pos - pos_focal  # (n_focal, space_dim)

        h_focal = [
            h[idx_focal] for h in h_compose
        ]  # (n_focal, in_sca,), (n_focal, in_vec, space_dim)

        feat_focal = self.gvp(h_focal)

        x_sca, x_vec = self.addi_gvp1(feat_focal)
        n = x_vec.shape[1]
        assert x_vec.shape == (n_focal, n, space_dim), x_vec.shape
        assert n % 2 == 0
        x_vec, x_vec2 = x_vec[:, : n // 2, :], x_vec[:, n // 2 :, :]
        inner_product = torch.sum(x_vec2 * relpos.unsqueeze(dim=1), dim=-1)
        x_sca = torch.cat([x_sca, inner_product], dim=1)
        x = self.addi_gvp2([x_sca, x_vec])

        out = self.final_linear(x)[0]  # (n_focal, 1)
        return out
