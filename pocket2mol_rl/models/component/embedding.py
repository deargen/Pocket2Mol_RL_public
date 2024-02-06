from torch.nn import Linear, Module


class AtomEmbedding(Module):
    def __init__(
        self,
        in_scalar,
        in_vector,
        out_scalar,
        out_vector,
        vector_normalizer=20.0,
        space_dim=3,
    ):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

        self.space_dim = space_dim

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (
            self.space_dim,
        ), "Not support. Only one vector can be input"
        sca_emb = self.emb_sca(scalar_input[:, : self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb
