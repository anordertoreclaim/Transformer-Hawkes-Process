import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer.Constants as Constants
from transformer.Modules import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class PerformerAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dropout = nn.Dropout(p=dropout)

        # Parameters of FAVOR+ attention
        self.feature_redraw_interval = 1000
        self.calls_since_last_redraw = 0
        self.random_features = None
        self.normalize_output = True
        self.normalization_stabilizer = 1e-6
        self.kernel_epsilon = 1e-4
        self.kernel_fn = lambda x, h: torch.exp(h + x)

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def redraw_features_now(self):
        device = self.random_features.device
        self._generate_feature_matrix(device)

        self.calls_since_last_redraw = 0

    def forward(self, q, k, v, mask=None):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_head, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """

        bs = q.size(0)

        def shape(x, is_value=False):
            """ separate heads """
            if is_value:
                return x.view(bs, -1, self.n_head, self.d_v).transpose(1, 2)
            else:
                return x.view(bs, -1, self.n_head, self.d_k).transpose(1, 2)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        q = shape(q)
        k = shape(k)
        v = shape(v, is_value=True)

        self._redraw_features_if_needed(q.device)

        # Get the transformed values of Q and K
        q_prime, k_prime = self.get_projected_queries_and_keys(q, k)
        output = self.compute_attention_with_projected_queries_and_keys(q_prime, k_prime, v, mask)
        output = self.dropout(self.fc(output)) + residual

        if not self.normalize_before:
            output = self.layer_norm(output)

        return output, None


    # Turns Q into Q', K into K'
    def get_projected_queries_and_keys(self, q, k):
        # Broadcast the feature matrix across the batch dimension
        new_shape = list(q.shape)
        new_shape[-2] = self.random_features.shape[-2]
        W_t = self.random_features.expand(new_shape).transpose(-2, -1)

        # Instead of dividing the product QK^T by sqrt(d), we divide Q and K by the 4th root of d.
        q = q / (self.d_model ** 0.25)
        k = k / (self.d_model ** 0.25)

        projected_q = q @ W_t
        projected_k = k @ W_t

        # Special logic for kernels that attempt to approximate softmax
        # The h(x) function is defined in Lemma 1 in Choromanski et al. pg. 4 as exp(-||x||**2 / 2). For numerical
        # stability we leverage the fact that exp(x)*exp(y) = exp(x + y) here and delay computing the exp().
        h_of_q = -torch.sum(q ** 2, dim=-1, keepdim=True) / 2
        h_of_k = -torch.sum(k ** 2, dim=-1, keepdim=True) / 2

        # Compute the numerical stabilizer that we subtract from the input to exp(). For some reason the original
        # Jax implementation uses different types of stabilizers for queries vs. keys, and we follow that here.
        # This is a workaround for very slow performance of torch.max(dim=N) on PyTorch 1.4 and earlier;
        # see this GitHub discussion: https://github.com/pytorch/pytorch/issues/36900
        q_indices = h_of_q.argmax(-1).unsqueeze(-1)
        q_stabilizer = h_of_q.gather(-1, q_indices)  # Note this is a (d_model, 1) matrix that gets broadcasted

        # This is just a scalar
        k_stabilizer = torch.max(h_of_k)

        q_kernel_output = self.kernel_fn(projected_q - q_stabilizer, h_of_q)
        k_kernel_output = self.kernel_fn(projected_k - k_stabilizer, h_of_k)

        # By multiplying by 1/sqrt(m), we ensure the final matrix product will contain a factor of 1/m. This means
        # each row of Q'K'^T can be interpreted as an average over the exp(omega^T * q) * exp(omega^T * k) terms.
        normalizing_constant = (q_kernel_output.shape[-1] ** -0.5)

        q_prime = normalizing_constant * (q_kernel_output + self.kernel_epsilon)
        k_prime = normalizing_constant * (k_kernel_output + self.kernel_epsilon)
        return q_prime, k_prime

    def compute_attention_with_projected_queries_and_keys(self, q_prime, k_prime, v, mask=None):
        # Apply the padding mask to K'. Also applying it to Q' would be redundant.
        if mask is not None:
            mask = torch.stack([torch.diag(mat) for mat in mask.type(torch.uint8)]).gt(0)
            mask = mask[:, None, :, None]
            k_prime.masked_fill_(mask, 0)

        D_inv = 1. / torch.einsum('...nd,...nd->...n', q_prime, k_prime.cumsum(dim=-2))
        context = torch.einsum('...nd,...ne->...nde', k_prime, v)
        context = context.cumsum(dim=-3)
        output = torch.einsum('...nde,...nd,...n->...ne', context, q_prime, D_inv)

        # Ensure that the output vectors are convex combinations of input vectors; that is,
        # the implied attention scores sum to 1
        if self.normalize_output:
            # Equivalent to multiplying K'^T by a ones vector
            d = q_prime @ k_prime.sum(dim=-2).unsqueeze(-1)

            # Avoid dividing by very small numbers
            d += 2 * self.normalization_stabilizer * (torch.abs(d) <= self.normalization_stabilizer)
            output /= d

        output = output.transpose(1, 2).contiguous().view(output.shape[0], -1, output.shape[1] * output.shape[-1])
        return output

    def _generate_feature_matrix(self, device):
        num_rows = int(round(self.d_k * np.log(self.d_k)))

        def get_square_block(size):
            unstructured_block = torch.randn(size, size, device='cpu')
            q, r = torch.qr(unstructured_block, some=True)
            return q.t()

        num_full_blocks = num_rows // self.d_k
        block_list = [get_square_block(self.d_k) for _ in range(num_full_blocks)]

        remaining_rows = num_rows - num_full_blocks * self.d_k
        if remaining_rows > 0:
            q = get_square_block(self.d_k)
            block_list.append(q[:remaining_rows])

        final_matrix = torch.cat(block_list)

        final_matrix *= self.d_k ** 0.5

        random_features = final_matrix.to(device)
        self.random_features = random_features

    def _redraw_features_if_needed(self, device):
        # We haven't created the projection matrix yet, let's create it
        if self.random_features is None:
            self._generate_feature_matrix(device)

        elif self.feature_redraw_interval is not None:
            if self.calls_since_last_redraw >= self.feature_redraw_interval:
                self.redraw_features_now()
            # Keep track of how many forward passes we do before we redraw again
            else:
                self.calls_since_last_redraw += 1
