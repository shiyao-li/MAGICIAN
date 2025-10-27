import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather


def attention(q, k, v, mask=None, dropout=None):
    """
    Main attention mechanism function.

    Takes queries, keys and values as inputs and computes attention scores.
    :param q: (Tensor) Queries tensor with shape (..., N, d)
    :param k: (Tensor) Keys tensor with shape (..., N, d)
    :param v: (Tensor) Values tensor with shape (..., N, d)
    :param mask: (Tensor) Mask tensor with shape (..., N, N). Optional.
    :param dropout: Dropout module to apply on computed scores.
    :return: scores: (Tensor) Attention scores tensor with shape (..., N)
    """
    # Query/Key matrix multiplication
    scores = q.matmul(k.transpose(-2, -1))

    # Apply mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)

    # Normalization
    scores /= np.sqrt(q.shape[-1])
    scores = nn.functional.softmax(scores, dim=-1)

    # Dropout
    scores = scores if dropout is None else dropout(scores)

    # Value matrix multiplication
    scores = scores.matmul(v)

    return scores


class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim,
                 dropout=None, gelu=True,
                 global_feature=False,
                 additional_feature_dim=0,
                 concatenate_input=True,
                 k_for_knn=0):
        """
        Class used to embed point clouds.

        :param input_dim: (int) Dimension of input points.
        :param output_dim: (int) Dimension of output features.
        :param dropout: Dropout module to apply on computed embeddings.
        :param gelu: (bool) If True, module uses GELU non-linearity. If False, module uses ReLU.
        :param global_feature: (bool) If True, output features are computed as the concatenation of per-point features
         and a global feature.
        :param additional_feature_dim: (int) Dimension of an additional feature to provide.
        :param concatenate_input: (bool) If True, concatenates input to the output features.
        :param k_for_knn: (int) if > 0, output features are computed by max-pooling features from k nearest neighbors.
        """
        super(Embedding, self).__init__()

        self.use_knn = k_for_knn > 0
        self.k = k_for_knn

        self.input_dim = input_dim

        self.global_feature = global_feature
        self.additional_feature_dim = additional_feature_dim
        self.concatenate_input = concatenate_input

        self.inner_dim = output_dim // 2
        self.feature_dim = output_dim

        if additional_feature_dim > 0:
            self.feature_dim -= additional_feature_dim
            self.inner_dim = self.feature_dim

        if concatenate_input:
            self.feature_dim -= input_dim
            self.inner_dim = self.feature_dim

        if global_feature:
            self.feature_dim = self.feature_dim // 2
            self.inner_dim = self.feature_dim

        if global_feature or self.use_knn:
            self.max_pool = nn.functional.max_pool1d

        self.linear1 = nn.Linear(self.input_dim, self.inner_dim)
        self.linear2 = nn.Linear(self.inner_dim, self.feature_dim)

        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        if gelu:
            self.nonlinear = nn.GELU()
        else:
            self.nonlinear = nn.ReLU(inplace=False)

    def forward(self, x, additional_feature=None):
        n_clouds, seq_len, x_dim = x.shape

        res = self.nonlinear(self.linear1(x))
        res = res if self.dropout is None else self.dropout(res)
        res = self.linear2(res)

        if self.use_knn:
            # Computing spatial kNN
            _, knn_idx, _ = knn_points(p1=x[..., :3], p2=x[..., :3], K=self.k)
            res_knn = knn_gather(res, knn_idx)

            # Pooling among kNN features
            res_knn = res_knn.view(n_clouds * seq_len, self.k, self.feature_dim)
            res = self.max_pool(input=res_knn.transpose(-1, -2),
                                kernel_size=self.k
                                ).view(n_clouds, seq_len, self.feature_dim)

        if self.global_feature:
            global_feature = self.max_pool(input=res.transpose(-1, -2),
                                           kernel_size=seq_len).view(n_clouds, 1, self.feature_dim)
            global_feature = global_feature.expand(-1, seq_len, -1)
            res = torch.cat((res, global_feature), dim=-1)

        if self.additional_feature_dim > 0:
            res = torch.cat((res, additional_feature), dim=-1)

        if self.concatenate_input:
            res = torch.cat((res, x), dim=-1)

        return res


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, in_dim, qk_dim, dropout=None):
        """
        Main class for Multi-Head Self Attention neural module.
        Credits to https://arxiv.org/pdf/2012.09688.pdf

        :param n_heads: (int) Number of heads in the attention module.
        :param in_dim: (int) Dimension of input.
        :param qk_dim: (int) Dimension of keys and queries.
        :param dropout: Dropout module to apply on computed embeddings.
        """
        super(MultiHeadSelfAttention, self).__init__()
        v_dim = in_dim

        self.n_heads = n_heads
        self.in_dim = in_dim
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.qk_dim_per_head = qk_dim // n_heads
        self.v_dim_per_head = v_dim // n_heads

        self.w_q = nn.Linear(in_dim, qk_dim)
        self.w_k = nn.Linear(in_dim, qk_dim)
        self.w_v = nn.Linear(in_dim, v_dim)

        self.dropout = nn.Dropout(dropout) if dropout is not None else None

        if n_heads > 1:
            self.out = nn.Linear(in_dim, in_dim)

    def split_heads(self, q, k, v):
        """
        Split all queries, keys and values between attention heads.

        :param q: (Tensor) Queries tensor with shape (batch_size, seq_len, qk_dim)
        :param k: (Tensor) Keys tensor with shape (batch_size, seq_len, qk_dim)
        :param v: (Tensor) Values tensor with shape (batch_size, seq_len, v_dim)
        :return: (3-tuple of Tensors) Queries, keys and values for each head.
        q_split and k_split tensors have shape (batch_size, seq_len, n_heads, qk_dim_per_head).
        v_split tensor has shape (batch_size, seq_len, n_heads, v_dim_per_head).
        """
        # Each have size n_screen_cameras * pts_len * n_heads * dim_per_head
        q_split = q.reshape(q.shape[0], -1, self.n_heads, self.qk_dim_per_head)
        k_split = k.reshape(k.shape[0], -1, self.n_heads, self.qk_dim_per_head)
        v_split = v.reshape(v.shape[0], -1, self.n_heads, self.v_dim_per_head)

        return q_split, k_split, v_split

    def forward(self, x, mask=None):
        """
        Forward pass.

        :param x: (Tensor) Input tensor with shape (batch_size, seq_len, in_dim)
        :param mask: (Tensor) Mask tensor with shape (batch_size, seq_len, seq_len). Optional.
        :return: scores (Tensor) Attention scores tensor with shape (batch_size, seq_len, in_dim)
        """
        # pts should have size BS * SEQ_LEN * IN_DIM
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Break into n_heads
        q, k, v = self.split_heads(q, k, v)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # BS * HEAD * SEQ_LEN * DIM_PER_HEAD

        scores = attention(q, k, v, mask, self.dropout)  # BS * HEAD * SEQ_LEN * DIM_PER_HEAD
        scores = scores.transpose(1, 2).contiguous().view(
            scores.shape[0], -1, self.v_dim)  # BS * SEQ_LEN * V_DIM

        if self.n_heads > 1:
            scores = self.out(scores)

        return scores


class FeedForward(nn.Module):
    def __init__(self, input_dim, inner_dim, gelu=True, dropout=None):
        """
        Feed Forward unit to use in attention encoder.

        :param input_dim: (int) Dimension of input tensor.
        :param inner_dim: (int) Dimension of inner tensor.
        :param gelu: (bool) If True, the unit uses GELU non-linearity. if False, it uses ReLU.
        :param dropout: Dropout module to apply on computed embeddings.
        """
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, input_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        if gelu:
            self.nonlinear = nn.GELU()
        else:
            self.nonlinear = nn.ReLU(inplace=False)

    def forward(self, x):
        """
        Forward pass.

        :param x: (Tensor) Input tensor with shape (..., input_dim)
        :return: (Tensor) Output tensor with shape (..., input_dim)
        """
        res = self.nonlinear(self.linear1(x))
        res = res if self.dropout is None else self.dropout(res)

        return self.linear2(res)


class Encoder(nn.Module):
    def __init__(self, seq_len, qk_dim, embedding_dim=128, n_heads=1,
                 dropout=None, gelu=True, FF=True):
        """
        Transformer encoder based on a Multi-Head Self Attention mechanism.

        :param seq_len: (int) Length of input sequence.
        :param qk_dim: (int) Dimension of keys and queries.
        :param embedding_dim: (int) Dimension of input embeddings, values, and attention scores.
        :param n_heads: (int) Number of heads in the self-attention module.
        :param dropout: Dropout module to apply on computed features.
        :param gelu: (bool) If True, the encoder uses GELU non-linearity. if False, it uses ReLU.
        :param FF: (bool) If True, the encoder applies an additional Feed Forward unit after
        the Multi-Head Self Attention unit.
        """
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.qk_dim = qk_dim  # self.embedding_dim // 4
        self.dropout = dropout
        self.FF = FF

        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.mhsa = MultiHeadSelfAttention(n_heads=self.n_heads,
                                           in_dim=self.embedding_dim,
                                           qk_dim=self.qk_dim,
                                           dropout=self.dropout)
        self.dropout1 = None if dropout is None else nn.Dropout(dropout)

        if FF:
            self.norm2 = nn.LayerNorm(self.embedding_dim)
            self.ff = FeedForward(input_dim=embedding_dim,
                                  inner_dim=2*embedding_dim,
                                  gelu=gelu,
                                  dropout=self.dropout)
            self.dropout2 = None if dropout is None else nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass.

        :param x: (Tensor) Input tensor with shape (batch_size, seq_len, embedding_dim).
        :param mask: (Tensor) Mask tensor with shape (batch_size, seq_len, seq_len). Optional.
        :return: (Tensor) The features encoded with a self-attention mechanism.
        Has shape (batch_size, seq_len, embedding_dim)
        """
        res = self.norm1(x)
        res = self.mhsa(res, mask=mask)
        if self.dropout is not None:
            res = self.dropout1(res)
        res = x + res

        if self.FF:
            res2 = self.norm2(res)
            res2 = self.ff(res2)
            if self.dropout is not None:
                res2 = self.dropout2(res2)
            res = res + res2

        return res