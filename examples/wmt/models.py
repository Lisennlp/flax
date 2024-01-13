# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based machine translation model."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic
# See issue #620.
# pytype: disable=wrong-arg-count
# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error

import math  # XD
from typing import Any, Callable, Optional, Tuple, Union, overload
import jax
from flax import linen as nn
from flax import struct
from flax.linen.normalization import LayerNorm
from flax.linen.dtypes import promote_dtype
from flax.linen.linear import PrecisionLike, default_kernel_init, DenseGeneral
from flax.linen import initializers  # XD
from einops import rearrange, repeat  # XD

from jax import lax
import jax.numpy as jnp
import numpy as np


PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


@struct.dataclass
class TransformerConfig:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  vocab_size: int
  output_vocab_size: int
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 2048
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  decode: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable] = None
  en_dynamic_compose: bool = False
  de_dynamic_compose1: bool = False
  de_dynamic_compose2: bool = False
  dynamic_dropout_rate: float = None

def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0)
  )
  return padded[:, :-1]


def sinusoidal_init(max_len=2048, min_scale=1.0, max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, : d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    decode: whether to run in single-position autoregressive mode.
  """

  config: TransformerConfig
  decode: bool = False

  @nn.compact
  def __call__(self, inputs, inputs_positions=None):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init in the configuration.

    Args:
      inputs: input data.
      inputs_positions: input position indices for packed sequences.

    Returns:
      output: `(bs, timesteps, in_dim)`
    """
    config = self.config
    # inputs.shape is (batch_size, seq_len, emb_dim)
    assert inputs.ndim == 3, (
        'Number of dimensions should be 3, but it is: %d' % inputs.ndim
    )
    length = inputs.shape[1]
    pos_emb_shape = (1, config.max_len, inputs.shape[-1])
    if config.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(max_len=config.max_len)(
          None, pos_emb_shape, None
      )
    else:
      pos_embedding = self.param(
          'pos_embedding', config.posemb_init, pos_emb_shape
      )
    pe = pos_embedding[:, :length, :]

    # We use a cache position index for tracking decoding position.
    if self.decode:
      is_initialized = self.has_variable('cache', 'cache_index')
      cache_index = self.variable(
          'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.uint32)
      )
      if is_initialized:
        i = cache_index.value
        cache_index.value = i + 1
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)

def unbind(ary, n, axis=0):
  return [jnp.squeeze(a, axis=axis) for a in jnp.split(ary, n, axis=axis)]

class DynamicWeightProjection(nn.Module):
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  n_splits: int = None
  num_heads: int = 0
  num_groups: int = 0
  input_dim: int = None
  dynamic_w_init: float = None
  dynamic_d_init: float = None
  dynamic_squeeze_ratio: int = None  # mqy
  decompose_dynamic_w: bool = True
  # dw_activation_cls: activations_lib.BaseActivation = None
  # dw1_norm_cls: normalizations.BaseNormalization = None  # not effective without learned bias # mqy
  dynamic_w_hidden_dim: int = None  # mqy
  # dynamic_d_hidden_dim: int = None
  merge_dynamic_w_hidden: bool = False
  # dw_hidden_activation_cls: activations_lib.BaseActivation = None  # mqy
  deterministic: bool = False
  dynamic_dropout_rate: Optional[float] = None

  def setup(self) -> None:
    self.num_heads_per_group = self.num_heads // self.num_groups
    kwargs = dict(
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      use_bias=False,
      precision=self.precision,
    )
    if self.dynamic_w_init is not None:
      dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio \
        if self.dynamic_squeeze_ratio is not None else 2
      print(f'input_dim: {self.input_dim} dynamic_w_hidden_dim: {self.dynamic_w_hidden_dim}')
      self.dw1 = DenseGeneral(features=(self.num_groups, self.n_splits, self.dynamic_w_hidden_dim),
        kernel_init=initializers.normal(math.sqrt(2.0 / (self.input_dim + self.dynamic_w_hidden_dim))), **kwargs)
      self.dw_hidden_activation = nn.gelu

      G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
      I = dynamic_hidden_dim * 2
      # if not self.decompose_dynamic_w: I = M
      shape = [G, self.n_splits, K, I, M]
      # self.qkw = DenseGeneral(axis=(-3, -2, -1), features=shape[3:],
      #   kernel_init=initializers.normal(self.dynamic_w_init), **kwargs)
      self.qkw = self.param('qkw', initializers.normal(self.dynamic_w_init), shape, self.param_dtype)
  
    if self.dynamic_d_init is not None:
      self.dd = DenseGeneral(features=(self.num_groups, self.num_heads_per_group * self.n_splits),
        kernel_init=initializers.normal(self.dynamic_d_init), **kwargs)

    self.dw_activation = nn.tanh
    self.dw1_norm = nn.RMSNorm(use_scale=False, **{k: v for k, v in kwargs.items() if k not in ['use_bias', 'precision']})

    if self.dynamic_dropout_rate is not None:
      self.dropout = nn.Dropout(self.dynamic_dropout_rate)

  def __call__(self, query_vec):
    print(f'dynamic_dropout_rate: {self.dynamic_dropout_rate}')
    if self.n_splits == 2:
      dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))   # BTG2,64
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)  # XD may add
      # w1, w2 = jnp.split(self.qkw(dw_hidden), 2, axis=-2)
      w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), 2, axis=-2)
      w1 = self.dw1_norm(w1)
      # w2 = self.dw_activation(w2)
      pre_w1, post_w1 = unbind(w1, 2, axis=3) # BTG2IM->[BTGIM]*2
      pre_w2, post_w2 = unbind(w2, 2, axis=3)

      dd = self.dd(query_vec) # jnp.einsum('BTD,DGM->BTGM', query_vec, theta.dd)
      dd = self.dw_activation(dd)
      if self.dynamic_dropout_rate is not None:
        dd = self.dropout(dd, deterministic=self.deterministic)  # XD may add
      pre_dd, post_dd = jnp.split(dd, 2, axis=-1)
      return (pre_w1, pre_w2, pre_dd), (post_w1, post_w2, post_dd)
    else:
      # dw_hidden = jnp.einsum('BTD,DGCK->BTGCK', query_vec, theta.dw1)  # C=4 [pre,post]*[query,key]
      # w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, theta.qkw), 2, axis=-2)
      dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)  # XD may add
      # w1, w2 = jnp.split(self.qkw(dw_hidden), 2, axis=-2)
      w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), 2, axis=-2)
      w1 = self.dw1_norm(w1)
      # w2 = self.dw_activation(w2)
      pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, axis=3) # BTG4IM->[BTGIM]*4
      pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, axis=3)

      dd = self.dd(query_vec) # jnp.einsum('BTD,DGM->BTGM', query_vec, theta.dd)
      dd = self.dw_activation(dd)
      if self.dynamic_dropout_rate is not None:
        dd = self.dropout(dd, deterministic=self.deterministic)  # XD may add
      pre_qdd, pre_kdd, post_qdd, post_kdd = jnp.split(dd, 4, axis=-1)
      return (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd), \
        (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

class CrossHeadProjection(nn.Module):
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None

  num_heads: int = 0
  num_groups: int = 0
  relative_scale: float = 0.1
  use_static_w: bool = True
  loop_over_dynamic_hd: bool = True
  tgt_dependent: bool = True
  src_dependent: bool = True

  def setup(self) -> None:
    self.num_heads_per_group = self.num_heads // self.num_groups
    kwargs = dict(
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      use_bias=False,
      precision=self.precision,
    )
    # self.w = DenseGeneral(axis=(1, 2), features=(self.num_heads_per_group,),  # BGMTS,GMN->BGNTS
    #   kernel_init=initializers.normal(math.sqrt(1. / self.num_heads_per_group) * self.relative_scale), **kwargs)
    shape = (self.num_groups, self.num_heads_per_group, self.num_heads_per_group)
    self.w = self.param('w', initializers.normal(math.sqrt(1. / self.num_heads_per_group) * self.relative_scale), shape, self.param_dtype)

  def __call__(self, inputs, qw1 = None, qw2 = None, kw1 = None, kw2 = None, qdd = None, kdd = None):
    shape = inputs.shape
    assert inputs.shape[1] == self.num_heads
    inputs = rearrange(inputs, 'B (G M) T S -> B G M T S', G=self.num_groups)
    inputs_label = 'BGMTS'

    ret = inputs
    # ret += self.w(inputs)  # BGMTS,GMN->BGNTS
    ret += jnp.einsum('BGMTS,GMN->BGNTS', inputs, self.w)

    if qw1 is not None:
      hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I')
      for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
        dw_label = f'B{sym}G{hidden_sym}M' if w1.shape[-1] == self.num_heads_per_group \
          else f'B{sym}GM{hidden_sym}'  # w1.shape[-2] == self.num_heads_per_group
        dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
        eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
        eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
        if sym == 'T' and self.tgt_dependent or sym == 'S' and self.src_dependent:
          if self.loop_over_dynamic_hd and dynamic_hidden_dim <= 2:
            for i in range(dynamic_hidden_dim):
              if dw_label[-1] == hidden_sym:
                hidden = jnp.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i])
                out = jnp.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i])
              else:
                assert dw_label[-2] == hidden_sym, dw_label
                hidden = jnp.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :])
                out = jnp.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :])
              ret = ret + out
          else:
            hidden = jnp.einsum(eqn1, inputs, w1)
            if self.decompose_dynamic_w:
              out = jnp.einsum(eqn2, hidden, w2)
              ret = ret + out
            else:
              ret = ret + hidden

    if qdd is not None:
      for sym, dd in zip(['T', 'S'], [qdd, kdd]):
        dd_label = f'B{sym}GM'
        if sym == 'T' and self.tgt_dependent or sym == 'S' and self.src_dependent or \
              not self.tgt_dependent and not self.src_dependent:
          dout = jnp.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd)
          ret = ret + dout
    return jnp.reshape(ret, shape)  # BGMTS->BNTS

class MultiHeadDotProductAttention(nn.Module):
  num_heads: int
  num_groups: int = 1
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[
    [PRNGKey, Shape, Dtype], Array
  ] = nn.zeros_init()
  use_bias: bool = True
  decode: bool = False
  # normalize_qk: bool = False
  dynamic_compose: bool = True  # XD
  is_cross_attention: bool = False  # XD
  dynamic_dropout_rate: float = None

  def setup(self):
    self.head_dim = self.qkv_features // self.num_heads
    kwargs = dict(
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
    )

    self.query_dense = DenseGeneral(features=(self.num_heads, self.head_dim), **kwargs)
    self.key_dense = DenseGeneral(features=(self.num_heads, self.head_dim), **kwargs)
    self.value_dense = DenseGeneral(features=(self.num_heads, self.head_dim), **kwargs)
    self.o_dense = nn.Dense(features=self.qkv_features, **kwargs)

    if self.dynamic_compose:
      input_dim = self.num_heads * self.head_dim
      I = 2
      num_heads_per_group = self.num_heads // self.num_groups
      dynamic_w_hidden_dim = num_heads_per_group * I * 2
      if self.is_cross_attention:
        for name in ['q_dyn_w_proj', 'k_dyn_w_proj']:
          setattr(self, name, DynamicWeightProjection(
            num_heads=self.num_heads, num_groups=self.num_groups,
            input_dim=self.num_heads * self.head_dim, n_splits=2,
            dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
            dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
            dynamic_squeeze_ratio=num_heads_per_group // I,
            dynamic_w_hidden_dim=dynamic_w_hidden_dim,
            dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
            deterministic=self.deterministic,
            dynamic_dropout_rate=self.dynamic_dropout_rate,
          ))
      else:
        self.dyn_w_proj = DynamicWeightProjection(
          num_heads=self.num_heads, num_groups=self.num_groups,
          input_dim=self.num_heads * self.head_dim, n_splits=4,
          dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
          dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
          dynamic_squeeze_ratio=num_heads_per_group // I,
          dynamic_w_hidden_dim=dynamic_w_hidden_dim,
          dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
          deterministic=self.deterministic,
          dynamic_dropout_rate=self.dynamic_dropout_rate,
        )
      for name in ['pre_proj', 'post_proj']:
        setattr(self, name, CrossHeadProjection(
          num_heads=self.num_heads, num_groups=self.num_groups,
          dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
        ))

  def dot_product_attention(
    self,
    query: Array,
    key: Array,
    value: Array,
    inputs_q: Optional[Array] = None,  # XD
    inputs_k: Optional[Array] = None,  # XD
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nn.Module] = None,
  ):
   # 所有参数转为dtype类型
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype
    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
    assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'
    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision
    )
    if self.dynamic_compose:  # XD
      if self.is_cross_attention:
        (pre_qw1, pre_qw2, pre_qdd), (post_qw1, post_qw2, post_qdd) = self.q_dyn_w_proj(inputs_q)
        (pre_kw1, pre_kw2, pre_kdd), (post_kw1, post_kw2, post_kdd) = self.k_dyn_w_proj(inputs_k)
      else:
        (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd), \
        (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd) = self.dyn_w_proj(inputs_q)
      attn_weights = self.pre_proj(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
      attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
      big_neg = jnp.finfo(dtype).min
      attn_weights = jnp.where(mask, attn_weights, big_neg)
    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
    if self.dynamic_compose:  # XD
      attn_weights = self.post_proj(attn_weights, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
    if module:
      module.sow('intermediates', 'attention_weights', attn_weights)
    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
      keep_prob = 1.0 - dropout_rate
      if broadcast_dropout:
        # dropout is broadcast across the batch + head dimensions
        dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
      else:
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
      multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
      attn_weights = attn_weights * multiplier

    outputs = jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision
    )
    return outputs

  @overload
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    ...

  @overload
  def __call__(
    self,
    inputs_q: Array,
    *,
    inputs_kv: Array = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    ...

  @nn.compact
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    inputs_kv: Optional[Array] = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
          'If either `inputs_k` or `inputs_v` is not None, '
          '`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` '
          'and `inputs_v` must be None. We recommend using `inputs_k` and '
          '`inputs_v` args, since `inputs_kv` will be deprecated soon. See '
          'https://github.com/google/flax/discussions/3389 for more '
          'information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
        'The inputs_kv arg will be deprecated soon. '
        'Use inputs_k and inputs_v instead. See '
        'https://github.com/google/flax/discussions/3389 '
        'for more information.',
        DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
            '`inputs_k` cannot be None if `inputs_v` is not None. '
            'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
            'value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
          f'You are passing an array of shape {inputs_v.shape} '
          'to the `inputs_v` arg, when you may have intended '
          'to pass it to the `mask` arg. As of Flax version '
          '0.7.4, the function signature of '
          "MultiHeadDotProductAttention's `__call__` method "
          'has changed to `__call__(inputs_q, inputs_k=None, '
          'inputs_v=None, *, inputs_kv=None, mask=None, '
          'deterministic=None)`. Use the kwarg `mask` instead. '
          'See https://github.com/google/flax/discussions/3389 '
          'and read the docstring for more information.',
          DeprecationWarning,
        )
    bsz, length, model_dim = inputs_q.shape
    # features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query = self.query_dense(inputs_q)
    key = self.key_dense(inputs_k)
    value = self.value_dense(inputs_v)
    # query = query.reshape(bsz, length, self.num_heads, self.head_dim)
    # key = key.reshape(bsz, length, self.num_heads, self.head_dim)
    # value = value.reshape(bsz, length, self.num_heads, self.head_dim)

    # if self.normalize_qk:
    if self.dynamic_compose:  # XD
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = LayerNorm(name='query_ln', use_bias=False)(query)  # type: ignore[call-arg]
      key = LayerNorm(name='key_ln', use_bias=False)(key)  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
          *batch_dims,
          max_length,
          num_heads,
          depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            'Autoregressive cache shape error, '
            'expected query shape %s instead got %s.'
            % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
        indices: tuple[Union[int, jax.Array], ...] = (zero,) * len(batch_dims) + (
          cur_index,
          zero,
          zero,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = nn.combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(max_length) <= cur_index,
            tuple(batch_dims) + (1, 1, max_length),
          ),
        )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = nn.merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True
    # bsz * lenght * n * head_dim
    x = self.dot_product_attention(
        query,
        key,
        value,
        inputs_q=inputs_q,  # XD
        inputs_k=inputs_k,  # XD
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        module=self if sow_weights else None,
      )  # pytype: disable=wrong-keyword-args
    x = x.reshape(bsz, length, -1)
    out = self.o_dense(x)
    return out


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    out_dim: optionally specify out dimension.
  """

  config: TransformerConfig
  out_dim: Optional[int] = None

  @nn.compact
  def __call__(self, inputs):
    """Applies Transformer MlpBlock module."""
    config = self.config
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        config.mlp_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(inputs)
    x = nn.relu(x)
    x = nn.Dropout(rate=config.dropout_rate)(
        x, deterministic=config.deterministic
    )
    output = nn.Dense(
        actual_out_dim,
        dtype=config.dtype,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
    )(x)
    output = nn.Dropout(rate=config.dropout_rate)(
        output, deterministic=config.deterministic
    )
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig = None

  def setup(self):
    config = self.config
    self.pre_norm = nn.LayerNorm(dtype=config.dtype)

    self.dot_attn = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
        dynamic_compose=config.en_dynamic_compose,
        dynamic_dropout_rate=config.dynamic_dropout_rate
    )
    self.dropout = nn.Dropout(rate=config.dropout_rate)
    self.post_norm = nn.LayerNorm(dtype=self.config.dtype)

    self.mlp = MlpBlock(config=self.config)
    

  @nn.compact
  def __call__(self, inputs, encoder_mask=None):
    """Applies Encoder1DBlock module.

    Args:
      inputs: input data.
      encoder_mask: encoder self-attention mask.

    Returns:
      output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = self.pre_norm(inputs)
    x = self.dot_attn(x, mask=encoder_mask)
    x = self.dropout(x, deterministic=self.config.deterministic)
    x = x + inputs

    # MLP block.
    y = self.post_norm(x)
    y = self.mlp(y)

    return x + y


class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  def setup(self):
    config = self.config
    self.pre_decoder_norm = nn.LayerNorm(dtype=config.dtype)
    self.decoder_dot_attn = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
        decode=config.decode,
        dynamic_compose=config.de_dynamic_compose1, # # True
        dynamic_dropout_rate=config.dynamic_dropout_rate
    )

    self.encoder_decoder_dot_attn = MultiHeadDotProductAttention(
        num_heads=config.num_heads,
        dtype=config.dtype,
        qkv_features=config.qkv_dim,
        kernel_init=config.kernel_init,
        bias_init=config.bias_init,
        use_bias=False,
        broadcast_dropout=False,
        dropout_rate=config.attention_dropout_rate,
        deterministic=config.deterministic,
        is_cross_attention=True,  # XD
        dynamic_compose=config.de_dynamic_compose2, # True
        dynamic_dropout_rate=config.dynamic_dropout_rate,
    )

    self.dropout = nn.Dropout(rate=config.dropout_rate)
    self.pre_endocer_decoder_norm = nn.LayerNorm(dtype=config.dtype)

    self.pre_mlp_norm = nn.LayerNorm(dtype=config.dtype)
    self.mlp = MlpBlock(config=config)
     
  @nn.compact
  def __call__(
      self, targets, encoded, decoder_mask=None, encoder_decoder_mask=None
  ):
    """Applies EncoderDecoder1DBlock module.

    Args:
      targets: input data for decoder
      encoded: input data from encoder
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output after transformer encoder-decoder block.
    """
    config = self.config
    # Decoder block.
    assert targets.ndim == 3
    x = self.pre_decoder_norm(targets)
    x = self.decoder_dot_attn(x, mask=decoder_mask)
    x = self.dropout(x, deterministic=config.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = self.pre_endocer_decoder_norm(x)
    y = self.encoder_decoder_dot_attn(y, encoded, mask=encoder_decoder_mask)
    y = self.dropout(y, deterministic=config.deterministic)
    y = y + x

    # MLP block.
    z = self.pre_mlp_norm(y)
    z = self.mlp(z)

    return y + z


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """

  config: TransformerConfig
  shared_embedding: Any = None

  def setup(self):
    if self.shared_embedding is None:
      self.input_embed = nn.Embed(
          num_embeddings=self.config.vocab_size,
          features=self.config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
      )
    else:
      self.input_embed = self.shared_embedding

    self.position_embedding = AddPositionEmbs(config=self.config, decode=False, name='posembed_input')
    self.net = [Encoder1DBlock(config=self.config, name=f'encoderblock_{lyr}') for lyr in range(self.config.num_layers)]
    self.dropout = nn.Dropout(rate=self.config.dropout_rate)
    self.final_ln = nn.LayerNorm(dtype=self.config.dtype, name='encoder_norm')

  @nn.compact
  def __call__(self, inputs, inputs_positions=None, encoder_mask=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data
      inputs_positions: input subsequence positions for packed examples.
      encoder_mask: decoder self-attention mask.

    Returns:
      output of a transformer encoder.
    """
    config = self.config
    assert inputs.ndim == 2  # (batch, len)
    x = inputs.astype('int32')
    # word embedding
    x = self.input_embed(x)
    # position embedding
    x = self.position_embedding(x, inputs_positions=inputs_positions)
    x = self.dropout(x, deterministic=config.deterministic)
    x = x.astype(config.dtype)

    # Input Encoder
    for lyr in range(config.num_layers):
      x = self.net[lyr](x, encoder_mask)

    encoded = self.final_ln(x)

    return encoded


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
    shared_embedding: a shared embedding layer to use.
  """

  config: TransformerConfig
  shared_embedding: Any = None

  def setup(self):
    config = self.config
    if self.shared_embedding is None:
      self.output_embed = nn.Embed(
          num_embeddings=config.output_vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
      )
    else:
      self.output_embed = self.shared_embedding

    self.position_embedding = AddPositionEmbs(config=config, decode=config.decode, name='posembed_output')
    self.dropout = nn.Dropout(rate=self.config.dropout_rate)
    self.net = [EncoderDecoder1DBlock(config=config, name=f'encoderdecoderblock_{lyr}') for lyr in range(self.config.num_layers)]
    self.final_ln = nn.LayerNorm(dtype=config.dtype, name='encoderdecoder_norm')

    self.lm_head = nn.Dense(
          config.output_vocab_size,
          dtype=config.dtype,
          kernel_init=config.kernel_init,
          bias_init=config.bias_init,
          name='logitdense',
      )
    
  @nn.compact
  def __call__(
      self,
      encoded,
      targets,
      targets_positions=None,
      decoder_mask=None,
      encoder_decoder_mask=None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      encoded: encoded input data from encoder.
      targets: target inputs.
      targets_positions: input subsequence positions for packed examples.
      decoder_mask: decoder self-attention mask.
      encoder_decoder_mask: encoder-decoder attention mask.

    Returns:
      output of a transformer decoder.
    """
    config = self.config

    assert encoded.ndim == 3  # (batch, len, depth)
    assert targets.ndim == 2  # (batch, len)

    y = targets.astype('int32')
    if not config.decode:
      y = shift_right(y)
    y = self.output_embed(y)
    y = self.position_embedding(y, inputs_positions=targets_positions)
    y = self.dropout(y, deterministic=config.deterministic)

    y = y.astype(config.dtype)

    # Target-Input Decoder
    for lyr in range(config.num_layers):
      y = self.net[lyr](y, encoded, decoder_mask=decoder_mask, encoder_decoder_mask=encoder_decoder_mask)

    y = self.final_ln(y)

    # Decoded Logits, True
    if config.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = self.lm_head(y)
    return logits


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation.

  Attributes:
    config: TransformerConfig dataclass containing hyperparameters.
  """

  config: TransformerConfig

  def setup(self):
    config = self.config

    if config.share_embeddings:
      if config.output_vocab_size is not None:
        assert (
            config.output_vocab_size == config.vocab_size
        ), "can't share embedding with different vocab sizes."
      self.shared_embedding = nn.Embed(
          num_embeddings=config.vocab_size,
          features=config.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0),
      )
    else:
      self.shared_embedding = None

    self.encoder = Encoder(
        config=config, shared_embedding=self.shared_embedding
    )
    self.decoder = Decoder(
        config=config, shared_embedding=self.shared_embedding
    )

  def encode(self, inputs, inputs_positions=None, inputs_segmentation=None):
    """Applies Transformer encoder-branch on the inputs.

    Args:
      inputs: input data.
      inputs_positions: input subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.

    Returns:
      encoded feature array from the transformer encoder.
    """
    config = self.config
    # Make padding attention mask.
    encoder_mask = nn.make_attention_mask(
        inputs > 0, inputs > 0, dtype=config.dtype
    )
    # Add segmentation block-diagonal attention mask if using segmented data.
    if inputs_segmentation is not None:
      encoder_mask = nn.combine_masks(
          encoder_mask,
          nn.make_attention_mask(
              inputs_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=config.dtype,
          ),
      )
    return self.encoder(
        inputs, inputs_positions=inputs_positions, encoder_mask=encoder_mask
    )

  def decode(
      self,
      encoded,
      inputs,  # only needed for masks
      targets,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None,
  ):
    """Applies Transformer decoder-branch on encoded-input and target.

    Args:
      encoded: encoded input data from encoder.
      inputs: input data (only needed for masking).
      targets: target data.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from transformer decoder.
    """
    config = self.config

    # Make padding attention masks.
    if config.decode:
      # for fast autoregressive decoding only a special encoder-decoder mask is used
      decoder_mask = None
      encoder_decoder_mask = nn.make_attention_mask(
          jnp.ones_like(targets) > 0, inputs > 0, dtype=config.dtype
      )
    else:
      decoder_mask = nn.combine_masks(
          nn.make_attention_mask(targets > 0, targets > 0, dtype=config.dtype),
          nn.make_causal_mask(targets, dtype=config.dtype),
      )
      encoder_decoder_mask = nn.make_attention_mask(
          targets > 0, inputs > 0, dtype=config.dtype
      )

    # Add segmentation block-diagonal attention masks if using segmented data.
    if inputs_segmentation is not None:
      decoder_mask = nn.combine_masks(
          decoder_mask,
          nn.make_attention_mask(
              targets_segmentation,
              targets_segmentation,
              jnp.equal,
              dtype=config.dtype,
          ),
      )
      encoder_decoder_mask = nn.combine_masks(
          encoder_decoder_mask,
          nn.make_attention_mask(
              targets_segmentation,
              inputs_segmentation,
              jnp.equal,
              dtype=config.dtype,
          ),
      )
    logits = self.decoder(
        encoded,
        targets,
        targets_positions=targets_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
    )
    return logits.astype(self.config.dtype)

  def __call__(
      self,
      inputs,
      targets,
      inputs_positions=None,
      targets_positions=None,
      inputs_segmentation=None,
      targets_segmentation=None,
  ):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data.
      targets: target data.
      inputs_positions: input subsequence positions for packed examples.
      targets_positions: target subsequence positions for packed examples.
      inputs_segmentation: input segmentation info for packed examples.
      targets_segmentation: target segmentation info for packed examples.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation,
    )

    return self.decode(
        encoded,
        inputs,  # only used for masks
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
    )
