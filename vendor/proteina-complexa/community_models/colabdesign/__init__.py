import functools
import os,jax
import types
import jax.numpy as jnp

if not hasattr(jax, "tree_map"):
  jax.tree_map = jax.tree_util.tree_map
if not hasattr(jax, "tree_leaves"):
  jax.tree_leaves = jax.tree_util.tree_leaves
if not hasattr(jax, "tree_flatten"):
  jax.tree_flatten = jax.tree_util.tree_flatten
if not hasattr(jax, "tree_unflatten"):
  jax.tree_unflatten = jax.tree_util.tree_unflatten
def _jax_util_wraps(wrapped, assigned=functools.WRAPPER_ASSIGNMENTS,
                    updated=functools.WRAPPER_UPDATES, docstr=None):
  if docstr is not None:
    assigned = tuple(x for x in assigned if x != "__doc__")
    def _decorator(wrapper):
      wrapper = functools.wraps(wrapped, assigned=assigned, updated=updated)(wrapper)
      wrapper.__doc__ = docstr
      return wrapper
    return _decorator
  return functools.wraps(wrapped, assigned=assigned, updated=updated)

if not hasattr(jax, "util"):
  jax.util = types.SimpleNamespace(wraps=_jax_util_wraps)
else:
  jax.util.wraps = _jax_util_wraps
_orig_clip = jnp.clip
def _compat_clip(a, a_min=None, a_max=None, *args, **kwargs):
  if "min" not in kwargs and a_min is not None:
    kwargs["min"] = a_min
  if "max" not in kwargs and a_max is not None:
    kwargs["max"] = a_max
  return _orig_clip(a, *args, **kwargs)
jnp.clip = _compat_clip
# disable triton_gemm for jax versions > 0.3
if int(jax.__version__.split(".")[1]) > 3:
  os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .shared.utils import clear_mem
from .af.model import mk_af_model
from .tr.model import mk_tr_model
from .mpnn.model import mk_mpnn_model

# backward compatability
mk_design_model = mk_afdesign_model = mk_af_model
mk_trdesign_model = mk_tr_model
