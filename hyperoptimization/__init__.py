from copy import deepcopy
import six

from .hyperspace import HyperSpace


def deep_update(base_dict, other_dict):
  out_dic = deepcopy(base_dict)

  def _update(d, o):
    for k, v in six.iteritems(o):
      if k in d and isinstance(v, dict):
        _update(d[k], v)
      else:
        d[k] = v

    return d

  return _update(out_dic, other_dict)



