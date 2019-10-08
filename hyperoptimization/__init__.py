from copy import deepcopy
import six


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


def enumerate_hyperspace(hyperspace):
  def merge_dicts(d, o):
    z = d.copy()
    z.update(o)
    return z

  def enum(h):
    opts = None
    for k, v in six.iteritems(h['parameters']):
      p_opts = []
      for v_i in v:
        if isinstance(v_i, dict) and 'parameters' in v_i:
          for s_v in enum(v_i):
            p_opts.append({
              k: {
                "name": v_i['name'],
                "parameters": s_v
              }
            })
        else:
          p_opts.append({k: v_i})

      if opts is None:
        opts = p_opts
      else:
        opts = [merge_dicts(o, p) for o in opts for p in p_opts]

    return opts

  return enum(hyperspace)
