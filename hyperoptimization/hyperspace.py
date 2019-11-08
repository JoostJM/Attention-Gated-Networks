import pandas as pd
import six

class HyperSpace:
  def __init__(self, hyperspace):
    hyperspace_enum = self._enumerate_hyperspace(hyperspace)
    self.hyperspace = pd.DataFrame(data=hyperspace_enum)
    self.hyperspace.index = pd.Index(range(1, self.hyperspace.shape[0] + 1))
    self.results = None

  def save_space(self, fname):
    hyperspace = self.hyperspace
    if self.results is not None:
      hyperspace = hyperspace.join(self.results.T)
    hyperspace.to_csv(fname)

  def add_result(self, result):
    if self.results is None:
      self.results = pd.DataFrame()

    for cfg_idx, cfg_result in six.iteritems(result):
      results_series = pd.Series()
      results_series.name = cfg_idx
      for split, split_result in six.iteritems(cfg_result):
        for k, v in six.iteritems(split_result):
          results_series['%s_%s' % (split, k)] = v

      self.results = self.results.join(results_series, how='outer')

  @classmethod
  def _enumerate_hyperspace(cls, hyperspace):
    maxBatchSize = hyperspace['maxBatchSize']

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
          elif k == 'batchSize':
            p_opts.append(cls.compute_batch_size(v_i, maxBatchSize))
          else:
            p_opts.append({k: v_i})

        if opts is None:
          opts = p_opts
        else:
          opts = [merge_dicts(o, p) for o in opts for p in p_opts]

      return opts

    return enum(hyperspace)

  @staticmethod
  def compute_batch_size(batchSize, maxBatchSize):
    assert maxBatchSize >= 1, 'maxBatchSize cannot be smaller than 1'
    accumulate_iter = 1
    if batchSize > maxBatchSize:
      b_i = batchSize / accumulate_iter
      b_m = batchSize % accumulate_iter
      while b_i > maxBatchSize | b_m != 0:
        accumulate_iter += 1
        b_i = batchSize / accumulate_iter
        b_m = batchSize % accumulate_iter

      return {
        'batchSize': int(b_i),
        'accumulate_iter': accumulate_iter
      }
    else:
      return {
        'batchSize': batchSize,
        'accumulate_iter': accumulate_iter
      }
