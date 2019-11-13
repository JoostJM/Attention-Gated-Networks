import logging
import os
import re

import numpy as np
import pandas as pd
import six


class HyperSpace:
  def __init__(self, hyperspace, out_dir):
    self.logger = logging.getLogger('hyperspace')
    hyperspace_enum = self._enumerate_hyperspace(hyperspace)
    self.hyperspace = pd.DataFrame(data=hyperspace_enum)
    self.hyperspace.index = pd.Index(range(1, self.hyperspace.shape[0] + 1))
    self.results = None
    self.out_dir = out_dir
    self.fname = os.path.join(out_dir, 'hyperspace.csv')

  def __iter__(self):
    if self.results is not None:
      return self.hyperspace.loc[self.hyperspace.index.difference(self.results.columns), :].iterrows()
    else:
      return self.hyperspace.iterrows()

  def __len__(self):
    if self.results is not None:
      return len(self.hyperspace.index.difference(self.results.columns))
    else:
      return self.hyperspace.shape[0]

  def save_space(self):
    hyperspace = self.hyperspace
    if self.results is not None:
      hyperspace = hyperspace.join(self.results.T)
    hyperspace.to_csv(self.fname)

  def load_results(self, rename_folders=True):
    self.logger.info('Loading results from %s', self.fname)

    h_file = pd.read_csv(self.fname, index_col=0)

    # Check that all parameter columns in current hyperspace also exist in saved hyperspace
    assert len(self.hyperspace.columns.difference(h_file.columns)) == 0, "Defined parameters don't match"

    # Check that all other columns in saved hyperspace are results columns (start with train/validation/test)
    col_pattern = re.compile(r'(train)|(validation)|(test)')
    assert np.all([col_pattern.match(c) is not None for c in h_file.columns.difference(self.hyperspace.columns)]), \
        'Saved hyperspace contains more parameter columns than current hyperspace!'

    # If the configuration file was changed, indices may not match up.
    # Therefore, match on the hyperspace itself by setting the hyperspace as index
    h_indices = pd.Series(h_file.index)

    h_file[self.hyperspace.columns] = h_file[self.hyperspace.columns].applymap(str)
    h_file = h_file.set_index(list(self.hyperspace.columns))

    h_indices.index = h_file.index
    h_indices.name = 'old_idx'

    current_str_params = self.hyperspace.applymap(str)
    current_str_params = current_str_params.set_index(list(current_str_params.columns))
    current_str_params['current_idx'] = self.hyperspace.index

    # Only retain results for which hyperspaces have been defined
    h_results = h_file.join(current_str_params, how='inner')
    h_results = h_results.set_index('current_idx')
    # Drop hyperspaces without a result
    h_results = h_results[h_results.notnull().all(axis=1)]

    if rename_folders:
      index_map = current_str_params.join(h_indices, how='inner')
      index_map = index_map.set_index('old_idx')['current_idx']
      h_map_pattern = re.compile(r'\d+')

      h_maps = [f for f in os.listdir(self.out_dir) if h_map_pattern.fullmatch(f)]
      renames = []
      for m in h_maps:
        if index_map.get(int(m), None) == int(m):
          continue  # no renaming needed

        renames.append(m)
        fldr = os.path.join(self.out_dir, m)
        os.rename(fldr, fldr + '_')

      for m in renames:
        new_fldr = index_map.get(int(m), None)
        if new_fldr is not None and new_fldr != int(m):
          new_fldr = '%.3i' % new_fldr
          self.logger.info('Renaming folder %s to %s', m, new_fldr)
          os.rename(os.path.join(self.out_dir, m) + '_', os.path.join(self.out_dir, new_fldr))

    self.results = h_results.T


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
