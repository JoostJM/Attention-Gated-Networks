from .unet_2D import *
from .unet_3D import *
from .unet_nonlocal_2D import *
from .unet_nonlocal_3D import *
from .unet_grid_attention_3D import *
from .unet_CT_dsv_3D import *
from .unet_CT_single_att_dsv_3D import *
from .unet_CT_multi_att_dsv_3D import *
from .sononet import *
from .sononet_grid_attention import *


def get_network(architecture, output_nc, input_nc=3, feature_scale=4, tensor_dim='2D',
                nonlocal_mode='embedded_gaussian', attention_dsample=(2, 2, 2),
                aggregation_mode='concat', **kwargs):
  model = _get_model_instance(architecture, tensor_dim)

  if architecture in ['unet', 'unet_ct_dsv']:
    model = model(n_classes=output_nc,
                  is_batchnorm=True,
                  in_channels=input_nc,
                  feature_scale=feature_scale,
                  is_deconv=False)
  elif architecture in ['unet_nonlocal']:
    model = model(n_classes=output_nc,
                  is_batchnorm=True,
                  in_channels=input_nc,
                  is_deconv=False,
                  nonlocal_mode=nonlocal_mode,
                  feature_scale=feature_scale)
  elif architecture in ['unet_grid_gating',
                        'unet_ct_single_att_dsv',
                        'unet_ct_multi_att_dsv']:
    model = model(n_classes=output_nc,
                  is_batchnorm=True,
                  in_channels=input_nc,
                  nonlocal_mode=nonlocal_mode,
                  feature_scale=feature_scale,
                  attention_dsample=attention_dsample,
                  is_deconv=False)
  elif architecture in ['sononet', 'sononet2']:
    model = model(n_classes=output_nc,
                  is_batchnorm=True,
                  in_channels=input_nc,
                  feature_scale=feature_scale)
  elif architecture in ['sononet_grid_attention']:
    model = model(n_classes=output_nc,
                  is_batchnorm=True,
                  in_channels=input_nc,
                  feature_scale=feature_scale,
                  nonlocal_mode=nonlocal_mode,
                  aggregation_mode=aggregation_mode)
  else:
    raise 'Model {} not available'.format(architecture)

  return model


def _get_model_instance(name, tensor_dim):
  return {
    'unet': {'2D': unet_2D, '3D': unet_3D},
    'unet_nonlocal': {'2D': unet_nonlocal_2D, '3D': unet_nonlocal_3D},
    'unet_grid_gating': {'3D': unet_grid_attention_3D},
    'unet_ct_dsv': {'3D': unet_CT_dsv_3D},
    'unet_ct_single_att_dsv': {'3D': unet_CT_single_att_dsv_3D},
    'unet_ct_multi_att_dsv': {'3D': unet_CT_multi_att_dsv_3D},
    'sononet': {'2D': sononet},
    'sononet2': {'2D': sononet2},
    'sononet_grid_attention': {'2D': sononet_grid_attention}
  }[name][tensor_dim]
