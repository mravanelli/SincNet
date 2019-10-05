# compute_d_vector.py
# Mirco Ravanelli
# Mila - University of Montreal

# Feb 2019

# Description:
# This code computes d-vectors using a pre-trained model

import collections

import numpy as np
import torch

from data_io import read_conf_inp
from data_io import str_to_bool
from dnn_models import MLP
from dnn_models import SincNet as CNN

# Model to use for computing the d-vectors
model_file = '/home/mirco/sincnet_models/SincNet_TIMIT/model_raw.pkl'  # This is the model to use for computing the d-vectors (it should be pre-trained using the speaker-id DNN)
cfg_file = '/home/mirco/SincNet/cfg/SincNet_TIMIT.cfg'  # Config file of the speaker-id experiment used to generate the model
te_lst = 'data_lists/TIMIT_test.scp'  # List of the wav files to process
out_dict_file = 'd_vect_timit.npy'  # output dictionary containing the a sentence id as key as the d-vector as value
data_folder = '/home/mirco/Dataset/TIMIT_norm_nosil'

avoid_small_en_fr = True
energy_th = 0.1  # Avoid frames with an energy that is 1/10 over the average energy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Reading cfg file
options = read_conf_inp(cfg_file)

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))


# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

BATCH_SIZE = 128

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop}

CNN_net = CNN(CNN_arch)
CNN_net.to(device)

DNN1_arch = {'input_dim': CNN_net.out_dim,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act}

DNN1_net = MLP(DNN1_arch)
DNN1_net.to(device)

checkpoint_load = torch.load(model_file, map_location=device)
model_trained_using_data_parallel = False
if model_trained_using_data_parallel:
  new_ckpt = {}
  for k, v in checkpoint_load.items():
    new_v = collections.OrderedDict()
    for kk, vv in v.items():
      if kk.startswith('module.'):
        kk = '.'.join(kk.split('.')[1:])
      else:
        assert False
      new_v[kk] = vv
    new_ckpt[k] = new_v
  checkpoint_load = new_ckpt
CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])

CNN_net.eval()
DNN1_net.eval()

d_vector_dim = fc_lay[-1]

def audio_samples_to_d_vectors(signal: np.ndarray):
  with torch.no_grad():
    # Amplitude normalization
    signal = signal / np.max(np.abs(signal))

    signal = torch.from_numpy(signal).float().to(device).contiguous()

    if avoid_small_en_fr:
      # computing energy on each frame
      en_N_fr_actual = 0
      en_N_fr = (signal.shape[0] - wlen) // wshift + 1
      en_arr = torch.zeros([en_N_fr]).float().cuda(device).contiguous()
      for i_sig, beg_samp in enumerate(range(0, signal.shape[0], wshift)):
        end_samp = beg_samp + wlen
        if end_samp > signal.shape[0]:
          break
        else:
          en_arr[i_sig] = torch.sum(signal[beg_samp:end_samp].pow(2)).item()
          en_N_fr_actual += 1
      assert en_N_fr == en_N_fr_actual

      en_arr_bin = en_arr > torch.mean(en_arr) * 0.1
      en_arr_bin.to(device)
      n_vect_elem = torch.sum(en_arr_bin)

      if n_vect_elem < 10:
        raise Exception('Low energy')

    sig_arr = []
    d_vectors = []
    for beg_samp in range(0, signal.shape[0], wshift):
      end_samp = beg_samp + wlen
      if end_samp > signal.shape[0]:
        break
      else:
        sig_arr.append(torch.unsqueeze(signal[beg_samp:end_samp], dim=0))
        if len(sig_arr) == BATCH_SIZE:
          out = DNN1_net(CNN_net(torch.cat(sig_arr, dim=0)))
          d_vectors.append(out)
          sig_arr = []
    if sig_arr:
      out = DNN1_net(CNN_net(torch.cat(sig_arr, dim=0)))
      d_vectors.append(out)
    if len(d_vectors) == 0:
      raise Exception('Empty d-vectors')
    d_vectors = torch.cat(d_vectors, dim=0)

    if avoid_small_en_fr:
      d_vectors = d_vectors.index_select(0, (en_arr_bin == 1).nonzero().view(-1))

    return d_vectors.cpu().numpy()


def normalize_d_vectors(d_vectors):
  with torch.no_grad():
    # averaging and normalizing all the d-vectors
    d_vectors = torch.from_numpy(d_vectors).to(device)
    d_vector_out = torch.mean(d_vectors / d_vectors.norm(p=2, dim=1).view(-1, 1), dim=0)

    # checks for nan
    nan_sum = torch.sum(torch.isnan(d_vector_out))

    if nan_sum > 0:
      return Exception('NaN encountered when normalizing d-vectors')
    else:
      return d_vector_out.cpu().numpy()
