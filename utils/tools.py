
import re
import os
import numpy as np
import ctypes
import logging
import oct2py
from mir_eval.separation import bss_eval_sources
from pesq import pesq
import torch
from itertools import permutations
from scipy.io import wavfile
from pystoi import stoi
import config as cfg
import soundfile as sf
from pypesq import pesq
from semetrics import composite

def cal_pesq(clean,generate):
    sr1, clean_wav = wavfile.read(clean)
    sr2, enhanced_wav = wavfile.read(generate)
    score = pesq(clean_wav, enhanced_wav, sr1)
    return score


############################################################################
#                                   MOS                                    #
############################################################################
# Reference
# https://github.com/usimarit/semetrics # https://ecs.utdallas.edu/loizou/speech/software.htm
logging.basicConfig(level=logging.ERROR)
oc = oct2py.Oct2Py(logger=logging.getLogger())

COMPOSITE =  "/home/zmw/projects/ProjectAD/dccrn/matlib_package/composite.m"
print(COMPOSITE)
def composite(clean: str, enhanced: str):
    [csig, cbak, covl, ssnr] = composite(clean,enhanced)
    return csig, cbak, covl, ssnr

def composite_(clean: str, enhanced: str):
    pesq_score = pesq_mos(clean, enhanced)
    csig, cbak, covl, ssnr = oc.feval(COMPOSITE, clean, enhanced, nout=4)
    csig += 0.603 * pesq_score
    cbak += 0.478 * pesq_score
    covl += 0.805 * pesq_score
    return pesq_score,csig, cbak, covl, ssnr


############################################################################
#                                   PESQ                                   #
############################################################################
# Reference
# https://github.com/usimarit/semetrics
# https://ecs.utdallas.edu/loizou/speech/software.htm

def pesq_mos(clean: str, enhanced: str):
    sr1, clean_wav = wavfile.read(clean)
    sr2, enhanced_wav = wavfile.read(enhanced)
    assert sr1 == sr2
    mode = "nb" if sr1 < 16000 else "wb"
    return pesq(clean_wav, enhanced_wav,sr1, False)


###############################################################################
#                           PESQ (another ref)                                #
###############################################################################
# pesq_dll = ctypes.CDLL('./PESQ.so')
# pesq_dll.pesq.restype = ctypes.c_double


# interface to PESQ evaluation, taking in two filenames as input
def run_pesq_filenames(clean, to_eval):
    pesq_regex = re.compile("\(MOS-LQO\):  = ([0-9]+\.[0-9]+)")

    pesq_out = os.popen("./PESQ" + cfg.fs + "wb " + clean + " " + to_eval).read()
    regex_result = pesq_regex.search(pesq_out)

    if (regex_result is None):
        return 0.0
    else:
        return float(regex_result.group(1))


# def run_pesq_waveforms(dirty_wav, clean_wav):
#     clean_wav = clean_wav.astype(np.double)
#     dirty_wav = dirty_wav.astype(np.double)
#     # return pesq(clean_wav, dirty_wav, fs=8000)
#     return pesq_dll.pesq(ctypes.c_void_p(clean_wav.ctypes.data),
#                          ctypes.c_void_p(dirty_wav.ctypes.data),
#                          len(clean_wav),
#                          len(dirty_wav))


# interface to PESQ evaluation, taking in two waveforms as input
# def cal_pesq(dirty_wavs, clean_wavs):
#     scores = []
#     for i in range(len(dirty_wavs)):
#         pesq = run_pesq_waveforms(dirty_wavs[i], clean_wavs[i])
#         scores.append(pesq)
#     return scores


###############################################################################
#                                     STOI                                    #
###############################################################################
def cal_stoi(estimated_speechs, clean_speechs):
    stoi_scores = []
    for i in range(len(estimated_speechs)):
        stoi_score = stoi(clean_speechs[i], estimated_speechs[i], cfg.fs, extended=False)
        stoi_scores.append(stoi_score)
    return stoi_scores


###############################################################################
#                                     SNR                                     #
###############################################################################
def cal_snr(s1, s2, eps=1e-8):
    signal = s2
    mean_signal = np.mean(signal)
    signal_diff = signal - mean_signal
    var_signal = np.sum(np.mean(signal_diff ** 2))  # # variance of orignal data

    noisy_signal = s1
    noise = noisy_signal - signal
    mean_noise = np.mean(noise)
    noise_diff = noise - mean_noise
    var_noise = np.sum(np.mean(noise_diff ** 2))  # # variance of noise

    if var_noise == 0:
        snr_score = 100  # # clean
    else:
        snr_score = (np.log10(var_signal/var_noise + eps))*10
    return snr_score


def cal_snr_array(estimated_speechs, clean_speechs):
    snr_score = []
    for i in range(len(estimated_speechs)):
        snr = cal_snr(estimated_speechs[i], clean_speechs[i])
        snr_score.append(snr)
    return snr_score




###############################################################################
#                                     SDR                                     #
###############################################################################

def SDR(est, egs, mix):
    '''
        calculate SDR
        est: Network generated audio
        egs: Ground Truth
    '''
    length = est.numpy().shape[0]
    sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], est.numpy()[:length])
    mix_sdr, _, _, _ = bss_eval_sources(egs.numpy()[:length], mix.numpy()[:length])
    return float(sdr-mix_sdr)


def permutation_sdr(est_list, egs_list, mix, per):
    n = len(est_list)
    result = sum([SDR(est_list[a], egs_list[b], mix)
                      for a, b in enumerate(per)])/n
    return result



###############################################################################
#                                     SI_SNR                                     #
###############################################################################

def SI_SNR(_s, s, mix, zero_mean=True):
    '''
         Calculate the SNR indicator between the two audios.
         The larger the value, the better the separation.
         input:
               _s: Generated audio
               s:  Ground Truth audio
         output:
               SNR value
    '''
    length = _s.shape[0]
    _s = _s[:length]
    s =s[:length]
    mix = mix[:length]
    if zero_mean:
        _s = _s - torch.mean(_s)
        s = s - torch.mean(s)
        mix = mix - torch.mean(mix)
    s_target = sum(torch.mul(_s, s))*s/(torch.pow(torch.norm(s, p=2), 2)+1e-8)
    e_noise = _s - s_target
    # mix ---------------------------
    mix_target = sum(torch.mul(mix, s))*s/(torch.pow(torch.norm(s, p=2), 2)+1e-8)
    mix_noise = mix - mix_target
    return 20*torch.log10(torch.norm(s_target, p=2)/(torch.norm(e_noise, p=2)+1e-8)) - 20*torch.log10(torch.norm(mix_target, p=2)/(torch.norm(mix_noise, p=2)+1e-8))


def permute_SI_SNR(_s_lists, s_lists, mix):
    '''
        Calculate all possible SNRs according to
        the permutation combination and
        then find the maximum value.
        input:
               _s_lists: Generated audio list
               s_lists: Ground truth audio list
        output:
               max of SI-SNR
    '''
    length = len(_s_lists)
    results = []
    per = []
    for p in permutations(range(length)):
        s_list = [s_lists[n] for n in p]
        result = sum([SI_SNR(_s, s, mix, zero_mean=True) for _s, s in zip(_s_lists, s_list)])/length
        results.append(result)
        per.append(p)
    return max(results), per[results.index(max(results))]


def read_and_config_file(wave_list):
    clean,noise = [],[]
    # duration = []
    with open(wave_list) as fid:
        for line in fid:
            tmp = line.strip().split()
            clean.append(tmp[1])
            noise.append(tmp[0])
            # duration.append(tmp[2])
    return noise,clean


def get_all_names(train_path,test_path):
    train_noisy_names, train_clean_names = read_and_config_file(train_path)
    test_noisy_names,test_clean_names = read_and_config_file(test_path)
    return train_noisy_names, train_clean_names,test_noisy_names, test_clean_names






