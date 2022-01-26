import os
from utils.tools2 import  compare
import numpy as np
from config import Config
def cal_metrix(path,num):
    if num == 0:
        generate_path = "/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/cl_generate_test/noise_0"
    elif num == 5:
        generate_path = "/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/cl_generate_test/noise_5"
    elif num == 10:
        generate_path = "/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/cl_generate_test/noise_10"
    elif num == 15:
        generate_path = "/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/cl_generate_test/noise_15"
    elif num == 20:
        generate_path = "/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/cl_generate_test/noise_20"
    else:
        print("error")
        return
    target_wav = []
    estimated_wav = []
    with open(path) as fid:
        for line in fid:
            tmp = line.strip().split()
            clean_path = tmp[1]
            clean_wav_name = str(clean_path.split('/')[-1])
            assert clean_wav_name.endswith(".wav")
            new_path = os.path.join(generate_path,clean_wav_name)
            target_wav.append(clean_path)
            estimated_wav.append(new_path)
    assert len(target_wav) == len(estimated_wav)
    CSIG, CBAK, CVOL, PESQ, SSNR = 0.0,0.0,0.0,0.0,0.0
    for es,clean in zip(estimated_wav, target_wav):
        res = compare(clean, es)
        pm = np.array([x[1:] for x in res])
        pm = np.mean(pm, axis=0)
        csig,cbak,covl,pesq,ssnr = pm
        if pesq == 0:
            continue
        if np.isnan(csig):
            continue
        if np.isnan(cbak):
            continue
        if np.isnan(covl):
            continue
        if np.isnan(pesq):
            continue
        if np.isnan(ssnr):
            continue
        CSIG += csig
        CBAK += cbak
        CVOL += covl
        PESQ += pesq
        SSNR += ssnr
    CSIG = CSIG/len(estimated_wav)
    CBAK = CBAK / len(estimated_wav)
    CVOL = CVOL / len(estimated_wav)
    PESQ = PESQ / len(estimated_wav)
    SSNR = SSNR / len(estimated_wav)
    print('PESQ1 {:.4} |  CSIG {:.4} | CBAK {:.4} | CVOL {:.4} | SSNR {:.4}'
          .format(PESQ, CSIG, CBAK, CVOL, SSNR))
def normalized_test_object(wave_inputs):
    dim = 4 * 16000
    if len(wave_inputs) > dim:
        # max_audio_start = len(wave_inputs) - dim
        # audio_start = np.random.randint(0, max_audio_start)
        data = wave_inputs[0:dim]
    else:
        data = np.pad(wave_inputs, (0, dim - len(wave_inputs)), "constant")
    return data





if __name__ == "__main__":
    # step1 generate the test data
    config = Config()


    # step 2, calculate the metrix
    test_0_list = config.test_0_list
    test_5_list = config.test_5_list
    test_10_list = config.test_10_list
    test_15_list = config.test_15_list
    test_20_list = config.test_20_list
    # test22(test_0_list,0)
    cal_metrix(test_0_list,0)
    cal_metrix(test_5_list, 5)
    cal_metrix(test_10_list, 10)
    cal_metrix(test_15_list, 15)
    cal_metrix(test_20_list, 20)





















