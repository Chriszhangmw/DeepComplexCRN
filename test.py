
"""
for checking speech quality with some metrics.
1. PESQ
2. STOI
3. CSIG, CBAK, COVL
"""

from config import Config
import os
import librosa
import soundfile as sf
from predict import predict_one_wav
from tqdm import  tqdm
import sys





def test(path,num):
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
    with open(path) as fid:
        for line in tqdm(fid):
            tmp = line.strip().split()
            clean_path = tmp[1]
            clean_path = str(clean_path.split('/')[-1])
            assert clean_path.endswith(".wav")
            noice_path = tmp[0]
            outputs = predict_one_wav(noice_path)
            new_path = os.path.join(generate_path,clean_path)
            sf.write(new_path, outputs, 16000)

if __name__ == "__main__":
    # step1 generate the test data
    config = Config()
    test_0_list = config.test_0_list
    test_5_list = config.test_5_list
    test_10_list = config.test_10_list
    test_15_list = config.test_15_list
    test_20_list = config.test_20_list
    test(test_0_list,0)
    test(test_5_list, 5)
    test(test_10_list, 10)
    test(test_15_list, 15)
    test(test_20_list, 20)























