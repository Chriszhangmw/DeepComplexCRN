import os
import librosa
import soundfile as sf
import time

import torch

from models.DCCRN import dccrn

import numpy as np

from config import Config
opt = Config




# when denoising, use cpu
def denoise(mode, speech_file, save_dir, pth=None):
    assert os.path.exists(speech_file), 'speech file does not exist!'

    assert speech_file.endswith('.wav'), 'non-supported speech format!'

    if not os.path.exists(save_dir):
        print('warning: save directory does not exist, it will be created automatically!')
        os.makedirs(save_dir)

    model = dccrn(mode)
    if pth is not None:
        model.load_state_dict(torch.load(pth), strict=True)

    noisy_wav, _ = librosa.load(speech_file, sr=16000)

    noisy_wav = torch.Tensor(noisy_wav).reshape(1, -1)

    torch.cuda.synchronize()
    start = time.time()

    _, denoised_wav = model(noisy_wav)

    torch.cuda.synchronize()
    end = time.time()

    print('process time {0}s on device {1}'.format(end - start, 'cpu'))

    speech_name = os.path.basename(speech_file)[:-4]

    noisy_path = os.path.join(save_dir, speech_name + '_' + 'noisy' + '.wav')
    denoised_path = os.path.join(save_dir, speech_name + '_' + 'denoised' + '.wav')

    noisy_wav = noisy_wav.data.numpy().flatten()
    denoised_wav = denoised_wav.data.numpy().flatten()

    sf.write(noisy_path, noisy_wav, 16000)
    sf.write(denoised_path, denoised_wav, 16000)

def process_predict(wav_path):
    def audioread(path):
        data, fs = sf.read(path)
        if len(data.shape) > 1:
            data = data[0]
        return data
    data = audioread(wav_path).astype(np.float32)
    inputs = np.reshape(data, [1, data.shape[0]])
    return inputs,data.shape[0]

def predict_one_wav(wav_path,use_cuda = True):
    args = Config()
    model = dccrn(args.mode)
    model.load_state_dict(torch.load(args.best_path), strict=True)
    inputs,nsamples = process_predict(wav_path)
    inputs = torch.from_numpy(inputs)
    _, denoised_wav = model(inputs)

    window = int(16000 * 4)  # 4s
    b, t = inputs.size()
    if t > int(1.5 * window):
        outputs = np.zeros(t)
        stride = int(window * 0.75)
        give_up_length = (window - stride) // 2
        current_idx = 0
        while current_idx + window < t:
            tmp_input = inputs[:, current_idx:current_idx + window]
            tmp_output = model(tmp_input, )[1][0].cpu().detach().numpy()
            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = tmp_output[:-give_up_length]

            else:
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = tmp_output[
                                                                                              give_up_length:-give_up_length]
            current_idx += stride
        if current_idx < t:
            tmp_input = inputs[:, current_idx:current_idx + window]
            tmp_output = model(tmp_input)[1][0].cpu().detach().numpy()
            length = tmp_output.shape[0]
            outputs[current_idx + give_up_length:current_idx + length] = tmp_output[give_up_length:]
    else:
        outputs = model(inputs)[1][0].cpu().detach().numpy()
    outputs = outputs[:nsamples]
    # sf.write('./BAC009S072.wav', outputs, 16000)
    return outputs



if __name__ == '__main__':
    wave_test = '/home/zmw/big_space/zhangmeiwei_space/asr_data/dsn/data/noise_dev/BAC009S0724W0132_-5.725_zOtr4awwLLo_5.725.wav'
    predict_one_wav(wave_test)






