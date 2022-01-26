import os
import librosa
import soundfile as sf
import time
import sys
from  utils.tools import get_all_names
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchnet.meter import AverageValueMeter
import torch.nn.parallel.data_parallel as data_parallel
from models.DCCRN import dccrn
from models.loss import SISNRLoss

from dataloader.dataloader import WavDataset
from config import Config
opt = Config

def train(mode):
    model = dccrn(mode)
    model.to(opt.device)

    train_noisy_names, train_clean_names, test_noisy_names, test_clean_names = get_all_names(opt.tr_list, opt.dev_list)

    train_dataset = WavDataset(train_noisy_names, train_clean_names)
    test_dataset = WavDataset(test_noisy_names, test_clean_names)
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer,
                            milestones=[int(opt.max_epoch * 0.5),
                                        int(opt.max_epoch * 0.7),
                                        int(opt.max_epoch * 0.9)],
                            gamma=opt.lr_decay)
    criterion = SISNRLoss()

    loss_meter = AverageValueMeter()
    num_batch = len(train_dataloader)
    current_val = opt.min_sisnr
    for epoch in range(0, opt.max_epoch):
        model.train()
        loss_meter.reset()
        one_epoch_count = 0
        for i, (data, label) in enumerate(train_dataloader):
            one_epoch_count += 1
            data = data.to(opt.device)
            label = label.to(opt.device)
            # spec, wav = model(data)
            spec, wav = data_parallel(model, (data,),device_ids=opt.device_ids,output_device=opt.device_ids[1])
            wav = wav.to(opt.device)
            label = label.to(opt.device)
            optimizer.zero_grad()
            loss = criterion(wav, label)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if (i + 1) % opt.verbose_inter == 0:
                print('Epoch {:3d}/{:3d} | batches {:5d}/{:5d} |'
                      'SiSNR {:2.4f}'
                      .format(
                          epoch, opt.max_epoch, i + 1, num_batch,
                    -loss_meter.value()[0] / one_epoch_count
                    ))
        print('epoch', epoch + 1, 'SI-SNR', -loss_meter.value()[0] / one_epoch_count)
        val_sisnr = validation(model, test_dataloader,criterion,opt)
        scheduler.step()
        if val_sisnr > current_val:
            print('Rejected !!! model validation results is not good')
        else:
            print('save model at epoch {0} ...'.format(epoch + 1))
            save_path = os.path.join(opt.checkpoint_root,
                                     'DCCRN_{0}_{1}.pth'.format(mode, epoch + 1))
            torch.save(model.state_dict(), save_path)
            current_val = val_sisnr
        sys.stdout.flush()



def validation(model, test_dataloader,criterion,opt):
    num_sample = len(test_dataloader) * opt.batch_size
    model.eval()
    with torch.no_grad():
        for ind, (data, label) in enumerate(test_dataloader):
            data = data.to(opt.device)
            label = label.to(opt.device)
            spec, wav = data_parallel(model, (data,), device_ids=opt.device_ids, output_device=opt.device_ids[1])
            wav = wav.to(opt.device)
            label = label.to(opt.device)
            loss = criterion(wav, label)
    return float(loss / num_sample)


if __name__ == '__main__':
    train(opt.mode)

