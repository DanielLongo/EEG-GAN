# %load_ext autoreload
# %autoreload 2
import os
# import joblib
# import braindecode
import sys

## sys.path.append("/home/hartmank/git/GAN_clean")
sys.path.append("../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../data_loaders/")
sys.path.append("../../../../../GANs/")
# from braindecode.datautil.iterators import get_balanced_batches
from eeggan.examples.conv_lin.c_model import Generator, Discriminator
from eeggan.util import weight_filler
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random

# Choose one or multiple

#from load_EEGs import EEGDataset
# from load_EEGs_improved import EEGDataset
# from load_eegs_one_c import EEGDataset
from load_eegs_one_c_improved import EEGDataset
from forward_model_dataloader_one_c import ForwardModelDataset
from utils import save_EEG

plt.switch_backend('agg')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

n_critic = 5
n_batch = 64
input_length = 768
jobid = 0
suffix = "testing"
n_z = 200
lr = 0.001
# lr = .003
n_blocks = 6
rampup = 2000.
block_epochs = [2000] + [4000] * 5
# block_epochs = [200] + [400] * 5

subj_ind = int(os.getenv('SLURM_ARRAY_TASK_ID', '0'))
task_ind = 0  # subj_ind

np.random.seed(task_ind)
torch.manual_seed(task_ind)
torch.cuda.manual_seed_all(task_ind)
random.seed(task_ind)
rng = np.random.RandomState(task_ind)
#csv_file = "/mnt/data1/eegdbs/all_reports_impress_blanked-2019-02-23.csv"
# csv_file = 

csv_file = "/mnt/data1/eegdbs/all_reports_impress_blanked-2019-03-01.csv"
# real_eegs = EEGDataset("/mnt/data1/eegdbs/SEC-0.1/stanford/", num_examples=64*3, num_channels=44,
                      # length=input_length, csv_file=csv_file)
# print("SKJFSKFJSD", len(real_eegs))
real_eegs = ForwardModelDataset(num_examples=64*8, batch_size=64, length=input_length)
# print("data loaded")
generator = Generator(1, n_z)
discriminator = Discriminator(1)

generator.train_init(alpha=lr, betas=(0., 0.99))
discriminator.train_init(alpha=lr, betas=(0., 0.99), eps_center=0.001,
                         one_sided_penalty=True, distance_weighting=True)
generator = generator.apply(weight_filler)
discriminator = discriminator.apply(weight_filler)

i_block_tmp = 0
i_epoch_tmp = 0
generator.model.cur_block = i_block_tmp
discriminator.model.cur_block = n_blocks - 1 - i_block_tmp
fade_alpha = 1.
generator.model.alpha = fade_alpha
discriminator.model.alpha = fade_alpha

generator = generator.cuda()
discriminator = discriminator.cuda()
generator.train()
discriminator.train()

losses_d = []
losses_g = []
i_epoch = 0
z_vars_im = rng.normal(0, 1, size=(1000, n_z)).astype(np.float32)


def normalize(batch):
    batch = batch - batch.mean()
    batch = batch / batch.std()
    batch = batch / np.abs(batch).max()
    return batch

def generate_y(batch_size, n_variations=8):
    vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9][:n_variations]
    onehot = torch.zeros(n_variations, n_variations).scatter_(1, torch.LongTensor(vals).view(n_variations,1), 1).view(n_variations, n_variations, 1, 1).squeeze()
    assert(batch_size % n_variations == 0)
    fill = torch.LongTensor(vals * int((batch_size/n_variations)))
    fill = onehot[fill]
    fill = fill[torch.randperm(fill.shape[0])]
    return fill


def main():
    for i_block in range(i_block_tmp, n_blocks):
        c = 0

        # train_tmp = discriminator.model.downsample_to_block(Variable(torch.from_numpy(train).cuda(), volatile=True),
        # discriminator.model.cur_block).data.cpu()

        for i_epoch in range(i_epoch_tmp, block_epochs[i_block]):
            real_eegs.shuffle()
            # for i, eegs in range(real_eegs):
            for i in range(len(real_eegs)):
                y = generate_y(64).cuda()
                # eegs = real_eegs.getEEGs(i)
                eegs = real_eegs[i]
                if eegs.shape[0] != n_batch:
                    continue
                eegs = normalize(eegs)
                # i_epoch_tmp = 0
                #
                # if fade_alpha < 1:
                #     fade_alpha += 1. / rampup
                #     generator.model.alpha = fade_alpha
                #     discriminator.model.alpha = fade_alpha
                #
                # batches = get_balanced_batches(train.shape[0], rng, True, batch_size=n_batch)
                # iters = int(len(batches) / n_critic)
                #
                eegs = discriminator.model.downsample_to_block(
                    Variable(eegs[:, :, :, None].view(64, 1, input_length, 1), requires_grad=False),
                    discriminator.model.cur_block).cuda()
                # print("downsized", eegs.shape)
                for i_critic in range(n_critic):
                    # train_batches = train_tmp[batches[it * n_critic + i_critic]]
                    batch_real = Variable(eegs, requires_grad=True).cuda()  # .view(n_batch, 1, input_length, 1)
                    # print("input batch", batch_real.shape)

                    # z_vars = rng.normal(0, 1, size=(len(batches[it * n_critic + i_critic]), n_z)).astype(np.float32)
                    z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
                    # print("z_vars", z_vars.shape)
                    # z_vars = Variable(torch.from_numpy(z_vars), volatile=True).cuda()
                    z_vars = Variable(torch.from_numpy(z_vars), requires_grad=False).cuda()
                    # print("z_vars", z_vars.size())
                    batch_fake = Variable(generator(z_vars, y).data, requires_grad=True).cuda()
                    loss_d = discriminator.train_batch(batch_real, batch_fake, y=y)
                    # print("loss d", loss_d)
                    assert (np.all(np.isfinite(loss_d)))
                z_vars = rng.normal(0, 1, size=(n_batch, n_z)).astype(np.float32)
                z_vars = Variable(torch.from_numpy(z_vars), requires_grad=True).cuda()
                loss_g = generator.train_batch(z_vars, discriminator, y=y)

            losses_d.append(loss_d)
            losses_g.append(loss_g)

            if i_epoch % 10 == 0:
                generator.eval()
                discriminator.eval()
                print("batch_fake", batch_fake.cpu().detach().numpy().shape)
                np.save("./saved_runs/" + suffix + str(i_epoch), batch_fake.cpu().detach().view(n_batch, -1, 1).numpy()[0])

                print('Epoch: %d   Loss_F: %.3f   Loss_R: %.3f   Penalty: %.4f   Loss_G: %.3f' % (
                    i_epoch, loss_d[0], loss_d[1], loss_d[2], loss_g))

                # From forked implementation to save generations

                # joblib.dump((i_epoch, losses_d, losses_g), os.path.join(modelpath, modelname % jobid + '_.data'),
                #             compress=True)
                # joblib.dump((i_epoch, losses_d, losses_g),
                #             os.path.join(modelpath, modelname % jobid + '_%d.data' % i_epoch), compress=True)
                # # joblib.dump((n_epochs,n_z,n_critic,batch_size,lr),os.path.join(modelpath,modelname%jobid+'_%d.params'%i_epoch),compress=True)
                #
                # freqs_tmp = np.fft.rfftfreq(train_tmp.numpy().shape[2],
                #                             d=1 / (250. / np.power(2, n_blocks - 1 - i_block)))
                #
                # train_fft = np.fft.rfft(train_tmp.numpy(), axis=2)
                # train_amps = np.abs(train_fft).mean(axis=3).mean(axis=0).squeeze()
                #
                # z_vars = Variable(torch.from_numpy(z_vars_im), volatile=True).cuda()
                # batch_fake = generator(z_vars)
                # fake_fft = np.fft.rfft(batch_fake.data.cpu().numpy(), axis=2)
                # fake_amps = np.abs(fake_fft).mean(axis=3).mean(axis=0).squeeze()
                #
                # plt.figure()
                # plt.plot(freqs_tmp, np.log(fake_amps), label='Fake')
                # plt.plot(freqs_tmp, np.log(train_amps), label='Real')
                # plt.title('Frequency Spektrum')
                # plt.xlabel('Hz')
                # plt.legend()
                # plt.savefig(os.path.join(modelpath, modelname % jobid + '_fft_%d_%d.png' % (i_block, i_epoch)))
                # plt.close()
                #
                # batch_fake = batch_fake.data.cpu().numpy()
                # plt.figure(figsize=(10, 10))
                # for i in range(10):
                #     plt.subplot(10, 1, i + 1)
                #     plt.plot(batch_fake[i].squeeze())
                #     plt.xticks((), ())
                #     plt.yticks((), ())
                # plt.subplots_adjust(hspace=0)
                # plt.savefig(os.path.join(modelpath, modelname % jobid + '_fakes_%d_%d.png' % (i_block, i_epoch)))
                # plt.close()
                #
                # discriminator.save_model(os.path.join(modelpath, modelname % jobid + '.disc'))
                # generator.save_model(os.path.join(modelpath, modelname % jobid + '.gen'))
                #
                # plt.figure(figsize=(10, 15))
                # plt.subplot(3, 2, 1)
                # plt.plot(np.asarray(losses_d)[:, 0], label='Loss Real')
                # plt.plot(np.asarray(losses_d)[:, 1], label='Loss Fake')
                # plt.title('Losses Discriminator')
                # plt.legend()
                # plt.subplot(3, 2, 2)
                # plt.plot(np.asarray(losses_d)[:, 0] + np.asarray(losses_d)[:, 1] + np.asarray(losses_d)[:, 2],
                #          label='Loss')
                # plt.title('Loss Discriminator')
                # plt.legend()
                # plt.subplot(3, 2, 3)
                # plt.plot(np.asarray(losses_d)[:, 2], label='Penalty Loss')
                # plt.title('Penalty')
                # plt.legend()
                # plt.subplot(3, 2, 4)
                # plt.plot(-np.asarray(losses_d)[:, 0] - np.asarray(losses_d)[:, 1], label='Wasserstein Distance')
                # plt.title('Wasserstein Distance')
                # plt.legend()
                # plt.subplot(3, 2, 5)
                # plt.plot(np.asarray(losses_g), label='Loss Generator')
                # plt.title('Loss Generator')
                # plt.legend()
                # plt.tight_layout()
                # plt.savefig(os.path.join(modelpath, modelname % jobid + '_losses.png'))
                # plt.close()

                generator.train()
                discriminator.train()

        fade_alpha = 0.
        generator.model.cur_block += 1
        discriminator.model.cur_block -= 1

    torch.save(discriminator.state_dict(), "discriminator-" + suffix + ".pt")
    torch.save(generator.state_dict(), "generator-" + suffix + ".pt")


if __name__ == "__main__":
    main()
    print("finished " + suffix)
