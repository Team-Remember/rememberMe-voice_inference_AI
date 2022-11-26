import os

import numpy as np
import torch
import torch.nn as nn

import model.fastspeech.hparams as hp

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = hp.synth_visible_devices

import argparse
import re
from string import punctuation

from model.fastspeech.fastspeech2 import FastSpeech2

from model.fastspeech.text import text_to_sequence
import model.fastspeech.utils as utils

from g2pk import G2p
from jamo import h2j

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def kor_preprocess(text):
    text = text.rstrip(punctuation)

    g2p=G2p()
    phone = g2p(text)
    print('after g2p: ', phone)
    phone = h2j(phone)
    print('after h2j: ', phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ', phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    print('after re.sub: ', phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone, hp.text_cleaners))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long().to(device)


def get_FastSpeech2(num):
    checkpoint_path = 'model/fastspeech/ckpt/kss/checkpoint_950000.pth.tar'
    model = nn.DataParallel(FastSpeech2())
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])
    model.requires_grad = False
    model.eval()
    return model


def synthesize(model, vocoder, text, sentence, user_id, we_id, prefix=''):
    sentence = sentence[:10]  # long filename will results in OS Error

    mean_mel, std_mel = torch.tensor(np.load("model/fastspeech/preprocessed/kss/mel_stat.npy"), dtype=torch.float).to(
        device)
    mean_f0, std_f0 = torch.tensor(np.load("model/fastspeech/preprocessed/kss/f0_stat.npy"), dtype=torch.float).to(
        device)
    mean_energy, std_energy = torch.tensor(np.load("model/fastspeech/preprocessed/kss/energy_stat.npy"),
                                           dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)
    mean_f0, std_f0 = mean_f0.reshape(1, -1), std_f0.reshape(1, -1)
    mean_energy, std_energy = mean_energy.reshape(1, -1), std_energy.reshape(1, -1)

    src_len = torch.from_numpy(np.array([text.shape[1]])).to(device)

    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(text, src_len)

    mel_torch = mel.transpose(1, 2).detach()
    mel_postnet_torch = mel_postnet.transpose(1, 2).detach()
    f0_output = f0_output[0]
    energy_output = energy_output[0]

    mel_torch = utils.de_norm(mel_torch.transpose(1, 2), mean_mel, std_mel)
    mel_postnet_torch = utils.de_norm(mel_postnet_torch.transpose(1, 2), mean_mel, std_mel).transpose(1, 2)
    f0_output = utils.de_norm(f0_output, mean_f0, std_f0).squeeze().detach().cpu().numpy()
    energy_output = utils.de_norm(energy_output, mean_energy, std_energy).squeeze().detach().cpu().numpy()

    if not os.path.exists(hp.test_path):
        os.makedirs(hp.test_path)

    # Audio.tools.inv_mel_spec(mel_postnet_torch[0], os.path.join(hp.test_path, '{}_griffin_lim_{}.wav'.format(prefix, sentence)))

    if vocoder is not None:
        if hp.vocoder.lower() == "vocgan":
            utils.vocgan_infer(mel_postnet_torch, vocoder,
                               path=os.path.join(hp.test_path, '{}_{}.wav'.format(user_id, we_id)))

    # utils.plot_data([(mel_postnet_torch[0].detach().cpu().numpy(), f0_output, energy_output)], ['Synthesized Spectrogram'], filename=os.path.join(hp.test_path, '{}_{}.png'.format(prefix, sentence)))


# if __name__ == "__main__":
def synthesize_voice(text, user_id, we_id):
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, default=30000)
    args = parser.parse_args(args=[])

    model = get_FastSpeech2(args.step).to(device)
    if hp.vocoder == 'vocgan':
        vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
    else:
        vocoder = None

    # kss
    test_sentence = [text]

    g2p=G2p()

    mode = '3'
    print('you went for mode {}'.format(mode))
    sentence = test_sentence

    print('sentence that will be synthesized: ')
    print(sentence)
    for s in sentence:
        text = kor_preprocess(s)
        synthesize(model, vocoder, text, s, user_id, we_id, prefix='step_{}'.format(args.step))
