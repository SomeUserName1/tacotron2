#!/usr/bin/env python
# coding: utf-8

# ## Tacotron 2 inference code 
# Edit the variables **checkpoint_path** and **text** to match yours and run the entire code to generate plots of mel outputs, alignments and audio synthesis from the generated mel-spectrogram using Griffin-Lim.

# #### Import libraries and setup matplotlib
import matplotlib
import matplotlib.pylab as plt

import sys
import argparse
import os
sys.path.append('waveglow/')
import numpy as np
import torch
import librosa
from tika import parser

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim

from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                        interpolation='none')

def main(text): 
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.gate_threshold = 0.1
    hparams.max_decoder_steps = 5000


    # #### Load model from checkpoint
    checkpoint_path = "tacotron2_statedict.pt"
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()


    # #### Load WaveGlow for mel2audio synthesis and denoiser
    waveglow_path = 'waveglow_256channels.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()

    for m in waveglow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')
            
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)


    # #### Prepare text input
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()


    # #### Decode text input and plot results
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T))
   

    # #### Synthesize audio from spectrogram using WaveGlow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        
    # #### (Optional) Remove WaveGlow bias
    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    
    # save
    if (os.path.isfile("out.wav")):
        x, sr = librosa.load("out.wav")
        out = np.append(x, audio[0].data.cpu().numpy().astype(np.float32))
    else:
        out = audio[0].data.cpu().numpy().astype(np.float32)
<<<<<<< HEAD

=======
        librosa.output.write_wav('./first.wav', out, 22050)
    
>>>>>>> 64598f0aa9e9388219aed0920f716cb2eccef77b
    librosa.output.write_wav('./out.wav', out, 22050)



if __name__ == '__main__':
    if os.path.exists("out.wav"):
        os.remove("out.wav");
    a_parser = argparse.ArgumentParser()
    a_parser.add_argument("--file", type=str)

    args = a_parser.parse_args()

    if not os.path.exists(args.file) or not os.path.isfile(args.file):
        raise Exception("Data set path given does not exists")
    elif args.file.endswith(".pdf"):
        print("Not Working yet! Special characters are not parsed correclty and not evaluatable line by line")
        raw = parser.from_file(args.file)
        raw = str(raw)
        safe_text = raw.encode('utf-8', errors='ignore')
    elif args.file.endswith(".txt"):
        with open(args.file, "r") as readfile:
<<<<<<< HEAD
            safe_text = readfile.read()

    print('--- safe text ---') 
    print(safe_text)

    for line in safe_text:
        main(line)
=======
            line = readfile.readline()
            while line:
                print(line)
                main(line)
                line = readfile.readline()
>>>>>>> 64598f0aa9e9388219aed0920f716cb2eccef77b
