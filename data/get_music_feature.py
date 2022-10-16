# get test music feature

import librosa
import numpy as np
import os
import torch
import warnings
warnings.filterwarnings('ignore')
# music feature util func
def get_music_feat(p, sr, fl, hl, fps, scaler_path, save_path = None):
    # p: path of the music(xxx.wav)
    # sr: sample rate
    # fl: frame_length
    # hl: hop legth
    # scaler_path: normalize data
    # save_path
    
    print('music_path:', p)
   
    music, _  = librosa.load(p, sr)
    music = music / np.max(music)
    
    # ------get music feature fps30 ------ #
    chroma = librosa.feature.chroma_stft(music, sr, win_length = fl, hop_length = hl)
    chroma = np.array(chroma).T
    print('chroma_shape:', chroma.shape)

    mfcc = librosa.feature.mfcc(music, sr, hop_length = hl, n_fft = fl)
    mfcc = np.array(mfcc).T
    print('mfcc_shape:', mfcc.shape)

    oenv = librosa.onset.onset_strength(music, sr, hop_length = hl)
    tempo, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr)
                                 
    beat_times = librosa.times_like(oenv, sr=sr, hop_length=hl)                      
    oenv = oenv.reshape(-1,1)
    beat_frame = np.array(np.rint(beat_times[beats] * fps), dtype=np.int8)
    bt = np.zeros(oenv.shape)
    for bf in beat_frame:
        bt[bf,0] = 1.0
    print('mic_oenv_shape:', oenv.shape)
    print('beat_shape:', bt.shape)

    spec = librosa.feature.spectral_centroid(music, sr, hop_length = hl, n_fft = fl)
    spec = spec.reshape(-1,1)
    print('spec_shape:', spec.shape)

    rmse = librosa.feature.rms(music, hop_length = hl, frame_length = fl)
    rmse = rmse.reshape(-1,1) * 1000 + 1.0
    rmse = np.log10(rmse)
    print(np.min(rmse))
    print(np.max(rmse))
    print('rmse_shape:', rmse.shape)

    pitch, _, _ = librosa.pitch.pyin(music, fill_na=0.0, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=fl, hop_length=hl)
    pitch = np.array(pitch).reshape(-1,1)
    print('pitch(f0):', pitch.shape)

    music_feat = np.concatenate((chroma, mfcc, oenv, spec, rmse, pitch, bt),-1)
    print('music_feature.shape:', music_feat.shape)
    
    m_feat = music_feat.reshape(-1, 37)
    mic_scaler = torch.load(scaler_path)['music_scaler']
    need_nor_feat = m_feat[:, :-1]
    beat = m_feat[:, -1].reshape(-1, 1)
    nor_m_feat = mic_scaler.transform(need_nor_feat)
    one_test = np.concatenate((nor_m_feat, beat), axis=1).astype(np.float32)

    if save_path:
        np.save(save_path, one_test)

#================
# NOT CHANGE
#================
fps = 30
sr = 24000
fl = sr // fps # frame length
hl = sr // fps # hop length
scaler_path = '/workspace/Demo_Dance/data/music_scaler.pth'       # the file u store the scaler info


#===================
# USER NEED CONFIG
#===================
music_path = '/workspace/Demo_Dance/data/test_music/0015.wav'     # the music path you want to process
save_path = '/workspace/Demo_Dance/data/test_music_feat/0015.npy' # the file path for save music feat

get_music_feat(
                p            = music_path,
                sr           = sr,
                fl           = fl,
                hl           = hl,
                fps          = fps,
                scaler_path  = '/workspace/Demo_Dance/data/music_scaler.pth',
                save_path    = save_path
              )

