## Overview
This is the PyTorch implementation to reproduce the multi-style dance generation process with the AIST++ dataset.

## Requirements
- pytorch 1.10.1 (ours)
- python 3.7
- librosa 0.9.1
- moviepy 1.0.3
- ffmpeg 1.4

## Dataset

- AIST++ dataset
    
    Download & introduction websit: https://google.github.io/aistplusplus_dataset/download.html

> For produce style embeddings, we put a small part of the AIST++ dataset in source code, only include 3 styles of dance.


## Model weight

The model weight 'gst_aist_compare_v051.ckpt' from: https://drive.google.com/drive/folders/1EtcKb4O4v17ncbY_OYk1KT7xymlQyQOw?usp=sharing

Put the file ```'gst_aist_compare_v051.ckpt'``` into ```log/```


## How to generate dance (with the public AIST++ dataset)

### 1. Prepare the music feature

- We provide part of our test music data in 'data/test_music' and 'data/test_music_feat', so you can skip this stage.

- If you want to generate with other music, please employ the 'data/get_music_feature.py' to extract the music feature.

### 2. Prepare the reference dance  (coach dance)

- Our proposed model can generate dance follow the style of coach dance, so you need prepare a clip of dance to guide dance generation. We provide three styles of reference dance in ```
/workspace/Demo_Dance/data/AIST_small```, you can select the style you like to guide dance generation. 

### 3. Diversity dance generation

- You can apply the command ```bash gen_diverse.sh``` to generate diverse dances accroding to the input music.
> The command will make our model generates two different dances(20s) and display them in a video clip. You can also run this shell many times to get different dances driven by the same music. The dance style will be influenced by the reference dance you select.

- Note: You can also change the args in ```gen_diverse.sh``` to generate dances according to your config.
```
# file gen_diverse.sh
python /workspace/Demo_Dance/gen_multi_dance.py \
                        --mic_path           '/workspace/datasets/AIST/test_music/0003.wav' \
                        --mic_feat_path      '/workspace/datasets/AIST/test_music_feature/0003.npy' \
                        --save_name          'generate_dance-test03-ho-01-01.mp4' \
                        --reference_path     '/workspace/Demo_Dance/data/AIST_small/house_dance/nor_k_feat/gHO_sFM_cAll_d19_mHO0_ch01.npy' \
                        --ckpt_file_path     '/workspace/Demo_Dance/log/gst_aist_compare_v051.ckpt' \
                        --kps_scaler_path    '/workspace/Demo_Dance/data/AIST_small/kps_scaler.pth' \
                        --offset             600

```

### Flag:
- mic_path: the path of the music file, end with .wav or .mp3.
- mic_feat_path: the path of the music feature of the input music.
- save_name: the name of outputs video.
- reference_path: the path of the reference dance clip(only need 150 frame, u can get the same style of dance following the reference).
- ckpt_file_path: the model weight path.
- kps_scaler_path: the path of the normalization scaler for data transform or inverse transform.
- offset: select the start frame of the music(the script can only generate 20s video).

### Outputs display：

<video src="/workspace/Demo_Dance/case3.mp4" controls="controls" width="500" height="300"></video>



### 4. Check the coach dance

- We also provide a script to show the coach dance(reference dance) and the generated dance.
- Note: You can also change the args in ```gen_with_coach.sh``` to generate dances according to your config.
```
python /workspace/Demo_Dance/gen_dance_with_coach.py \
                        --mic_path           '/workspace/Demo_Dance/data/test_music/0015.wav' \
                        --mic_feat_path      '/workspace/Demo_Dance/data/test_music_feat/0015.npy' \
                        --save_name          'generate_dance_03.mp4' \
                        --reference_path     '/workspace/Demo_Dance/data/AIST_small/house_dance/nor_k_feat/gHO_sFM_cAll_d19_mHO0_ch01.npy' \
                        --ckpt_file_path     '/workspace/Demo_Dance/log/gst_aist_compare_v051.ckpt' \
                        --kps_scaler_path    '/workspace/Demo_Dance/data/AIST_small/kps_scaler.pth' \
                        --offset             300 \
                        --enable_coach       True

```

### Flag:
- mic_path: the path of the music file, end with .wav or .mp3.
- mic_feat_path: the path of the music feature of the input music.
- save_name: the name of outputs video.
- reference_path: the path of the reference dance clip(only need 150 frame, u can get the same style of dance following the reference).
- ckpt_file_path: the model weight path.
- kps_scaler_path: the path of the normalization scaler for data transform or inverse transform.
- offset: select the start frame of the music(the script can only generate 20s video).
- enable_coach: show coach dance(reference dance) with generated dance.

### Outputs display:

<video src="/workspace/Demo_Dance/case3.mp4" controls="controls" width="500" height="300"></video>

### 5. Cases

We also provide some cases generated by these shells, see file ```case1 (2,3,4,5,6).mp4```.







