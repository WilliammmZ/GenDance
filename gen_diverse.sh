python /workspace/Demo_Dance/gen_multi_dance.py \
                        --mic_path           '/workspace/Demo_Dance/data/test_music/0015.wav' \
                        --mic_feat_path      '/workspace/Demo_Dance/data/test_music_feat/0015.npy' \
                        --save_name          'g15-ho01-900-01.mp4' \
                        --reference_path     '/workspace/Demo_Dance/data/AIST_small/house_dance/nor_k_feat/gHO_sFM_cAll_d19_mHO0_ch01.npy' \
                        --ckpt_file_path     '/workspace/Demo_Dance/log/gst_aist_compare_v051.ckpt' \
                        --kps_scaler_path    '/workspace/Demo_Dance/data/AIST_small/kps_scaler.pth' \
                        --offset             900

### flag:
# - mic_path: the path of the music file, end with .wav or .mp3
# - mic_feat_path: the path of the music feature of the input music
# - save_name: the name of outputs video
# - reference_path: the path of the reference dance clip (only need 150 frame, u can get the same style of dance following the reference)
# - ckpt_file_path: the model weight path
# - kps_scaler_path: the path of the normalization scaler for data transform or inverse transform
# - offset: select the start frame of the music (the script can only generate 20s video)