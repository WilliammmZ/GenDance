import argparse
import yaml
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from moviepy import *
from moviepy.editor import *

from utils import util
import time
from model.gst_model import GST_Model
from model.init_pose_vae import PoseVae

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='/workspace/Demo_Dance/config/test.yaml', 
        help='Path to option YAML file.'
        )
    parser.add_argument('--mic_path', type=str, default='/workspace/Demo_Dance/data/test_music/0001.wav',
                                 help='The path of the music file (e.g. xxx.wav)')
    parser.add_argument('--mic_feat_path', type=str, default='/workspace/Demo_Dance/data/test_music_feat/0001.npy',
                                 help='The path of the music feature (process by get_music_feature.py)')
    parser.add_argument('--save_name', type=str, default='generate_dance_01.mp4',
                                 help='The name of the output name')
    parser.add_argument('--reference_path', type=str, default='/workspace/Demo_Dance/data/AIST_small/krump_dance/nor_k_feat/gKR_sFM_cAll_d28_mKR0_ch01.npy',
                                 help='The dance used for profuce style embedding')
    parser.add_argument('--ckpt_file_path', type=str, default='/workspace/Demo_Dance/log/gst_aist_compare_v051.ckpt',
                                 help='The path of model weight')
    parser.add_argument('--kps_scaler_path', type=str, default='/workspace/Demo_Dance/data/AIST_small/kps_scaler.pth',
                                 help='The normalize scaler for dance kps')
    parser.add_argument('--enable_coach', type=bool, default=True,
                                 help='Whether show the coach dance video in the final video')    
    parser.add_argument('--offset', type=int, default=300,
                                 help='the offset of music frame')  
    args = parser.parse_args()

    with open(args.opt_path, 'r') as fp:
        opt = yaml.load(fp, Loader=yaml.FullLoader)
    return opt, args




def label2img(label):
    img = np.ones((label.shape[0],label.shape[1],3), dtype=np.uint8) * (0,0,0)  
    label[np.where(label!=0)] = 1
    label[np.where(label!=1)] = 0
    mask = np.where(label==1)
    for x,y in zip(mask[0], mask[1]):    
        img[x,y,0]  = 241
        img[x,y,1]  = 196
        img[x,y,2]  = 15
    return np.array(img).astype(np.uint8)

def make_video(img_list, video_name, w, h):
    video=cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'MJPG'), 30, (w,h))
    for img in tqdm(img_list):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(w,h))
        video.write(img)
    video.release()


def gen_dance():
    opt, args = receive_arg()
    mic_size           = opt['mic_size']
    kps_size           = opt['kps_size']
    time_step          = opt['gstc']['time_step']
    seed               = opt['random_seed']
    
    # for kps_normalize
    kps_scaler = torch.load(args.kps_scaler_path)['kps_scaler']
    ckpt_file_path = args.ckpt_file_path
    music_path = args.mic_path
    music_feature_path = args.mic_feat_path
    reference_path = args.reference_path
    save_name = args.save_name
    enable_coach = args.enable_coach
    offset = args.offset
    
    # fix random seed
    # util.set_random_seed(seed)
    torch.backends.cudnn.benchmark = True  # speed up
    
    # load model
    param = torch.load(ckpt_file_path)
    new_param = {}
    for k,v in param.items():
        name = k[7:]
        new_param[name] = v
    
    dance_model = GST_Model(
        mic_size        =  mic_size,
        kps_size        =  kps_size,
        d_mic_embed     = opt['gstc']['d_mic_embed'],
        d_model         = opt['gstc']['d_model'],
        d_inner         = opt['gstc']['d_inner'],
        n_head          = opt['gstc']['n_head'],
        d_k             = opt['gstc']['d_k'],
        d_v             = opt['gstc']['d_v'],
        n_layers        = opt['gstc']['n_layers'],
        dropout         = opt['gstc']['dropout'],
        max_len         = opt['gstc']['max_len'],
        lstm_hidden_dim = opt['gstc']['lstm_hidden_dim'],
        lstm_n_layer    = opt['gstc']['lstm_n_layer'],
        n_token         = opt['gstc']['n_token'],
        s_n_head        = opt['gstc']['s_n_head'],
        shift           = opt['gstc']['shift']   
    )
    
    
    dance_model.load_state_dict(new_param)
    if torch.cuda.is_available():
        dance_model.cuda()
        
    # user also can use init pose vae to produce the init pose
    # init_gen = PoseVae(kps_size, 16)
    # init_gen.load_state_dict(torch.load('/workspace/Demo_Dance/log/init_pose.ckpt'))
    
    # load music feat
    mic_feat_seq = np.load(music_feature_path).astype(np.float32)
    test_num_seg = mic_feat_seq.shape[0] // time_step
    one_test = torch.FloatTensor(mic_feat_seq)
    pre_dance_list = []
    
    # load reference kps feat
    reference_kps_seq = np.load(reference_path).astype(np.float32)
    len_ref = reference_kps_seq.shape[0]
    reference_kps_seq = reference_kps_seq.reshape(-1, kps_size)
    mic_pos = torch.LongTensor([[i+1 for i in range(time_step)]for _ in range(1)]) # for pos  embedding
    if torch.cuda.is_available():
        one_test = one_test.cuda()
        reference_kps_seq = torch.from_numpy(reference_kps_seq).cuda()
        mic_pos = mic_pos.cuda()
    
    test_num_seg = 4
    all_reference = []
    print('Generating dance frame by our model ........')
    for idx in tqdm(range(test_num_seg)):
        start = idx * time_step
        mic_seg = one_test[start+offset:start+offset+time_step,:] 
        mic_seg = mic_seg.view(1, time_step, mic_size) #1 x len x mic_size
        
        ref_start = int(torch.randint(0, len_ref-time_step, (1,)))
        kps_seg = reference_kps_seq[ref_start:ref_start+time_step].view(1, time_step, kps_size)
            
        if idx == 0:    
            # initpose = np.zeros((kps_size,))
            # initpose = initpose.reshape(1,-1)
            # z = torch.FloatTensor(initpose)
            # if torch.cuda.is_available():
            #     z = z.cuda()
            # init_kps = dance_model.decode(z)
            
            # get init kps            
            select_idx = int(torch.randint(0, len_ref, (1,)))
            init_kps = reference_kps_seq[select_idx]
            init_kps = init_kps.view(1, kps_size)
                    
        r_kps_seq =kps_seg
        r_kps_seq = r_kps_seq.view(1, time_step, kps_size) 
       
                   
        enc_outputs, *_ = dance_model.music_encoder(mic_seg, mic_pos, mask=None)
        style_embedding, _ = dance_model.g_style(r_kps_seq)
        style_embedding = style_embedding.repeat(1, time_step, 1) 
        all_hidden = enc_outputs + style_embedding
        all_reference.extend(util.to_numpy(r_kps_seq))
            
        if idx == 0:
            vec_h, vec_c = dance_model.all_decoder.init_state(1)
            
        preds = []
        dec_input = init_kps
        for i in range(time_step):               
            dec_output, vec_h, vec_c = dance_model.all_decoder(dec_input, vec_h, vec_c)              
            dec_output = torch.cat([dec_output, all_hidden[:, i]], 1)
            dec_output = dance_model.linear(dec_output)
            preds.append(dec_output)
            dec_input = dec_output # dec_output
            
        outputs = [z.unsqueeze(1) for z in preds]
        pre_dance = torch.cat(outputs, dim=1)
        init_kps = pre_dance[:,-1,:]
            
        pre_dance_list.extend(util.to_numpy(pre_dance.view(time_step, kps_size)))
       

    pre_dance_list = np.array(pre_dance_list).reshape(-1, kps_size)
    pre_reference_kps = np.array(all_reference).reshape(-1, kps_size)
    save_pre_dance = util.inverse_normalize(pre_dance_list, kps_scaler, kps_size)
    save_ref_dance = util.inverse_normalize(pre_reference_kps, kps_scaler, kps_size)
    
    print('Dance frames generated complete ~ Now make video with the keypoints info ~')
    print('Because we convert the keypoints to 1080x720 (enable coach:1080x1440) image frames to make video, it will take a little long time ~')
    all_frames = save_pre_dance.shape[0]
    all_time = all_frames // 30
    all_frames = all_time * 30
    img_list = []
    
    if enable_coach:
        for i in tqdm(range(all_frames)): # cut the slience
            img = label2img(util.make_heatmap_34(save_pre_dance[i]))
            img_coach = label2img(util.make_heatmap_34(save_ref_dance[i]))
            
            img_coach = img_coach[260:1100,:]
            img_coach = img_coach[:,320:1600]
            
            img = img[260:1100,:] #crop black
            img = img[:,320:1600]
            
            text1 = 'coach_dance'
            text2 = 'generated_dance'
            
            cv2.putText(img_coach, text1, (40,80),cv2.FONT_HERSHEY_COMPLEX, 2.0, (255,255,255),2)
            cv2.putText(img, text2, (40,80),cv2.FONT_HERSHEY_COMPLEX, 2.0, (255,255,255),2)
        
            img = cv2.resize(img,(1080,720))
            img_coach = cv2.resize(img_coach,(1080,720))
            img = np.concatenate((img_coach, img), 0)
            img_list.append(img)
        make_video(img_list, 'tmp.avi', 1080, 1440)
    else:
        for i in tqdm(range(all_frames)):
            img = label2img(util.make_heatmap_34(save_pre_dance[i]))
           
            img = img[260:1100,:] #crop black
            img = img[:,320:1600]
            
            text2 = 'generated_dance'
            cv2.putText(img, text2, (40,80),cv2.FONT_HERSHEY_COMPLEX, 2.0, (255,255,255),2)
        
            img = cv2.resize(img,(1080,720))         
            img_list.append(img)
        make_video(img_list, 'tmp.avi', 1080, 720)
    
    video = VideoFileClip('tmp.avi')
    audio = AudioFileClip(music_path)
    audio_clip1 = audio.subclip(0+offset/30, all_time+offset/30) # jump the slience
    video = video.set_audio(audio_clip1)
    video.write_videofile(save_name, fps=video.fps, audio_codec='aac')# 输出


if __name__ == '__main__':
    gen_dance()
    