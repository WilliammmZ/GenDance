import os
import os.path as op
import torch
from torch.utils import data
import numpy as np
import random

# data_root = '/workspace/datasets/AIST/train_dance'
class AIST_Music(data.Dataset):
    def __init__(self, data_root, time_step, stride):
        mic_path = op.join(data_root, 'all_nor_m_feat')
        n_frame = 0
        self.music_type = []
        self.music_list = []
        self.stride = stride
        self.time_step = time_step
        
        for file_name in os.listdir(mic_path):
            m_type = file_name.split('.')[0][:3]
            if m_type not in self.music_type:
                self.music_type.append(m_type)

        for t in self.music_type:
            single_type_music = []
            for file_name in os.listdir(mic_path):
                m_type = file_name.split('.')[0][:3]
                if m_type == t:
                    mic_seq = np.load(op.join(mic_path, file_name)).astype(np.float32)
                    n_frame = n_frame + mic_seq.shape[0]
                    single_type_music.append(mic_seq)       
            self.music_list.append(single_type_music)
        self.dataset_size = n_frame // self.stride
        self.cls = len(self.music_type)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        select_laebl = int(torch.randint(0, self.cls, (1,)))
        aud_list = self.music_list[select_laebl]
        select_idx = int(torch.randint(0, len(aud_list), (1,)))
        aud_feat = aud_list[select_idx]

        start = int(torch.randint(0, (aud_feat.shape[0] - self.time_step)//self.stride + 1, (1,)))
        start = start * self.stride
        aud_seg = aud_feat[start : start+self.time_step, :]
        input_dict = {'mic':aud_seg, 'label':select_laebl}
        return input_dict
        
            
class AIST_Nvidia_De(data.Dataset):
    def __init__(self, data_root, dance_style, time_steps, kps_scaler, stride):
        self.kps_path = os.path.join(data_root, dance_style[0],'nor_k_feat')
        self.kps_files = sorted(os.listdir(self.kps_path))
        self.dance_count = len(self.kps_files)
        self.kps_seq = []
        n_frame = 0
        for i in range(self.dance_count):
            kps = np.load(os.path.join(self.kps_path, self.kps_files[i])).astype(np.float32)
            n_frame = n_frame + kps.shape[0]
            self.kps_seq.append(kps)

        self.time_steps = time_steps
        self.stride = stride
        self.kps_scaler = kps_scaler
        self.dataset_size = n_frame // self.stride
        print(self.dataset_size)


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        select_idx = int(torch.randint(0, self.dance_count, (1,)))
        kps = self.kps_seq[select_idx]
        data_frames = kps.shape[0]
        start = int(torch.randint(0, (data_frames - self.time_steps) // self.stride + 1, (1,)))

        start = start * self.stride
        ori_kps = kps[start:start + self.time_steps, :]

        xjit = np.random.uniform(low=-30, high=30)
        yjit = np.random.uniform(low=-10, high=10)
        kp_1 = ori_kps.copy()
        kp_1 = kp_1.reshape(-1,2)
        kp_1 = self.kps_scaler.inverse_transform(kp_1)
        kp_1[:,0] = kp_1[:,0] + xjit
        kp_1[:,1] = kp_1[:,1] + yjit
        kp_1 = self.kps_scaler.transform(kp_1)
        kp_1 = kp_1.reshape(self.time_steps, -1)

        xjit = np.random.uniform(low=-30, high=30)
        yjit = np.random.uniform(low=-10, high=10)
        kp_2 = ori_kps.copy()
        kp_2 = kp_2.reshape(-1,2)
        kp_2 = self.kps_scaler.inverse_transform(kp_2)
        kp_2[:,0] = kp_2[:,0] + xjit
        kp_2[:,1] = kp_2[:,1] + yjit
        kp_2 = self.kps_scaler.transform(kp_2)
        kp_2 = kp_1.reshape(self.time_steps, -1)

        input_dict = {'kp_1': kp_1, 'kp_2': kp_2}
        return input_dict
     
class AIST_Pose(data.Dataset):
    def __init__(self, data_root, dance_style, kps_size):
        self.data_root = data_root
        self.dance_style = dance_style
        self.kps_seq = []
       
        for d_s in self.dance_style:
            kps_dir =  op.join(self.data_root, d_s, 'nor_k_feat')
            for n in os.listdir(kps_dir):
                kps = np.load(op.join(kps_dir, n)).astype(np.float32)
                kps = np.array(kps).reshape(-1, kps_size)
                self.kps_seq.extend(kps)
        
        random.shuffle(self.kps_seq)
        # print('dataset size:', len(self.kps_seq)) #230558
        
    def __len__(self):
        return len(self.kps_seq)

    def __getitem__(self, index):
        kp = self.kps_seq[index]
        return kp
      
class AIST_Dance(data.Dataset):
    def __init__(self, data_root, dance_style, time_step, stride, kps_size, mic_size, compare = False):
       
        self.kps_seq_list = []
        self.mic_seq_list = []
        self.time_step = time_step
        self.stride = stride
        self.cls = len(dance_style)
        self.compare = compare
        self.kps_size = kps_size
        
        n_frame = 0
        for d_s in dance_style:
            d_path = op.join(data_root, d_s)
            kps_path = op.join(d_path, 'nor_k_feat')
            mic_path = op.join(data_root, 'all_nor_m_feat')
            kps_list = []
            mic_list = []
            for k_p in os.listdir(kps_path):
                mic_name = k_p.split('_')[4]
                mic_seq = np.load(op.join(mic_path, str(mic_name + '.npy'))).astype(np.float32)
                mic_seq = mic_seq.reshape(-1, mic_size)
                
                
                kps_seq = np.load(op.join(kps_path, k_p)).astype(np.float32)
                kps_seq = kps_seq.reshape(-1, kps_size)
                n_frame = n_frame + kps_seq.shape[0]
                
                kps_list.append(kps_seq)
                mic_list.append(mic_seq)
                
            self.kps_seq_list.append(kps_list)
            self.mic_seq_list.append(mic_list)
        if compare:
            self.data_size = n_frame // 2 // stride
        else:
            self.data_size = n_frame // stride
        print(self.data_size)
    
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index):
        if self.compare:
            # select style
            select_label_1 = int(torch.randint(0, self.cls, (1,)))
            kps_list_1 = self.kps_seq_list[select_label_1]
            mic_list_1 = self.mic_seq_list[select_label_1]

            select_label_2 = int(torch.randint(0, self.cls, (1,)))
            kps_list_2 = self.kps_seq_list[select_label_2]
            mic_list_2 = self.mic_seq_list[select_label_2]

            # select dance clip
            select_idx = int(torch.randint(0, len(kps_list_1), (1,)))
            kps_1 = kps_list_1[select_idx]
            mic_1 = mic_list_1[select_idx]
            select_idx = int(torch.randint(0, len(kps_list_2), (1,)))
            kps_2 = kps_list_2[select_idx]
            mic_2 = mic_list_2[select_idx]

            # select dance seq
            data_frames = kps_1.shape[0]
            start = int(torch.randint(0, (data_frames - self.time_step)//self.stride + 1, (1,)))
            start = start * self.stride
            kps_seg_1 = kps_1[start: start + self.time_step, :]
            mic_seg_1 = mic_1[start: start + self.time_step, :]

            data_frames = kps_2.shape[0]
            start = int(torch.randint(0, (data_frames - self.time_step)//self.stride + 1, (1,)))
            start = start * self.stride
            kps_seg_2 = kps_2[start: start + self.time_step, :]
            mic_seg_2 = mic_2[start: start + self.time_step, :]

            match = 0
            if select_label_1 == select_label_2:
                match = 1
            else:
                match = 0

            input_dict = {'mic_1': mic_seg_1, 'kps_1': kps_seg_1, 'cls_1': select_label_1,
                            'mic_2': mic_seg_2, 'kps_2': kps_seg_2, 'cls_2': select_label_2, 
                            'match':match}
        
        else:
            # select dance style
            select_label = int(torch.randint(0, self.cls, (1,)))
            kps_list = self.kps_seq_list[select_label]
            mic_list = self.mic_seq_list[select_label]
            
            # select dance clip
            select_idx = int(torch.randint(0, len(kps_list), (1,)))
            kps = kps_list[select_idx]
            mic = mic_list[select_idx]

            # select dance seq
            data_frames = kps.shape[0]
            start = int(torch.randint(0, (data_frames - self.time_step)//self.stride + 1, (1,)))
            start = start * self.stride
            kps_seg = kps[start: start + self.time_step, :]
            mic_seg = mic[start: start + self.time_step, :]

            input_dict = {'mic': mic_seg, 'kps': kps_seg, 'cls': select_label}

        return input_dict 
    
    def get_start(self):
        return np.array(self.kps_seq_list[0][0][0]).reshape(self.kps_size,)
    





            