import os
import numpy as np
import cv2
import math
import torch
import time
import random

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
        
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_timestr():
    """Return current time str."""
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def dict2str(input_dict, indent=0):
    """Dict to string for printing options."""
    msg = ''
    indent_msg = ' ' * indent
    for k, v in input_dict.items():
        if isinstance(v, dict):  # still a dict
            msg += indent_msg + k + ':[\n'
            msg += dict2str(v, indent+2)
            msg += indent_msg + '  ]\n'
        else:  # the last level
            msg += indent_msg + k + ': ' + str(v) + '\n'
    return msg


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def to_numpy(x):
    return x.cpu().data.numpy()

def inverse_normalize(kps, scaler, kps_size=42):
    for i, kp in enumerate(kps):
        kp = kp.reshape(-1, 2)
        kp = scaler.inverse_transform(kp)
        kps[i] = kp.reshape(-1, kps_size)
    return kps

def center_aist(kps):
    kps = kps.reshape(-1,2)
    max_x = np.max(kps[:, 0])
    min_x = np.min(kps[:, 0])
    warp_x = max_x - min_x
    warp_x = int(warp_x)
    max_y = np.max(kps[:, 1])
    min_y = np.min(kps[:, 1])
    warp_y = max_y - min_y
    warp_y = int(warp_y)
    # print(warp_x,warp_y)
    #center
    margin_max_x = 1920-max_x
    margin_min_x = min_x
    move_x = (margin_max_x - margin_min_x)/2
    kps[:,0] = kps[:, 0] + move_x
    
    margin_max_y = 1080-max_y
    margin_min_y = min_y
    move_y = (margin_max_y - margin_min_y)/2
    kps[:,1] = kps[:, 1] + move_y
    
    kps = kps.reshape(-1, 34)    
    return kps
    

def tensor2heatmap(kps, mode=0):
    _, kps_size = kps.shape[0], kps.shape[1]
    if kps_size == 34:
        kps = center_aist(kps)
        kps = scale_aist_toB(kps)
    heatmaps = np.zeros((kps.shape[0], 180, 320, 3))
    for ids, kp in enumerate(kps):
        if kps_size == 42:
            h_map = make_heatmap_42(kp)
        else:
            h_map = make_heatmap_34(kp) 
        h_map[np.where(h_map!=0)]=1
        h_map[np.where(h_map!=1)]=0
        mask = np.where(h_map==1)
        h_map = h_map.reshape(h_map.shape[0], h_map.shape[1], 1)
        h_map = np.concatenate((h_map, h_map, h_map), -1)
        for x,y in zip(mask[0], mask[1]):    
            h_map[x,y,0]  = 241
            h_map[x,y,1]  = 196
            h_map[x,y,2]  = 15
        # h_map = color_label(h_map)
        heatmaps[ids] = h_map
    if mode == 0:
        heatmaps = np.transpose(heatmaps, (0, 3, 1, 2))
    return heatmaps.astype(np.uint8)


def scale_aist_toB(kps):
# scale aist data to BDANCE size (1920,1080) -> (360, 180)
    tar_x = 320.0
    tar_y = 180.0
    scalex = tar_x / 1920.0
    scaley = tar_y / 1080.0
    #move (0,0) ->(-160,-90)
    x_kps = kps[:,::2]
    y_kps = kps[:,1::2]
    p_x_kps = np.where(x_kps == 0.0)
    p_y_kps = np.where(y_kps == 0.0)
    
    x_kps = x_kps - 1920 // 2
    y_kps = y_kps - 1080 // 2
    x_kps = x_kps * scalex
    y_kps = y_kps * scaley
    
    x_kps = x_kps + tar_x // 2
    y_kps = y_kps + tar_y // 2

    for x,y in zip(p_x_kps[0],p_x_kps[1]):
        x_kps[x,y] = 0.0
    for x,y in zip(p_y_kps[0],p_y_kps[1]):
        y_kps[x,y] = 0.0
        
    kps[:,::2] = x_kps
    kps[:,1::2] = y_kps
    return kps 


#plot the keypoints into heatmap
def make_heatmap_42(kps, h = 180, w = 320, line=2):
    connection_index=np.array([[15,0],[0,16],[0,1],[1,2],[2,3],[3,4],[4,20],[1,5],[5,6],[6,7],[7,19],[1,8],[8,9],[9,10],[10,11],[11,18],[8,12],[12,13],[13,14],[14,17]])
    kp = kps.reshape(-1,2)
    label = np.zeros((h,w), dtype=np.uint8)
    limb_type= 0
    for cord_index in connection_index:
        joint_coords = kp[cord_index, :2]
        if np.any(joint_coords==0.0):
            limb_type = limb_type + 1
            if not np.any(joint_coords[0,:]==0.0):
                polygon = cv2.ellipse2Poly(tuple(joint_coords[0,:].astype(int)), (int(16 / 2), line), 0, 0, 360, 1)
                cv2.fillConvexPoly(label, polygon, limb_type)  
            if not np.any(joint_coords[1,:]==0.0):
                polygon = cv2.ellipse2Poly(tuple(joint_coords[1,:].astype(int)), (int(16  / 2), line), 0, 0, 360, 1)
                cv2.fillConvexPoly(label, polygon, limb_type)    
            
        else:
            limb_type = limb_type + 1
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), line), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(label, polygon, limb_type)
    return label

#make heatmap for aist
def make_heatmap_34(kps, h = 1080, w = 1920, line=1):
    connection_index=np.array([[0,2],[0,1],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]])
    kp = kps.reshape(-1,2)
    label = np.zeros((h,w), dtype=np.uint8)
    limb_type= 0
    for cord_index in connection_index:
        joint_coords = kp[cord_index, :2]
        if np.any(joint_coords==0.0):
            limb_type = limb_type + 1
            if not np.any(joint_coords[0,:]==0.0):
                polygon = cv2.ellipse2Poly(tuple(joint_coords[0,:].astype(int)), (int(16 / 2), line), 0, 0, 360, 1)
                cv2.fillConvexPoly(label, polygon, limb_type)  
            if not np.any(joint_coords[1,:]==0.0):
                polygon = cv2.ellipse2Poly(tuple(joint_coords[1,:].astype(int)), (int(16  / 2), line), 0, 0, 360, 1)
                cv2.fillConvexPoly(label, polygon, limb_type)    
            
        else:
            limb_type = limb_type + 1
            coords_center = tuple(np.round(np.mean(joint_coords, 0)).astype(int))
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))
            polygon = cv2.ellipse2Poly(coords_center, (int(limb_length / 2), line), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(label, polygon, limb_type)
    return label