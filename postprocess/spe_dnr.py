import torch
import numpy as np
from tifffile import tifffile
import torch.nn as nn
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import sys
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.dirname(CURRENT_DIR))
            

from utils.gpu import get_gpu_with_max_free_vram



'''
CenterlineNet
'''
class conv_block(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm3d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class conv_block_2D(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block_2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm2d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

    
class CenterlineNet_Discrimintor_2D_Radii_32(nn.Module):
    def __init__(self, NUM_ACTIONS=1024, n=10):
        super(CenterlineNet_Discrimintor_2D_Radii_32, self).__init__()
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.layer1 = conv_block_2D(n-1, 32, 3, stride=2, p_size=0)
        self.layer2 = conv_block_2D(32, 32, 3, stride=1, p_size=1)
        self.layer3 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block_2D(32, 32, 3, stride=1, p_size=0, dilation=4)
        
        self.discriminator = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),                     
        )        
        self.dis_out = nn.Conv2d(64, 2+1, kernel_size=1, stride=1, padding=0)
        
        self.tracker = nn.Sequential(
            conv_block_2D(32, 64, 3, stride=1, p_size=0),
            conv_block_2D(64, 64, 1, stride=1, p_size=0),
            nn.Conv2d(64, NUM_ACTIONS, kernel_size=1, stride=1, padding=0)         
        )                    


    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        # out = self.down_sampling(out)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        # print(out.shape)
        out_dis = self.discriminator(out)
        # print(out_dis.shape)
        out_dis = self.dis_out(out_dis)
        # print(out_dis.shape)
        out = self.tracker(out)       

        return out, out_dis
    
'''
utils
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

def saveswc(filepath, swc):
    if swc.shape[1] > 7:
        swc = swc[:, :7]

    with open(filepath, 'w') as f:
        for i in range(swc.shape[0]):
            print('%d %d %.3f %.3f %.3f %.3f %d' %
                  tuple(swc[i, :].tolist()), file=f)

class Node(object):
   def __init__(self, position, conf, radius, node_type=3):
       self.position = position
       self.conf = conf
       self.radius = radius
       self.nbr = []
       self.node_type = node_type

def get_undiscover(dist):
    for i in range(dist.shape[0]):
        if dist[i] == 100000:
            return i
    return -1

def compute_trees(n0):
    n0_size = len(n0)
    treecnt = 0
    q = [] # bfs queue
    n1 = []
    dist = np.ones([n0_size,1], dtype=np.int64)*100000
    nmap = np.ones([n0_size,1], dtype=np.int64)*-1 # index in output tree n1
    parent = np.ones([n0_size,1], dtype=np.int64)*-1 # parent index in current tree n0
    # print('Search for Soma')
    for i in range(n0_size):
        if n0[i].node_type==1:
            q.append(i)
            dist[i] = 0
            nmap[i] = -1
            parent[i] = -1
    # BFS
    while len(q)>0:
        curr = q.pop(0)    
        n = Node(n0[curr].position, n0[curr].conf, n0[curr].radius, treecnt+2)
        if parent[curr]>0:
            n.nbr.append(nmap[parent[curr]])
        n1.append(n)
        nmap[curr] = len(n1)
        for j in range(len(n0[curr].nbr)):
            adj = n0[curr].nbr[j]
            if dist[adj] == 100000:
                dist[adj] = dist[curr] + 1
                parent[adj] = curr
                q.append(adj)
                
    while ((get_undiscover(dist))>=0):
        treecnt += 1
        seed = get_undiscover(dist)
        dist[seed] = 0
        nmap[seed] = -1
        parent[seed] = -1
        q.append(seed)
        while len(q)>0:
            curr = q.pop(0)    
            n = Node(n0[curr].position, n0[curr].conf, n0[curr].radius, treecnt+2)
            if parent[curr]>0:
                n.nbr.append(nmap[parent[curr]])
            n1.append(n)
            nmap[curr] = len(n1)
            for j in range(len(n0[curr].nbr)):
                adj = n0[curr].nbr[j]
                if dist[adj] == 100000:
                    dist[adj] = dist[curr] + 1
                    parent[adj] = curr
                    q.append(adj)      
    return n1

def build_nodelist(tree):
    _data = np.zeros((1, 7))
    cnt_recnodes = 0
    for i in range(len(tree)):
        if len(tree[i].nbr)==0:
            cnt_recnodes += 1
            pid = -1
            new_node = np.asarray([cnt_recnodes, 
                        tree[i].node_type, 
                        tree[i].position[1], 
                        tree[i].position[0], 
                        tree[i].position[2], 
                        tree[i].radius, 
                        pid])
            _data = np.vstack((_data, new_node))
            
        else:
            for j in range(len(tree[i].nbr)):
                cnt_recnodes += 1
                pid = tree[i].nbr[j].squeeze()
                new_node = np.asarray([cnt_recnodes, 
                        tree[i].node_type, 
                        tree[i].position[1], 
                        tree[i].position[0], 
                        tree[i].position[2], 
                        tree[i].radius, 
                        pid])
                _data = np.vstack((_data, new_node))
    _data = _data[1:,:]
    return _data

def local_max(Im, wsize=3, thre=255*0.5):
    nZ, nY, nX = Im.shape
    suppress = np.zeros_like(Im)
    # thre =0.01*255
    potential_points = np.where(Im>thre)
    num_points = len(potential_points[0])
    coordinates = []
    for i in tqdm(range(num_points), disable=True):
        z = potential_points[0][i]
        y = potential_points[1][i]
        x = potential_points[2][i]
        if wsize == 3:
            if x < 1 or y < 1 or z < 1 or x > nX-2 or y > nY-2 or z > nZ-2:
                continue
            
            img_patch = Im[z-1:z+2,y-1:y+2,x-1:x+2]
            if img_patch.max() == img_patch[1,1,1]:
                suppress[z,y,x]=255
                coordinates.append([y,x,z,Im[z,y,x]])
        if wsize == 5:
            if x < 2 or y < 2 or z < 2 or x > nX-3 or y > nY-3 or z > nZ-3:
                continue
            
            img_patch = Im[z-2:z+3,y-2:y+3,x-2:x+3]
            if img_patch.max() == img_patch[2,2,2]:
                suppress[z,y,x]=255
                coordinates.append([y,x,z,Im[z,y,x]])
                
        if wsize == 7:
            if x < 3 or y < 3 or z < 3 or x > nX-4 or y > nY-4 or z > nZ-4:
                continue
            
            img_patch = Im[z-3:z+4,y-3:y+4,x-3:x+4]
            if img_patch.max() == img_patch[3,3,3]:
                suppress[z,y,x]=255
                coordinates.append([y,x,z,Im[z,y,x]])            
    return suppress, coordinates

def Spherical_Patches_Extraction(img2, position, n, sphere_core, node_step=1):

    x = position[0]
    y = position[1]
    z = position[2]
    radius = 0
    j=np.arange(radius+1,n*node_step+radius+1,node_step).reshape(-1,n)
    ray_x = x+(sphere_core[:,0].reshape(-1,1))*j
    ray_y = y+(sphere_core[:,1].reshape(-1,1))*j
    ray_z = z+(sphere_core[:,2].reshape(-1,1))*(j*0.35)
    
    
    Rray_x=np.rint(ray_x).astype(int)
    Rray_y=np.rint(ray_y).astype(int)
    Rray_z=np.rint(ray_z).astype(int)

    Spherical_patch_temp = img2[Rray_z,Rray_x,Rray_y]
    Spherical_patch = Spherical_patch_temp[:,1:n]
        
    return Spherical_patch

def savemarker(filepath,marker):    
    with open(filepath, 'w') as f:
        for i in range(marker.shape[0]):
            markerp=[marker[i,1],marker[i,0],marker[i,2],0,1,' ',' ']        
            print('%.3f, %.3f, %.3f, %d, %d, %s, %s'  %  (markerp[0], markerp[1], markerp[2], markerp[3],markerp[4], markerp[5], markerp[6]),file=f)
                     
def generate_sphere(Ma,Mp):
    #generate 3d sphere
    m1=np.arange(1,Ma+1,1).reshape(-1,Ma)
    m2=np.arange(1,Mp+1,1).reshape(-1,Mp)
    alpha=2*(np.pi)*m1/Ma
    phi=-(np.arccos(2*m2/(Mp+1)-1)-(np.pi))
    xm=(np.cos(alpha).reshape(Ma,1))*np.sin(phi)
    ym=(np.sin(alpha).reshape(Ma,1))*np.sin(phi)
    zm=np.cos(phi)
    zm=np.tile(zm,(Mp,1))
    sphere_core=np.concatenate([xm.reshape(-1,1), ym.reshape(-1,1), zm.reshape(-1,1)],axis=1) #y_axis=alpha[0:Ma],x_axis=phi[0:Mp]
    return sphere_core, alpha, phi

def soma_point(soma_img, distance_transform):

    maxr = np.max(distance_transform)
    print(maxr)
    position = np.argwhere(distance_transform == maxr)
    soma_point = np.argwhere(soma_img >= 200)
    temp = position[0] - soma_point
    tmp1 = np.linalg.norm(temp, axis=1)
    radius = np.max(tmp1)
    print(radius)
    return position, radius

def inSphere(x, y, z,ball_center_x, ball_center_y, ball_center_z, radius):

    dist = (x - ball_center_x) ** 2 + (y - ball_center_y) ** 2 + (z - ball_center_z) ** 2
    return dist < (radius ** 2)

def connected(soma_img, img_tif, singletree_swc, swc_path, distance_transform):

    swc_connected = np.copy(singletree_swc)
    # find soma center and soma radius
    ballcenter_candidate, radius = soma_point(soma_img, distance_transform)
    ball_center_x = ballcenter_candidate[0, 2]
    ball_center_y = ballcenter_candidate[0, 1]
    ball_center_z = ballcenter_candidate[0, 0]

    swc_connected[:, 0] = swc_connected[:, 0] + 1
    swc_connected[:, 6] = swc_connected[:, 6] + 1

    for i in range(len(singletree_swc)):
        if singletree_swc[i, 6] == -1:
            if i+1 in singletree_swc[:, 6]:
                if inSphere(singletree_swc[i, 2], singletree_swc[i, 3], singletree_swc[i, 4], ball_center_x, ball_center_y, ball_center_z, radius+30):
                    swc_connected[i, 6] = 1
                    swc_connected[i, 1] = 2
                else:
                    swc_connected[i, 6] = -1

    first_row = np.array([1, 2, ball_center_x, ball_center_y, ball_center_z, radius, -1])
    swc_connected = np.insert(swc_connected, 0, values=first_row, axis=0)

    a = []
    for j in range(len(swc_connected)):
        if swc_connected[j, 6] == 0:
            a.append(j)
    swc_connected = np.delete(swc_connected, a, axis=0)

    save_swc_path1 = swc_path + '_connected_soma.swc'
    saveswc(save_swc_path1, swc_connected)
    print('--------')

def inSphere_yzk(x, y, z, soma, radius):

    dist = ((x - soma.center[0])*soma.scale[0]) ** 2 + ((y - soma.center[1])*soma.scale[1]) ** 2 + ((z - soma.center[2])*soma.scale[2]) ** 2
    return dist < ((soma.center.radius + radius) ** 2)

def connected_yzk(somas, singletree_swc, swc_path):

    swc_connected = np.copy(singletree_swc)

    swc_connected[:, 0] = swc_connected[:, 0] + len(somas)
    swc_connected[:, 6] = swc_connected[:, 6] + len(somas)
    swc_connected[:, 5] = 0.1

    node_to_delete = []
    for i in range(len(singletree_swc)):
        if singletree_swc[i, 6] == -1:
            if singletree_swc[i, 0] in singletree_swc[:, 6]:
                f = True
                for j, soma in enumerate(somas):
                    if inSphere_yzk(singletree_swc[i, 2], singletree_swc[i, 3], singletree_swc[i, 4], soma, 10):
                        swc_connected[i, 6] = j+1
                        swc_connected[i, 1] = 2
                        f = False
                        break
                if f:
                    swc_connected[i, 6] = -1
            else:
                node_to_delete.append(i)
    swc_connected = np.delete(swc_connected, node_to_delete, axis=0)

    for i, soma in enumerate(somas):
        first_row = np.array([i+1, 1, soma.center[0], soma.center[1], soma.center[2], 0.1, -1])
        swc_connected = np.insert(swc_connected, i, values=first_row, axis=0)
    saveswc(swc_path, swc_connected)
    
'''
Tracker
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from tqdm import tqdm

class Tracker(object):
    def __init__(self, img2, soma_mask2, terminations, dis2centerline, Lambda, K, 
                 angle_T, max_iter, step_size, node_step, mask_size, psize,
                 Ma, Mp, n, Xb, Yb, Zb, 
                 model, sphere_core, sphere_core_label, device):
        
        self.img2 = img2
        self.soma_mask2 = soma_mask2
        self.indx_map = np.zeros_like(img2, dtype=np.int64) # index map to label the index of traced point
        self.terminations = terminations
        self.dis2centerline = dis2centerline
        self.angle_T = angle_T
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_step = node_step
        self.Ma = Ma
        self.Mp = Mp
        self.n = n
        self.psize = psize
        self.Xb = Xb
        self.Yb = Yb
        self.Zb = Zb
        self.device = device
        self.model = model
        self.sphere_core = sphere_core
        self.sphere_core_label = sphere_core_label 
        self.traced_seed = []     
        self.track = []   
        self.direction_track = []
        self.confidence_track = []
        self.boundary_location = []
        self.success_location = []
        self.terminated_location = []
        self.soma_location = []
        self.mask_size = mask_size
        self.pt_index = 0 # index of the points
        self.ndlist = [] # final node list
        self.rlist = []
        self.relu = torch.nn.ReLU(inplace=True)
        self.R = Lambda*Lambda
        self.K = K

    def _mask_point(self, nd, radii, index=0):
        n = np.rint(nd).astype(int)
        radii = 2*np.rint(radii).astype(int)
        Z, X, Y = np.meshgrid(
                    constrain_range(n[2] - radii, n[2] + radii + 1, self.psize, self.Xb - self.psize - 1),
                    constrain_range(n[0] - radii, n[0] + radii + 1, self.psize, self.Yb - self.psize - 1),
                    constrain_range(n[1] - radii, n[1] + radii + 1, self.psize, self.Zb - self.psize - 1), indexing='ij')
        if index == 0:
            self.indx_map[Z, X, Y] = self.pt_index
        else:
            self.indx_map[Z, X, Y] = index
            
    def trace_JointDecision(self):        
        lent = self.terminations.shape[0]
        
        for i in tqdm(range(lent), disable=True):
            position = self.terminations[i, 0:3]
            pix_id = self.indx_map[position[2], position[0], position[1]]
            if pix_id > 0:
                # print('Traced Seed')
                self.traced_seed.append(position)
                continue           

            if position[0] <= self.psize or position[1] <= self.psize or \
               position[2] <= self.psize or position[0] >= self.Xb-self.psize-1 or \
               position[1] >= self.Yb - self.psize - 1 or \
               position[2] >= self.Zb - self.psize - 1:
                continue
            
            # SPE for feature extraction
            Spherical_patch = Spherical_Patches_Extraction(self.img2, position,
                                                           self.n, self.sphere_core, 
                                                           self.node_step)
            SP = Spherical_patch.reshape([1, self.Ma, self.Mp, self.n-1]).transpose([0,3,1,2])
            SP = np.asarray(SP)
            pmax = SP.max()
            
            if pmax > 0:
                SP = SP/pmax
            
            data = torch.from_numpy(SP)
            inputs = data.type(torch.FloatTensor).to(self.device)
            with torch.no_grad():
                outputs1, stop_flag = self.model(inputs)
            
            outputs = outputs1
            if stop_flag.shape[1] > 2:
                radii = self.relu(stop_flag[:,-1,:,:]).cpu().detach().numpy().squeeze() + 1
                stop_flag = stop_flag[:,:2,:,:]
            else:
                radii = 1
            
            outputs = torch.nn.functional.softmax(outputs,1)
            stop_flag = torch.nn.functional.softmax(stop_flag,1)
            direction_vector = outputs.cpu().detach().numpy().reshape([self.K,1])
            stop_flag = stop_flag.cpu().detach().numpy().squeeze().argmax()
            if stop_flag == 1:
                # skip without adding it to ndlist
                self.terminated_location.append(position)
                # print('terminated by discriminater', 0)
                continue

            # determine two initial direction        
            max_id = np.argmax(direction_vector)
            direction1 = self.sphere_core_label[max_id, :]
            
            cos_angle = np.sum(direction1*self.sphere_core_label, axis=1)
            cos_angle[cos_angle>1] = 1
            cos_angle[cos_angle<-1] = -1
            angle = np.arccos(cos_angle).reshape([self.K,1])
            direction_vector[angle<=np.pi/2] = 0
            max_id = np.argmax(direction_vector)            
            direction2 = self.sphere_core_label[max_id, :]
            _confidence = direction_vector[max_id]
            self.confidence_track.append(_confidence)
            
            soma_reached = self.soma_mask2[position[2], position[0], position[1]]
            
            if soma_reached:
                # skip after adding it to ndlist
                self.soma_location.append(position)
                # print('Soma Reached', 0)
                nd = Node(position, _confidence, radii, 1)
                self.ndlist.append(nd)
                self.rlist.append(radii)
                self.pt_index += 1 # current value of pt_index = len(ndlist)
                self.track.append(position)
                continue
            
            nd = Node(position, _confidence, radii)
            self.ndlist.append(nd)
            self.rlist.append(radii)
            self.pt_index += 1 # current value of pt_index = len(ndlist)
            self.track.append(position)
            self.position_id = self.pt_index # used for masking position location after bidirectional tracking                       
            previous_nd_len = len(self.ndlist) # length of ndlist in previous iteration
            
            # trace towards direction1
            track_neg = False
            self._Track_Pos(position, direction1, track_neg, radii)
            
            # trace towards direction2
            position = self.terminations[i, 0:3]
            track_neg = True
            self._Track_Pos(position, direction2, track_neg, radii)
            
            # label the traced branches
            self._mask_point(position, radii, self.position_id)
            len_branch = len(self.ndlist) - previous_nd_len # length of new added branches
            for j in range(len_branch):
                position_m = self.ndlist[previous_nd_len + j].position
                self._mask_point(position_m, self.rlist[previous_nd_len + j], previous_nd_len+j+1) 
            
    def _Track_Pos(self, position, direction1, track_neg, radii):
        cc = 0 # steps counter
        correct_flag = 0
        
        while cc < self.max_iter:
            cc += 1
            if correct_flag == 0: # next_position is determined by seed points if correct_flag==1
                next_position = position + direction1 * np.max([radii, self.step_size])
            correct_flag = 0
            
            if next_position[0]<=self.psize or next_position[1]<=self.psize or \
               next_position[2]<=self.psize or next_position[0]>=self.Xb-self.psize-1 or \
               next_position[1]>=self.Yb-self.psize-1 or \
               next_position[2]>=self.Zb-self.psize-1:
                # print('reached boundary', next_position, cc)
                self.boundary_location.append(next_position)
                break
            
            position = next_position.copy()     
            position_1 = np.round(position).astype(int)
            pix_id = self.indx_map[position_1[2], position_1[0], position_1[1]]
            if pix_id > 0:
                # print('Meet Traced Region', cc)
                # build connection between the met points
                # radii = self.rlist[pix_id]
                # nd = Node(position, direction1, radii)
                # self.ndlist.append(nd)
                # self.rlist.append(radii)
                # self.pt_index += 1
                
                # if track_neg==False:
                #     self.ndlist[self.pt_index-2].nbr.append(self.pt_index-1) 
                #     self.ndlist[self.pt_index-1].nbr.append(self.pt_index-2)
                # else:
                #     self.ndlist[self.position_id].nbr.append(self.pt_index-1) 
                #     self.ndlist[self.pt_index-1].nbr.append(self.position_id)
                #     track_neg = False
                    
                # self.ndlist[pix_id-1].nbr.append(self.pt_index-1)
                # self.ndlist[self.pt_index-1].nbr.append(pix_id-1)
                break

            # # 脱离中心线
            # if self.dis2centerline[position_1[2], position_1[0], position_1[1]] > 3:
            #     break
            
            # SPE for feature extraction
            Spherical_patch = Spherical_Patches_Extraction(self.img2, position, 
                                                           self.n, self.sphere_core, 
                                                           self.node_step)
            SP = Spherical_patch.reshape([1, self.Ma, self.Mp, self.n-1]).transpose([0,3,1,2])
            SP = np.asarray(SP)
            pmax = SP.max()
            
            if pmax > 0:
                SP = SP/pmax
            
            data = torch.from_numpy(SP)
            inputs = data.type(torch.FloatTensor).to(self.device) 
            with torch.no_grad():
                outputs1, stop_flag = self.model(inputs)
            
            outputs = outputs1
            if stop_flag.shape[1] > 2:
                radii = self.relu(stop_flag[:,-1,:,:]).cpu().detach().numpy().squeeze() + 1
                stop_flag = stop_flag[:,:2,:,:]
            else:
                radii = 1
            
            outputs = torch.nn.functional.softmax(outputs, 1)
            stop_flag = torch.nn.functional.softmax(stop_flag, 1)
            direction_vector = outputs.cpu().detach().numpy().reshape([self.K, 1])
            stop_flag = stop_flag.cpu().detach().numpy().squeeze().argmax()
            if stop_flag == 1:
                self.terminated_location.append(position)
                # print('terminated by discriminater', cc)
                break
            
            soma_reached = self.soma_mask2[position_1[2], position_1[0], position_1[1]]
            if soma_reached:
                self.soma_location.append(position)
                # print('Soma Reached', cc)
                nd = Node(position, direction_vector.max(), radii, 1)
                self.ndlist.append(nd)
                self.rlist.append(radii)
                self.pt_index += 1
                # biuld connection between the met points
                if track_neg==False:
                    self.ndlist[self.pt_index - 2].nbr.append(self.pt_index - 1) 
                    self.ndlist[self.pt_index - 1].nbr.append(self.pt_index - 2)
                else:
                    self.ndlist[self.position_id].nbr.append(self.pt_index - 1) 
                    self.ndlist[self.pt_index - 1].nbr.append(self.position_id)
                    track_neg = False
                break
            
            cos_angle = np.sum(direction1 * self.sphere_core_label, axis=1)
            cos_angle[cos_angle > 1] = 1
            cos_angle[cos_angle < -1] = -1
            angle = np.arccos(cos_angle).reshape([self.K, 1])
            direction_vector[angle > self.angle_T] = 0
            max_id = np.argmax(direction_vector)
            _confidence = direction_vector[max_id]
            
            nd = Node(position, _confidence, radii)
            self.ndlist.append(nd)
            self.rlist.append(radii)
            self.pt_index += 1
            
            # biuld connection between the met points
            if track_neg==False:
                self.ndlist[self.pt_index - 2].nbr.append(self.pt_index - 1) 
                self.ndlist[self.pt_index - 1].nbr.append(self.pt_index - 2)
            else:
                self.ndlist[self.position_id].nbr.append(self.pt_index - 1) 
                self.ndlist[self.pt_index - 1].nbr.append(self.position_id)
                track_neg = False
            
            self.track.append(position)
             
            self.confidence_track.append(_confidence)
            
            # joint decision
            dist_position_seed = np.sum(np.square(position - self.terminations[:, :3]), axis=1)
            seed_remain = np.where((dist_position_seed < (self.R * radii * radii)) & (dist_position_seed > 1.5))# find the closest seed point to current position
            remain_size = seed_remain[0].size
            if remain_size == 0:
                direction1 = self.sphere_core_label[max_id, :]
            else:
                seed_remain_position = self.terminations[seed_remain[0], :3]
                ds_vectors = (seed_remain_position - position) / np.linalg.norm(seed_remain_position - position, axis=1).reshape([remain_size, 1])# vector from current positon to the closest seed
                ds_cos = np.sum(ds_vectors * direction1, axis=1)
                ds_cos[ds_cos > 1] = 1
                ds_cos[ds_cos < -1] = -1
                ds_angle = np.arccos(ds_cos)
                ds_angle_min = ds_angle.min()
                if ds_angle_min > self.angle_T:
                    direction1 = self.sphere_core_label[max_id, :]
                else:
                    vid = np.argmin(ds_angle)
                    if _confidence * 2 < self.terminations[seed_remain[0][vid], 3] / 255:
                        direction1 = ds_vectors[vid]
                        correct_flag = 1
                        next_position = seed_remain_position[vid, :]
                    else:                    
                        direction1 = self.sphere_core_label[max_id, :]
            
        if cc == self.max_iter:
            self.success_location.append(position)
            
def constrain_range(min, max, minlimit, maxlimit):
    return list(
        range(min if min > minlimit else minlimit, max
              if max < maxlimit else maxlimit))
        

def trace_spe_dnr(raw, seg, out_path, soma_mask=None):
    if isinstance(raw, str) or isinstance(raw, Path):
        img_raw = tifffile.imread(raw)
    else:
        img_raw = raw
    img = (img_raw - img_raw.min())/(img_raw.max() - img_raw.min())
    if isinstance(seg, str) or isinstance(seg, Path):
        img_seg = tifffile.imread(seg)
    else:
        img_seg = seg
    
    ## Parameters
    Ma = 32
    Mp = 32
    K = 1024
    Mc = int(np.sqrt(K))
    Lambda = 4 # can be tuned between 1 to 4 for optimal result
    angle_T = np.pi/3 # angle threshold
    n = 10 # maximun radius of the spherical core
    psize = n
    node_step = 1
    max_iter = 1000
    step_size = 1
    mask_size = 3
    sphere_core, _, _ = generate_sphere(Ma, Mp) # for spherical patches extraction
    sphere_core_label, _, _ = generate_sphere(Mc, Mc) # for direction determination

    ## model loading
    best_i, best_free, info_list = get_gpu_with_max_free_vram()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{best_i}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CenterlineNet_Discrimintor_2D_Radii_32(NUM_ACTIONS=K, n=n).to(device)
    checkpoint_path = os.path.join(CURRENT_DIR, 'weights', 'spe-dnr-weight.pkl')
    # checkpoint = torch.load(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['net_dict'])    
    model.eval()

    if img_seg.max() <= 1:
        img_seg = img_seg*255 
    # image padding
    img = np.pad(img, ((psize,psize),(psize,psize),(psize,psize)), 'constant',
                  constant_values=((0.0,0.0),(0.0,0.0),(0.0,0.0)))  
    if soma_mask is not None:
        soma_mask = np.pad(soma_mask, ((psize,psize),(psize,psize),(psize,psize)),
                            'constant', constant_values=((0.0,0.0),(0.0,0.0),(0.0,0.0)))
    else:
        soma_mask = np.zeros_like(img)
    img_seg2 = np.pad(img_seg, ((psize,psize),(psize,psize),(psize,psize)),
                        'constant', constant_values=((0.0,0.0),(0.0,0.0),(0.0,0.0)))
    Zb,Xb,Yb = np.shape(img)
    
    img_seg_bin = (img_seg2>0.5*255)
    dis2centerline = distance_transform_edt(1-img_seg_bin)

    # find seeds for tracing
    _, candidate_sup = local_max(img_seg, wsize=3, thre=0.5*255)
    candidate_file = np.array(candidate_sup)
    candidate_file = candidate_file[np.argsort(-candidate_file[:,-1])]  
    candidate_file[:, :3] += psize

    # tracing
    tracker = Tracker(img, soma_mask, candidate_file, dis2centerline, Lambda, K,
                      angle_T, max_iter, step_size, node_step, mask_size, psize,
                      Ma, Mp, n, Xb, Yb, Zb, 
                      model, sphere_core, sphere_core_label, device)    
    tracker.trace_JointDecision()
    
    # graph reconstruction
    n0 = tracker.ndlist
    tree = compute_trees(n0)
    swc = build_nodelist(tree)
    swc[:,2:5] = swc[:,2:5] - psize

    # save to swc
    saveswc(out_path, swc)


if __name__ == '__main__':
    img_path = '/data1/yangzekang/neuron/neuron-trace/examples/real/id1_f3_b0_raw.tif'
    seg_path = '/data1/yangzekang/neuron/neuron-trace/examples/real/id1_f3_b0_mask.tif'
    out_path = '/data1/yangzekang/neuron/neuron-trace/examples/real/id1_f3_b0_pred-spe-dnr.swc'

    trace_spe_dnr(img_path, seg_path, out_path)

