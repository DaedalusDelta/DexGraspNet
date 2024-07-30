import os
import sys


import argparse
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
import math
import random
import transforms3d

from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
from utils.energy import cal_energy
from utils.optimizer import Annealing
from utils.rot6d import robust_compute_rotation_matrix_from_ortho6d

from torch.multiprocessing import set_start_method


import plotly.graph_objects as go

try:
    set_start_method('spawn')
except RuntimeError:
    pass

sys.path.append(os.path.realpath('.'))


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.seterr(all='raise')

device = torch.device('cuda')

'''hand_model = HandModel(
    mjcf_path='mjcf/shadow_hand_wrist_free.xml',
    mesh_path='mjcf/meshes',
    contact_points_path='mjcf/contact_points.json',
    penetration_points_path='mjcf/penetration_points.json',
    device=device
)'''


hand_model = HandModel(
    mjcf_path='./leap_aligned_fingers_2/mjmodel.xml',
    mesh_path='./leap_aligned_fingers_2',
    contact_points_path='./leap_aligned_fingers_2/contact_points.json',
    penetration_points_path='./leap_aligned_fingers_2/penetration_points.json',
    device=device
)

print(hand_model.joints_names)

'''translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = ['index_mcpf', 'index_mcps', 'index_pip', 'index_dip', 'middle_mcpf', 'middle_mcps', 'middle_pip',
               'middle_dip', 'ring_mcpf', 'ring_mcps', 'ring_pip', 'ring_dip', 'pinky_mcpf', 'pinky_mcps', 'pinky_pip',
               'pinky_dip', 'palm_4_finger', 'palm_thumb', 'thumb_mcpf', 'thumb_mcps', 'thumb_ip']

joint_angles = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=device)
rotation = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 2, 0, axes='rzxz'), dtype=torch.float, device=device)
hand_pose = torch.cat([torch.tensor([0, 0, 0], dtype=torch.float, device=device), rotation.T.ravel()[:6], joint_angles])
hand_model.set_parameters(hand_pose.unsqueeze(0))
hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.5, color='lightblue', with_contact_points=False)
fig = go.Figure(hand_plotly)
fig.update_layout(scene_aspectmode='data')
fig.show()'''



object_model = ObjectModel(
    data_root_path= '/home/mli/DexGraspNet/data/meshdata',
    batch_size_each=1,
    num_samples=1,
    device=device
)
object_model.initialize(['mujoco-Ecoforms_Plant_Plate_S11Turquoise'])

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--result_path', default="../data/graspdata", type=str)
parser.add_argument('--data_root_path', default="../data/meshdata", type=str)
parser.add_argument('--object_code_list', nargs='*', type=str)
parser.add_argument('--all', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--todo', action='store_true')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--n_contact', default=4, type=int)
parser.add_argument('--batch_size_each', default=500, type=int)
parser.add_argument('--max_total_batch_size', default=1000, type=int)
parser.add_argument('--n_iter', default=6000, type=int)
# hyper parameters
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--step_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_spen', default=10.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0.1, type=float)
parser.add_argument('--distance_lower', default=0.2, type=float)
parser.add_argument('--distance_upper', default=0.3, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()
initialize_convex_hull(hand_model, object_model, args)

hand_en_plotly = hand_model.get_plotly_data(i=0, opacity=1, color='lightblue', with_contact_points=False)
object_plotly = object_model.get_plotly_data(i=0, color='lightgreen', opacity=1)
fig = go.Figure(hand_en_plotly + object_plotly)
fig.update_layout(scene_aspectmode='data')
fig.show()