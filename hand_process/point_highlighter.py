import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt
import json
import torch
import xml.etree.ElementTree as ET
import os
from matplotlib.colors import CSS4_COLORS,to_rgb
import transforms3d.quaternions as tq



def quaternion_multiply(q1, q2):
    return tq.qmult(q1, q2)

def rotation_matrix_from_quaternion(quat_angle):
    x = quat_angle[0]
    y = quat_angle[1]
    z = quat_angle[2]
    w = quat_angle[3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_x), -np.sin(roll_x)],
                    [0, np.sin(roll_x), np.cos(roll_x)]])

    R_y = np.array([[np.cos(pitch_y), 0, np.sin(pitch_y)],
                    [0, 1, 0],
                    [-np.sin(pitch_y), 0, np.cos(pitch_y)]])

    R_z = np.array([[np.cos(yaw_z), -np.sin(yaw_z), 0],
                    [np.sin(yaw_z), np.cos(yaw_z), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def combine_transformations(body_pos, body_quat, geom_pos, geom_quat=None):
    R_body = rotation_matrix_from_quaternion(body_quat)
    global_geom_pos = np.dot(R_body, geom_pos) + body_pos

    '''if geom_quat is not None:
        R_geom = rotation_matrix_from_quaternion(geom_quat)
        R_combined = np.dot(R_body, R_geom)
    else:
        R_combined = R_body'''
    if geom_quat is not None:
        global_geom_quat = quaternion_multiply(body_quat, geom_quat)
    else:
        global_geom_quat = body_quat

    return global_geom_pos, global_geom_quat


'''def parse_robot_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    print(root)

    bodies = {}
    for body in root.findall('.//body'):
        body_name = body.get('name')
        print(body_name)
        pos = [float(x) for x in body.get('pos', '0 0 0').split()]
        quat = [float(x) for x in body.get('quat', '0 0 0 0').split()] if body.get('quat') else [0, 0, 0, 0]

        bodies[body_name] = {
            'pos': pos,
            'quat': quat,
            'geoms': []
        }

        for geom in body.findall('geom'):
            geom_name = geom.get('name')
            print(geom_name)
            geom_pos = [float(x) for x in geom.get('pos', '0 0 0').split()]
            geom_quat = [float(x) for x in geom.get('quat', '0 0 0 0').split()] if geom.get('quat') else [0, 0, 0, 0]
            geom_size = [float(x) for x in geom.get('size', '0 0 0').split()]
            geom_mesh = geom.get('mesh', None)

            if geom_mesh:  # Only consider geoms with meshes
                bodies[body_name]['geoms'].append({
                    'name': geom_name,
                    'pos': geom_pos,
                    'quat': geom_quat,
                    'size': geom_size,
                    'mesh': geom_mesh
                })

    return bodies'''



'''def get_global_geom_positions(robot_bodies):
    global_geom_positions = {}

    for body_name, body_info in robot_bodies.items():
        body_pos = np.array(body_info['pos'])
        body_quat = np.array(body_info['quat'])

        for geom in body_info['geoms']:
            geom_mesh = geom['mesh']
            geom_pos = np.array(geom['pos'])
            geom_quat = np.array(geom['quat'])
            global_pos, global_rot = combine_transformations(body_pos, body_quat, geom_pos, geom_quat)
            global_geom_positions[body_name] = (global_pos, global_rot)

    return global_geom_positions'''


def parse_robot_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    bodies = {}
    for body in root.findall('.//body'):
        body_name = body.get('name')
        pos = [float(x) for x in body.get('pos', '0 0 0').split()]
        quat = [float(x) for x in body.get('quat', '0 0 0 0').split()] if body.get('quat') else [0, 0, 0, 0]

        bodies[body_name] = {
            'name': body_name,
            'pos': pos,
            'quat': quat,
            'geoms': [],
            'children': []
        }

        for geom in body.findall('geom'):
            geom_name = geom.get('name')
            geom_pos = [float(x) for x in geom.get('pos', '0 0 0').split()]
            geom_quat = [float(x) for x in geom.get('quat', '0 0 0 0').split()] if geom.get('quat') else [0, 0, 0, 0]
            geom_size = [float(x) for x in geom.get('size', '0 0 0').split()]
            geom_mesh = geom.get('mesh', None)

            if geom_mesh:  # Only consider geoms with meshes
                bodies[body_name]['geoms'].append({
                    'name': geom_name,
                    'pos': geom_pos,
                    'quat': geom_quat,
                    'size': geom_size,
                    'mesh': geom_mesh
                })

    for body in root.findall('.//body'):
        parent_name = body.get('name')
        for child in body.findall('body'):
            child_name = child.get('name')
            bodies[parent_name]['children'].append(child_name)

    return bodies

def get_global_geom_positions(robot_bodies, root_body_name, parent_pos=None, parent_quat=None):
    if parent_pos is None:
        parent_pos = np.array([0, 0, 0])
    if parent_quat is None:
        parent_quat = np.array([0, 0, 0, 1])  # Identity quaternion

    global_geom_positions = {}

    body_stack = [(root_body_name, parent_pos, parent_quat)]

    while body_stack:
        current_body, accumulated_pos, accumulated_quat = body_stack.pop()
        body_info = robot_bodies[current_body]

        current_pos = np.array(body_info['pos'])
        current_quat = np.array(body_info['quat'])

        global_pos, global_quat = combine_transformations(accumulated_pos, accumulated_quat, current_pos, current_quat)

        for geom in body_info['geoms']:
            geom_mesh = geom['mesh']
            geom_pos = np.array(geom['pos'])
            geom_quat = np.array(geom['quat'])
            global_geom_pos, global_geom_quat = combine_transformations(global_pos, global_quat, geom_pos, geom_quat)
            global_geom_positions[body_info['name']] = (global_geom_pos, global_geom_quat)

        for child in body_info['children']:
            body_stack.append((child, global_pos, global_quat))

    return global_geom_positions


def transform_points_to_global(points, geom_positions):
    new_points = {}
    for key, value in points.items():
        new_key = key.replace('_child', '')
        new_points[new_key] = value

    global_points = {}

    for geom_name, local_points in new_points.items():
        if geom_name in geom_positions:
            global_geom_pos, global_quat = geom_positions[geom_name]
            global_rot = rotation_matrix_from_quaternion(global_quat)

            for point in local_points:
                local_point = np.array(point)
                rotated_point = np.dot(global_rot, local_point)
                global_point = global_geom_pos + rotated_point
                if geom_name not in global_points:
                    global_points[geom_name] = []
                global_points[geom_name].append(global_point)
        else:
            print("point list component not used in asset: " + geom_name)
    return global_points


# Generate a distinct color for each mesh
def generate_color_palette(n):
    color_names = list(CSS4_COLORS.keys())
    selected_colors = color_names[:n]
    colors = [CSS4_COLORS[color] for color in selected_colors]
    return selected_colors,colors


def create_sphere_actor(gym, sim, env, point, color):
    sphere_opts = gymapi.AssetOptions()
    sphere_opts.disable_gravity = True
    sphere_asset = gym.create_sphere(sim, 0.001, sphere_opts)  # Radius of 0.01

    sphere_pose = gymapi.Transform()
    sphere_pose.p = gymapi.Vec3(point[0], point[1], point[2])

    # Create actor
    actor_handle = gym.create_actor(env, sphere_asset, sphere_pose, "point_sphere", 0, -1)

    # Set color
    gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)


def highlight_points(gym,sim, env,points_path,hand_path,pose):
    points_dict = json.load(open(points_path, 'r')) if points_path is not None else None
    robot_bodies = parse_robot_xml(hand_path)
    root_body_name = list(robot_bodies.keys())[0]
    # change pose of root body in worldbody based on pose.r and pose.p
    #robot_bodies[root_body_name]['pos'] = [pose.p.x, pose.p.y, pose.p.z]
    #robot_bodies[root_body_name]['quat'] = [pose.r.x, pose.r.y, pose.r.z, pose.r.w]  # Set the orientation directly from pose

    #global_geom_pos = get_global_geom_positions(robot_bodies)
    global_geom_pos = get_global_geom_positions(robot_bodies,root_body_name,parent_pos=np.array([pose.p.x, pose.p.y, pose.p.z]),parent_quat=np.array([pose.r.x, pose.r.y, pose.r.z, pose.r.w]))
    global_points = transform_points_to_global(points_dict,global_geom_pos)
    body_names = list(robot_bodies.keys())
    num_bodies = len(body_names)
    colors_names,colors = generate_color_palette(num_bodies)
    color_mapping = {body_names[i]: gymapi.Vec3(*to_rgb(colors[i])) for i in range(num_bodies)}
    print('Highlighting Colors: ' + str([(body_names[i], colors_names[i]) for i in range(num_bodies)]))
    for geom_name, global_point_list in global_points.items():
        color = color_mapping.get(geom_name, gymapi.Vec3(1, 1, 1))  # Default to white if not found
        for point in global_point_list:
            create_sphere_actor(gym,sim, env, point, color)


if __name__ == '__main__':
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()

    args = gymutil.parse_arguments(
        description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
        custom_parameters=[
            {"name": "--asset_root", "type": str, "default": './shadow_hand', "help": "Root directory of asset to use"},
            {"name": "--asset_file", "type": str, "default": 'shadow_hand.xml', "help": "Name of asset to use"},
            {"name": "--worldbody_file", "type": str, "default": 'robot.xml', "help": "Asset file used for world body"},
            {"name": "--contact_points_file", "type": str, "default": 'contact_points.json', "help": "Name of contact points to use"}])

    if args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.shape_collision_margin = 0.25
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.substeps = 1
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)



    # load hand asset
    hand_asset_options = gymapi.AssetOptions()
    hand_asset_options.disable_gravity = True
    hand_asset_options.fix_base_link = True
    hand_asset_options.collapse_fixed_joints = True
    '''
    hand_asset_options.flip_visual_attachments = False
    hand_asset_options.fix_base_link = True
    hand_asset_options.collapse_fixed_joints = True
    hand_asset_options.disable_gravity = True
    hand_asset_options.thickness = 0.001
    hand_asset_options.angular_damping = 0.01
    # Convex decomposition
    hand_asset_options.vhacd_enabled = True
    hand_asset_options.vhacd_params.resolution = 100000
    # hand_asset_options.vhacd_params.max_convex_hulls = 30
    # hand_asset_options.vhacd_params.max_num_vertices_per_ch = 64
    hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    '''
    asset = gym.load_asset(sim, args.asset_root, args.asset_file, hand_asset_options)

    # create env
    spacing = 5
    lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    upper = gymapi.Vec3(spacing, spacing, spacing)
    env = gym.create_env(sim, lower,upper, 1)

    # create actor
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    pose.p = gymapi.Vec3(0,1,0)
    ahandle = gym.create_actor(env, asset, pose, 'hand',0,-1,0)

    # create highlight spheres
    points_path = os.path.join(args.asset_root,args.contact_points_file)
    hand_path = os.path.join(args.asset_root,args.worldbody_file)
    highlight_points(gym,sim, env,points_path,hand_path,pose)

    # create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()

    # position viewer
    gym.viewer_camera_look_at(viewer, env, gymapi.Vec3(0.05, 0.8, 0.1), gymapi.Vec3(0.05, 1, 0.09))

    #start viewing
    while not gym.query_viewer_has_closed(viewer):
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        #gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)








