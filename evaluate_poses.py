import os.path as osp
import time
from glob import glob
import os
import numpy as np
import pybullet as p
import pybullet_data
import trimesh.transformations as T
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Simulate pose.')
parser.add_argument('--obj_path', type=str, default='./test_urdf/folk2r651.urdf')
parser.add_argument('--sup_path', type=str, default='./test_urdf/bowl1r182.urdf')
parser.add_argument('--init_obj_pose', type=str, default='./test_plys/folk2r651_init_pose.txt')
parser.add_argument('--init_sup_pose', type=str, default='./test_plys/bowl1r182_init_pose.txt')
parser.add_argument('--transforms', type=str, default='')
parser.add_argument('--save_dir', type=str, default='./debug',
                    help='path to save records.')
parser.add_argument('--render', action='store_true')

args = parser.parse_args()

MAX_OBS_TIME=20
LIN_V_TH=0.005
ANG_V_TH=0.1

# set up pybullet environment
if args.render:
	physicsClient = p.connect(p.GUI) # turn off
else:
	physicsClient = p.connect(p.DIRECT) # turn off

p.setAdditionalSearchPath(pybullet_data.getDataPath())

def step_simulation(n, p):
	for i in range(n):
		p.stepSimulation()

def p_enable_physics(p):
	p.setPhysicsEngineParameter(enableConeFriction=1)
	p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
	p.setPhysicsEngineParameter(allowedCcdPenetration=0.0)
	p.setPhysicsEngineParameter(numSolverIterations=40)
	p.setPhysicsEngineParameter(numSubSteps=40)
	p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG, globalCFM=0.000001)
	p.setPhysicsEngineParameter(enableFileCaching=0)
	p.setTimeStep(1 / 100.0)
	p.setGravity(0, 0, -9.81)

p_enable_physics(p)
p.resetDebugVisualizerCamera(cameraDistance=0.4, cameraYaw=30, cameraPitch=-50, cameraTargetPosition=[0,0,0])
planeId = p.loadURDF("plane.urdf")

def load_transforms(pose_dir,):
	all_files = glob('{}/*.npy'.format(pose_dir))
	all_transforms = np.array([np.load(f)[0] for f in all_files]) # (n, 4, 4)
	return all_transforms

def simulate(p, sup, obj):
	p.resetBaseVelocity(sup, [0, 0, 0], [0, 0, 0])
	p.resetBaseVelocity(obj, [0, 0, 0], [0, 0, 0])
	tic = time.time()
	obj_trs, obj_quat = p.getBasePositionAndOrientation(obj)
	sup_trs, sup_quat = p.getBasePositionAndOrientation(sup)

	while True:
		step_simulation(1, p)
		cur_obj_trs, cur_obj_quat = p.getBasePositionAndOrientation(obj)
		cur_sup_trs, cur_sup_quat = p.getBasePositionAndOrientation(sup)
		if not same_pose(init_pose_7q=obj_trs + obj_quat,
		                 cur_pose_7q=cur_obj_trs + cur_obj_quat,
		                 dist_th=0.03, ang_th=30):
			# print('object change! ')
			return False

		if not same_pose(init_pose_7q=sup_trs + sup_quat,
		                 cur_pose_7q=cur_sup_trs + cur_sup_quat,
		                 dist_th=0.02, ang_th=10):
			# print('support change! ')
			return False

		if time.time() - tic > MAX_OBS_TIME or simulation_stoped(p, obj):
			return True

def same_pose(init_pose_7q, cur_pose_7q, dist_th, ang_th):
	assert len(init_pose_7q) == len(cur_pose_7q)
	assert len(init_pose_7q) == 7

	init_pose_7q = np.array(init_pose_7q)
	cur_pose_7q = np.array(cur_pose_7q)
	init_quat = init_pose_7q[3:]
	cur_quat = cur_pose_7q[3:]
	init_trs = init_pose_7q[:3]
	cur_trs = cur_pose_7q[:3]

	assert len(cur_trs) == 3
	assert len(cur_quat) == 4

	tmp = np.clip(np.abs(np.sum(init_quat*cur_quat)), 0., 1., )
	deg_diff = 2 * 180 / np.pi * np.arccos(tmp)

	if deg_diff > ang_th:
		# print('degree change: ', deg_diff)
		return False

	trs_diff = np.sqrt(((init_trs - cur_trs)**2).sum())
	if trs_diff > dist_th:
		# print('position change: ', trs_diff)
		return False

	return True

def simulation_stoped(p, sID):
	lin_v, ang_v = p.getBaseVelocity(sID)
	if np.allclose(np.array(lin_v), np.zeros((3, )), rtol=1, atol=LIN_V_TH) \
			and np.allclose(np.array(ang_v), np.zeros((3, )), rtol=1, atol=ANG_V_TH):
		return True
	else:
		return False

def set_transform(subjectId, transform):
	trs = transform[:3, 3]
	rot = transform[:3, :3]
	euler = T.euler_from_matrix(rot, 'sxyz')
	quat = p.getQuaternionFromEuler(euler)
	p.resetBasePositionAndOrientation(subjectId, trs, quat)

def transform_to_pose7(transform):
	transform = np.array(transform)
	euler = T.euler_from_matrix(transform[:3, :3], 'sxyz')
	trs = transform[:3, 3]
	quat = p.getQuaternionFromEuler(euler)
	pose7 = trs.tolist() + list(quat)
	return pose7

def count_diff_pose(all_poses):
	pose_buff = []
	for T in all_poses:
		if not pose_buff:
			pose_buff.append(T)
		else:
			uniq = True
			for Tb in pose_buff:
				Tb_7 = transform_to_pose7(Tb)
				T_7 = transform_to_pose7(T)
				if same_pose(Tb_7, T_7, 0.03, 30):
					uniq = False
					break
			if uniq:
				pose_buff.append(T)

	return len(pose_buff)

def write_file(path, data_list):
	dir_name = osp.dirname(path)
	if dir_name:
		os.makedirs(dir_name, exist_ok=True)
	with open(path, 'w') as f:
		json.dump(data_list, f)

def process_one_pair(supportId, objectId, all_transforms, init_sup_pose, init_obj_pose):

	results = []
	for transform in tqdm(all_transforms):

		p.resetBaseVelocity(supportId, [0, 0, 0], [0, 0, 0])
		p.resetBaseVelocity(objectId, [0, 0, 0], [0, 0, 0])
		p.resetBasePositionAndOrientation(objectId, [0, 0, 10],
		                                  p.getQuaternionFromEuler([0, 0, 0]))

		set_transform(supportId, init_sup_pose)

		tic = time.time()
		while True:
			step_simulation(1, p)
			if simulation_stoped(p, supportId) or time.time() - tic > MAX_OBS_TIME:
				step_simulation(5, p)
				break

		real_transform = transform @ init_obj_pose
		real_transform[2, 3] += 0.01
		set_transform(objectId, real_transform)

		p.stepSimulation()

		contacts1 = p.getContactPoints(supportId, objectId)
		contacts2 = p.getContactPoints(planeId, objectId)
		contacts = contacts1 + contacts2

		fail = False
		for c in contacts:
			if c[8] < -0.00001:
				fail = True
				break

		if not fail:
			result = simulate(p, supportId, objectId)
		else:
			result = False
		results.append(result)

	return np.array(results)

def process(args):
	supportId = p.loadURDF(args.sup_path,)
	objectId = p.loadURDF(args.obj_path,)

	init_sup_pose = np.loadtxt(args.init_sup_pose)
	init_obj_pose = np.loadtxt(args.init_obj_pose)

	all_transforms = load_transforms(args.transforms)

	all_paths = glob('{}/*.npy'.format(args.transforms))
	probs = np.array([float(osp.basename(f).split('_')[0]) for f in all_paths])
	thresh = [0.6, 0.7, 0.8]

	all_transforms = all_transforms[probs >= thresh[0]]
	probs = probs[probs >= thresh[0]]


	os.makedirs(args.save_dir, exist_ok=True)
	sup_name = osp.basename(args.sup_path).split('.')[0]
	obj_name = osp.basename(args.obj_path).split('.')[0]

	results = process_one_pair(supportId=supportId, objectId=objectId,
	                           all_transforms=all_transforms,
	                           init_obj_pose=init_obj_pose,
	                           init_sup_pose=init_sup_pose)


	report = {'cnt': {}, 'acc': {}}
	for th in thresh:
		inds = probs > th
		acc_t = np.nanmean(results[inds])

		select_poses = all_transforms[(probs > th) * results]
		cnt_t = count_diff_pose(select_poses)

		report['acc'][th] = str(acc_t)
		report['cnt'][th] = str(cnt_t)

	print(report)
	save_path = '{}/{}-{}.json'.format(args.save_dir, sup_name, obj_name)
	write_file(save_path, [report])


process(args)
p.disconnect()

