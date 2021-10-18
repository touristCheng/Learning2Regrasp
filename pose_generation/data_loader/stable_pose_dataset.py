import torch
from torch.utils.data import Dataset

import numpy as np
import os.path as osp
import open3d as o3d
import json
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import os

def write_file(path, data_list):
	dir_name = osp.dirname(path)
	if dir_name:
		os.makedirs(dir_name, exist_ok=True)
	with open(path, 'w') as f:
		json.dump(data_list, f)

def read_file(path):
	with open(path, 'r') as fd:
		data_list = json.load(fd, object_hook=OrderedDict)
		return data_list

def parse_data(data_pairs):
	name2data = {}
	for pair in data_pairs:
		pair_id = pair['pair_id']
		if not pair['stable']:
			print("{}_{:06d} not stable, skip.".format(pair_id, pair['index']))
			continue
		if len(pair['support_contact']) == 0:
			print("{}_{:06d} no support contact, skip.".format(pair_id, pair['index']))
			continue

		transform = np.array(pair['transform'], dtype='float32')
		if pair_id in name2data:
			name2data[pair_id]['stable_transforms'].append(transform)
		else:
			name2data[pair_id] = {'sup_init_path': pair['sup_init_path'],
			                      'obj_init_path': pair['obj_init_path'],
			                      'stable_transforms': [transform, ]
			                      }
	print('stable data pairs: ', name2data.keys())
	return list(name2data.values())

def load_data(root_dir, list_path):
	data_list = open(list_path, 'r').readlines()
	data_list = list(map(str.strip, data_list))
	all_pairs = []
	for line in data_list:
		one_pair = read_file(osp.join(root_dir, line))
		all_pairs += one_pair
	return parse_data(all_pairs)

def load_ply(path, pc_len):
	pcd = o3d.io.read_point_cloud(path)
	point_cloud = np.asarray(pcd.points)
	colors = np.asarray(pcd.colors)
	if len(point_cloud) < pc_len:
		ind = np.random.choice(len(point_cloud), pc_len-len(point_cloud))
		point_cloud = np.concatenate([point_cloud, point_cloud[ind]], 0)
	elif len(point_cloud) > pc_len:
		ind = np.random.choice(len(point_cloud), pc_len)
		point_cloud = point_cloud[ind]
	return point_cloud

def load_mv_ply(root_dir, path, pc_len):
	flag = float(np.random.uniform(0, 1, ()))
	if flag < 0.25:
		num_v = 1
	else:
		num_v = 2
	v_inds = np.random.choice(range(4), num_v, replace=False)
	pcs = []
	for i in range(num_v):
		path_i = path.rsplit('.ply', 1)[0]+'.v{:04d}.ply'.format(int(v_inds[i]))
		pcs.append(load_ply(osp.join(root_dir, path_i), pc_len))
	point_cloud = np.concatenate(pcs, 0)
	if len(point_cloud) < pc_len:
		ind = np.random.choice(len(point_cloud), pc_len-len(point_cloud))
		point_cloud = np.concatenate([point_cloud, point_cloud[ind]], 0)
	elif len(point_cloud) > pc_len:
		ind = np.random.choice(len(point_cloud), pc_len)
		point_cloud = point_cloud[ind]
	return point_cloud

class StablePoseDataset(Dataset):
	def __init__(self, root_dir, list_path, pose_num=256, pc_len=1024, use_aug=True):
		super(StablePoseDataset, self).__init__()
		self.root_dir = root_dir
		self.pose_num = pose_num
		self.pc_len = pc_len
		self.use_aug = use_aug

		data_pairs = load_data(root_dir, list_path)
		np.random.shuffle(data_pairs)

		self.data_pairs = data_pairs
		print('Total pairs [{}]: '.format(list_path), len(self.data_pairs))

	def __getitem__(self, index):
		sup_ply = load_mv_ply(self.root_dir, self.data_pairs[index]['sup_init_path'],
		                      pc_len=self.pc_len)
		obj_ply = load_mv_ply(self.root_dir, self.data_pairs[index]['obj_init_path'],
		                      pc_len=self.pc_len)
		# (N, 3), (N, 3)

		stable_transform = np.array(self.data_pairs[index]['stable_transforms'], dtype='float32')
		# (M, 4, 4)

		if len(stable_transform) >= self.pose_num:
			select_inds = np.random.choice(len(stable_transform),
			                               self.pose_num, replace=False)
		else:
			select_inds = np.random.choice(len(stable_transform),
			                               self.pose_num, replace=True)
		stable_transform = stable_transform[select_inds.tolist()]
		assert stable_transform.shape == (self.pose_num, 4, 4), stable_transform.shape

		if self.use_aug:
			sup_ply, obj_ply, stable_transform = random_transform_pair(support=sup_ply,
			                                                          object=obj_ply,
			                                                          transforms=stable_transform)

		ret = {'support': sup_ply.T, 'object': obj_ply.T,
		       'transforms': stable_transform,
		       'sup_path': self.data_pairs[index]['sup_init_path'],
		       'obj_path': self.data_pairs[index]['obj_init_path']}
		return ret

	def __len__(self):
		return len(self.data_pairs)

def apply_transform(t, points):
	'''

	:param t: (4, 4)
	:param points: (N, 3)
	:return:
	'''
	N = points.shape[0]
	ones = np.ones((N, 1))
	points = np.concatenate([points, ones], 1) # (N, 4)
	points = np.expand_dims(points, 2) # (N, 4, 1)
	t = np.expand_dims(t, 0) # (1, 4, 4)
	points = np.matmul(t, points)[:, :3, 0] # ()
	return points


def random_transform_pair(support, object, transforms):
	'''

	:param support: (N, 3)
	:param object: (N, 3)
	:param transforms: (M, 4, 4)
	:return:
	'''

	degs = np.random.uniform(0, 360, size=(2, ))
	r = R.from_euler('z', degs, degrees=True)

	t0 = np.eye(4)
	t1 = np.eye(4)
	t0[:3, :3] = r.as_matrix()[0]
	t1[:3, :3] = r.as_matrix()[1]

	xyz_range = np.array([[-0.005, -0.005, -0.005],
	                      [0.005, 0.005, 0.005]])
	scales = np.random.uniform(0, 1, size=(2, 3))
	offset = scales * (xyz_range[1:2]-xyz_range[0:1]) + xyz_range[0:1]

	t0[:3, 3] = offset[0]
	t1[:3, 3] = offset[1]

	object_t = apply_transform(t0, object)
	support_t = apply_transform(t1, support)

	t0 = np.expand_dims(t0, axis=0)
	t1 = np.expand_dims(t1, axis=0)
	transforms_t = np.matmul(t1, np.matmul(transforms, np.linalg.inv(t0)))

	sigma = 0.003
	noise_s = np.random.normal(0, sigma, size=(support_t.shape[0], 3))
	support_t += noise_s
	noise_o = np.random.normal(0, sigma, size=(object_t.shape[0], 3))
	object_t += noise_o

	return support_t, object_t, transforms_t



if __name__ == '__main__':
	from torch.utils.data import DataLoader

	dataset = StablePoseDataset('../../dataset',
                                list_path='../../dataset/data_list/train_generator.txt')
	loaderV = DataLoader(dataset, 2, sampler=None, num_workers=1,
                         drop_last=False, shuffle=False)

	def visualize(sup, obj, ):
		pcd = o3d.geometry.PointCloud()
		points = np.concatenate([sup, obj], 0)
		colors = np.zeros((len(points), 3))
		colors[:len(sup), 2] = 1.
		colors[len(sup):, 1] = 1.
		pcd.points = o3d.utility.Vector3dVector(points)
		pcd.colors = o3d.utility.Vector3dVector(colors)

		o3d.visualization.draw_geometries([pcd])

	for i, data in enumerate(iter(dataset)):
		print(data.keys())
		sup_ply = data['support'].T
		obj_ply = data['object'].T
		transforms = data['transforms'][0]
		print(sup_ply.shape, transforms.shape)
		obj_ply = apply_transform(transforms, obj_ply)
		visualize(sup_ply, obj_ply)

		if i > 100:
			assert 0




