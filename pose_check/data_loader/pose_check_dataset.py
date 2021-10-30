import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import os.path as osp
import numpy as np
import json
import os
from collections import OrderedDict
import open3d as o3d

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

def load_collision_data(root_dir, list_path, samples=None):
	data_list = open(list_path, 'r').readlines()
	data_list = list(map(str.strip, data_list))
	all_odgts = []
	sim_odgts = []
	for line in data_list:
		one_odgt = read_file(osp.join(root_dir, line))
		if line.startswith('simulation'):
			one_odgt = list(filter(lambda x: x['stable'], one_odgt))
			sim_odgts += one_odgt
		else:
			all_odgts += one_odgt

	np.random.shuffle(sim_odgts)
	sim_odgts = sim_odgts[:min(50000, len(sim_odgts))]
	all_odgts += sim_odgts
	np.random.shuffle(all_odgts)
	print('hard samples: ', len(sim_odgts))
	print('total samples: ', len(all_odgts))
	if samples:
		print('Sample ratio: ', samples)
		all_odgts = all_odgts[:samples]

	return all_odgts

def load_multi_task_data(root_dir, list_path):
	data_list = open(list_path, 'r').readlines()
	data_list = list(map(str.strip, data_list))

	random_data = []
	simulation_data = []
	for line in data_list:
		one_pair_data = read_file(osp.join(root_dir, line))
		file_name = osp.basename(line)
		if file_name.startswith('random'):
			random_data += one_pair_data
		else:
			for data in one_pair_data:
				if data['stable'] and len(data['support_contact']) == 0:
					continue
				simulation_data.append(data)

	print('simulation data: ', len(simulation_data))
	print('random data: ', len(random_data))

	all_data = random_data + simulation_data
	np.random.shuffle(all_data)
	return all_data

class PoseCheckDataset(Dataset):
	def __init__(self, root_dir, list_path, samples, label_name='collision', pc_len=1024, use_aug=True):
		super(PoseCheckDataset, self).__init__()
		self.root_dir = root_dir

		odgt_data = load_collision_data(root_dir, list_path, samples=samples)

		self.data_pairs = odgt_data
		print('Total pairs: ', len(self.data_pairs))
		self.pc_len = pc_len
		self.label_name = label_name
		self.use_aug = use_aug

	def __getitem__(self, index):
		items:dict = self.data_pairs[index]

		transform = np.array(items['transform'])
		cls = int(items[self.label_name])

		sup_ply = load_mv_ply(self.root_dir, items['sup_init_path'],
		                      pc_len=self.pc_len)
		obj_ply = load_mv_ply(self.root_dir, items['obj_init_path'],
		                      pc_len=self.pc_len)

		obj_ply = apply_transform(transform, obj_ply) # (N, 3)
		comp_ply = np.concatenate([sup_ply, obj_ply], 0)
		if self.use_aug:
			comp_ply = random_transform_points(comp_ply) # (N, 3)
			comp_ply = random_offset_points(comp_ply)

		mask_ply = np.zeros((len(comp_ply), 1), dtype='float32')
		mask_ply[len(sup_ply):, 0] = 1

		full_ply = np.concatenate([comp_ply, mask_ply], 1)

		full_ply = full_ply.T.astype('float32') # (4, N)

		ret = {'data': full_ply, 'label': cls,
		       'pair_id': items['pair_id'], 'index': items['index']}

		return ret

	def __len__(self):
		return len(self.data_pairs)

class MultiTaskDatasetV2(Dataset):
	def __init__(self, root_dir, list_path, pc_len=1024, max_contact=200, use_aug=True):
		super(MultiTaskDatasetV2, self).__init__()
		self.root_dir = root_dir

		data_pairs = load_multi_task_data(root_dir, list_path)
		self.data_pairs = data_pairs

		print('Total pairs [{}]: '.format(list_path), len(self.data_pairs))
		self.pc_len = pc_len
		self.max_contact = max_contact
		self.use_aug = use_aug

	def __getitem__(self, index):
		items: dict = self.data_pairs[index]

		transform = np.array(items['transform'])

		sup_ply = load_mv_ply(self.root_dir, items['sup_init_path'],
		                      pc_len=self.pc_len)
		obj_ply = load_mv_ply(self.root_dir, items['obj_init_path'],
		                      pc_len=self.pc_len)

		obj_ply = apply_transform(transform, obj_ply) # (N, 3)
		comp_ply = np.concatenate([sup_ply, obj_ply], 0)

		contact_label = int(items['contact'])
		stable_label = int(items['stable'])

		contact_points = np.zeros((self.max_contact, 3), dtype='float32')
		total_contacts = 0

		if contact_label == 1:
			if len(items['support_contact']) == 0:
				contact_label = 255

		if contact_label == 1:
			all_contacts = items['support_contact']
			np.random.shuffle(all_contacts)
			total_contacts = min(len(all_contacts), self.max_contact)
			contact_points[:total_contacts] = np.array(all_contacts, dtype='float32')[:total_contacts]

		if self.use_aug:
			comp_ply, contact_points = random_transform_pair(comp_ply, contact_points) # (N, 3), (M, 3)
			comp_ply, contact_points = random_offset_pair(comp_ply, contact_points)


		mask_ply = np.zeros((len(comp_ply), 1), dtype='float32')
		mask_ply[len(sup_ply):, 0] = 1

		full_ply = np.concatenate([comp_ply, mask_ply], 1)
		full_ply = full_ply.T.astype('float32') # (4, N)

		ret = {'data': full_ply, 'contact_points': contact_points.astype('float32'),
		       'total_contacts': total_contacts,
		       'stable': stable_label, 'contact': contact_label,
		       'pair_id': items['pair_id'], 'index': items['index']}
		# (4, N)

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

def random_transform_pair(points, contacts):
	'''

	:param points: (N, 3)
	:param contacts: (M, 3)
	:return:
	'''
	deg = float(np.random.uniform(0, 360, size=()))
	r = R.from_euler('z', deg, degrees=True)
	t = np.eye(4)
	t[:3, :3] = r.as_matrix()

	points = apply_transform(t, points)
	contacts = apply_transform(t, contacts)
	return points, contacts

def random_transform_points(points):
	'''

	:param points: (N, 3)
	:param contacts: (M, 3)
	:return:
	'''

	deg = float(np.random.uniform(0, 360, size=()))
	r = R.from_euler('z', deg, degrees=True)
	t = np.eye(4)
	t[:3, :3] = r.as_matrix()

	points = apply_transform(t, points)
	return points

def random_offset_pair(points, contacts):
	'''

	:param points:
	:param contacts:
	:return:
	'''
	xyz_range = np.array([[-0.02, -0.02, -0.002],
	                      [0.02, 0.02, 0.002]])
	offset = np.random.uniform(0, 1, size=(1, 3))
	offset = offset * (xyz_range[1:2]-xyz_range[0:1]) + xyz_range[0:1]
	points += offset
	contacts += offset

	n = points.shape[0]
	sigma = 0.003
	noise = np.random.normal(0, sigma, size=(n,))
	points[:, 2] += noise
	return points, contacts

def random_offset_points(points):
	'''

	:param points:
	:param contacts:
	:return:
	'''
	xyz_range = np.array([[-0.02, -0.02, -0.002],
	                      [0.02, 0.02, 0.002]])
	offset = np.random.uniform(0, 1, size=(1, 3))
	offset = offset * (xyz_range[1:2]-xyz_range[0:1]) + xyz_range[0:1]
	points += offset

	n = points.shape[0]
	sigma = 0.003
	noise = np.random.normal(0, sigma, size=(n, 3))
	points += noise
	return points

def debug_visualization(sample):
	'''

	:param transform: (4, 4)
	:param result:
	:param ind:
	:param pair_id:
	:return:
	'''

	ply = sample['data'][0].numpy().T # (N, 4)

	n_ = 3000
	plane = np.zeros((n_, 3))
	plane[:, :2] = np.random.uniform(-0.2, 0.2, size=(n_, 2))
	plane = np.concatenate([plane, np.zeros((n_, 1))], 1)
	ply = np.concatenate([ply, plane], 0)

	points = ply[:, :3]
	mask = ply[:, 3:]

	if sample['stable'][0] == 1:
		obj_color = [0, 120, 0]
	else:
		obj_color = [120, 0, 0]

	color = mask @ np.array(obj_color).reshape((1, 3)) + \
	        (1-mask) @ np.array([0, 0, 200]).reshape((1, 3))

	contacts = []
	for pid in range(int(sample['total_contacts'][0])):
		cp = sample['contact_points'][0, pid].numpy()
		sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
		sphere.translate([cp[0], cp[1], cp[2]])
		sphere.paint_uniform_color([0.9, 0.1, 0.1])
		contacts.append(sphere)

	sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
	sphere.paint_uniform_color([0.5, 0.5, 0.1])
	contacts.append(sphere)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(color/255.)
	o3d.visualization.draw_geometries([pcd] + contacts)

if __name__ == '__main__':
	# datasetV = PoseCheckDataset('../../../debug/dataset/plys', '../../../debug/dataset/debug.txt')
	# loaderV = DataLoader(datasetV, 2, sampler=None, num_workers=1,
	#                      drop_last=False, shuffle=False)


	# train_set = MultiTaskDatasetV2(root_dir='../../dataset',
	#                                list_path='../../dataset/data_list/train_classifier.txt',
	#                                use_aug=True)
	test_set = MultiTaskDatasetV2(root_dir='../../dataset',
								  list_path='../../dataset/data_list/test_classifier.txt',
								  use_aug=False)

	loaderV = DataLoader(test_set, 4, sampler=None, num_workers=2,
	                     drop_last=False, shuffle=False)
	print('all samples: ', len(loaderV))

	for sample in loaderV:
		print(sample['data'].shape)
		print(sample['stable'])
		debug_visualization(sample)
