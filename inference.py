import argparse
import os
import os.path as osp
import time

import numpy as np
import open3d as o3d
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
from tqdm import tqdm

from ES import Searcher, Critic, Actor, apply_transform, matrix2vectors
from pose_check.models.uninet_mt import UniNet_MT_V2
from pose_generation.models.vnet import VNet

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Test UCSNet.')

parser.add_argument('--root_path', type=str, help='path to root directory.')
parser.add_argument('--test_list', type=str, default='./pose_generation/dataset/test_list.txt')
parser.add_argument('--save_path', type=str, help='path to save depth maps.')
parser.add_argument('--real_data', action='store_true')
parser.add_argument('--render_ply', action='store_true')
parser.add_argument('--filter', action='store_true')

#test parameters
parser.add_argument('--generator_ckpt', type=str, help='the path for pre-trained model.',
                    default='./checkpoints/')
parser.add_argument('--stable_critic_ckpt', type=str,
                    default='./checkpoints/')
parser.add_argument('--pose_num', type=int, default=16)
parser.add_argument('--rot_rp', type=str, default='6d')
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--num_iter', type=int, default=2)

parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

def read_ply(path, pc_len):
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

def load_mv_ply(path, num_v=2, pc_len=1024):
	assert num_v <= 4
	pcs = []
	for i in range(num_v):
		dir_name = osp.dirname(path)
		base_name = osp.basename(path).split('.')[0]+'.v{:04d}.ply'.format(i)
		path_i = osp.join(dir_name, base_name)
		pcs.append(read_ply(path_i, pc_len))

	point_cloud = np.concatenate(pcs, 0)
	if len(point_cloud) < pc_len:
		ind = np.random.choice(len(point_cloud), pc_len-len(point_cloud))
		point_cloud = np.concatenate([point_cloud, point_cloud[ind]], 0)
	elif len(point_cloud) > pc_len:
		ind = np.random.choice(len(point_cloud), pc_len)
		point_cloud = point_cloud[ind]
	return point_cloud

def write_ply(points, colors, save_path):
	if colors.max() > 1:
		div_ = 255.
	else:
		div_ = 1.

	dir_name = osp.dirname(save_path)
	os.makedirs(dir_name, exist_ok=True)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors / div_)
	o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)


def load_data(data_root, data_list, pc_len=1024, is_real=True):
	for subject_names in data_list:
		subjects = []
		for name in subject_names:
			sub_path = '{}/{}.ply'.format(data_root, name)
			if is_real:
				subject_ply = read_ply(sub_path, pc_len=pc_len)
			else:
				subject_ply = load_mv_ply(sub_path, pc_len=pc_len, num_v=2)
			print('pc shape: ', subject_ply.shape)
			subject_tensor = torch.from_numpy(subject_ply).float().to(args.device) # (N, 3)
			subjects.append(subject_tensor)
		yield subjects

def main(args):
	# build model
	# support = 0, object = 1, mask shape (B, 4, N)
	stable_critic = UniNet_MT_V2(mask_channel=True, only_test=True)

	generator = VNet(mask_channel=False, rot_rep=args.rot_rp,
	                 z_dim=args.z_dim, obj_feat=128, sup_feat=128, z_feat=64,
	                 only_test=True)

	# load checkpoint file specified by args.loadckpt
	print("Loading model {} ...".format(args.generator_ckpt))
	g_state_dict = torch.load(args.generator_ckpt, map_location=torch.device("cpu"))
	generator.load_state_dict(g_state_dict['model'], strict=True)
	print('Success!')

	print("Loading model {} ...".format(args.stable_critic_ckpt))
	s_state_dict = torch.load(args.stable_critic_ckpt, map_location=torch.device("cpu"))
	stable_critic.load_state_dict(s_state_dict['model'], strict=True)
	print('Success!')

	generator = nn.DataParallel(generator)
	generator.to(args.device)
	generator.eval()

	stable_critic = nn.DataParallel(stable_critic)
	stable_critic.to(args.device)
	stable_critic.eval()

	critic = Critic(stable_critic, device=args.device, mini_batch=64, use_filter=args.filter)
	actor = Actor(generator, device=args.device, z_dim=args.z_dim, batch_size=1)
	data_list = open(args.test_list, 'r').readlines()
	data_list = list(map(lambda x: str(x).strip().split('-'), data_list))

	data_loader = load_data(args.root_path, data_list,
	                        pc_len=1024, is_real=args.real_data)

	for j, candidates in enumerate(tqdm(data_loader)):
		pair_id = '-'.join(data_list[j])
		print('Processing {} ...'.format(pair_id))

		solutions = search_solution(candidates=candidates,
		                            actor=actor, critic=critic,
		                            centralize=True,
		                            num_iter=args.num_iter,
		                            n_samp=args.pose_num,
		                            )
		print('Total solutions: ', len(solutions))

		save_predictions(candidates, solutions, pair_id, render_ply=args.render_ply)

		del candidates
		del solutions

		torch.cuda.empty_cache()

def post_refine(support_ply, object_ply, init_transform, critic, num_iter=2):
	'''

	:param support_ply: (N, 3)
	:param object_ply: (N, 3)
	:param init_transform: (B, 4, 4)
	:param critic:
	:return:
	'''
	if num_iter == 0:
		init_transform_6d = matrix2vectors(init_transform)
		scores = critic(tr6d=init_transform_6d,
		                support_ply=support_ply,
		                object_ply=object_ply)
		return init_transform, scores

	cem_searcher = Searcher(action_dim=6, pop_size=4, parents=2, sigma_init=1e-4,
	                        clip=0.003, damp=0.001, damp_limit=0.00001, device=init_transform.device)
	refined_transforms, scores = cem_searcher.search(action_init=init_transform,
	                                           support_ply=support_ply,
	                                           object_ply=object_ply,
	                                           critic=critic,
	                                           n_iter=num_iter,
	                                          )
	return refined_transforms, scores

def search_solution(candidates, actor, critic, centralize, num_iter=2, n_samp=64,):
	solutions = []

	def dfs(support, layer_id, actions=[]):
		if layer_id >= len(candidates):
			return
		selected = candidates[layer_id] # (N, 3)

		tic = time.time()

		if centralize:
			assert support.shape[1] == 3
			assert len(support.shape) == 2
			assert selected.shape[1] == 3
			assert len(selected.shape) == 2

			sup_cent = torch.zeros((1, 3), device=support.device,
								   dtype=support.dtype)
			sup_cent[0, :2] = torch.mean(support, 0, keepdim=True)[0, :2]
			sup_cent[0, 2] = torch.min(support, 0, keepdim=True)[0][0, 2]

			obj_cent = torch.zeros((1, 3), device=selected.device,
									   dtype=selected.dtype)
			obj_cent[0, :2] = torch.mean(selected, 0, keepdim=True)[0, :2]
			obj_cent[0, 2] = torch.min(selected, 0, keepdim=True)[0][0, 2]

			support -= sup_cent
			selected -= obj_cent

			# write_ply(support, np.zeros_like(support), './debug_support.ply')
			# write_ply(selected, np.zeros_like(selected), './debug_object.ply')


		proposals = actor(support, selected, n_samp=n_samp) # (M, 4, 4)
		print('# Time [actor]: {:.2f}'.format(time.time() - tic))

		tic = time.time()
		proposals, scores = post_refine(support, selected, proposals, critic,
		                                num_iter=num_iter) # (M, 4, 4), (M, )
		print('# Time [post refine]: {:.2f}'.format(time.time() - tic))


		if centralize:
			support += sup_cent
			selected += obj_cent
			base2cent = torch.eye(4, dtype=proposals.dtype, device=proposals.device).view((1, 4, 4))
			base2cent[0, :3, 3] = -obj_cent[0, :3]
			cent2base = torch.eye(4, dtype=proposals.dtype, device=proposals.device).view((1, 4, 4))
			cent2base[0, :3, 3] = sup_cent[0, :3]
			proposals = cent2base @ (proposals @ base2cent)

		print('layer {} scores: '.format(layer_id), scores)

		# proposals = proposals[scores >= 0.5]
		# scores = scores[scores >= 0.5]
		print('search layer {}, keep nodes: '.format(layer_id), proposals.shape, scores.shape)

		for action_i, score_i in zip(proposals, scores):
			actions.append((action_i.detach(), score_i.detach()))
			if layer_id == len(candidates)-1:
				# collect action seq
				solutions.append(actions.copy())
			else:
				selected_t = apply_transform(action_i, selected) # (N, 3)
				next_support = torch.cat([support, selected_t], ) # (2*N, 3)
				dfs(next_support, layer_id+1, actions)
			actions.pop()

	with torch.no_grad():
		# [s, o_i, ...]
		dfs(candidates[0], 1, [])
	return solutions

def save_predictions(candidations, solutions, pair_id, render_ply):
	save_dir = osp.join(args.save_path, pair_id)
	os.makedirs(save_dir, exist_ok=True)
	for ind, solution in enumerate(solutions):
		save_one_pair(candidations, solution, save_dir, ind, render_ply=render_ply)

def save_one_pair(point_clouds, solution, save_dir, index, render_ply=True):
	t2n = lambda x: x.detach().cpu().numpy()
	colors = [[20, 20, 160], [20, 160, 200]]

	scores = ['{:.2f}'.format((np.round(x[1].item(), 2))) for x in solution]
	transforms = [x[0] for x in solution]

	file_name = '_'.join(scores) + '_{:04d}.ply'.format(index)

	transforms_np = list(map(t2n, transforms))
	mat_name = file_name.replace('.ply', '.npy')
	np.save(osp.join(save_dir, mat_name), transforms_np)

	if not render_ply:
		return

	assert len(transforms) + 1 == len(point_clouds)
	assert len(point_clouds) == len(colors)

	ret = [point_clouds[0], ]

	for i in range(len(point_clouds)-1):
		subject_i = apply_transform(transforms[i], point_clouds[i+1])
		ret.append(subject_i)

	ply_buffers = []
	for i in range(len(ret)):
		points = t2n(ret[i])
		color = np.ones((len(points), 1)) @ np.array(colors[i]).reshape((1, 3))
		subject_ply = np.concatenate([points, color], 1) # (N, 6)
		ply_buffers.append(subject_ply)
	full_ply = np.concatenate(ply_buffers, 0)

	write_ply(full_ply[:, :3], full_ply[:, 3:], osp.join(save_dir, file_name))



if __name__ == '__main__':
	with torch.no_grad():
		main(args)