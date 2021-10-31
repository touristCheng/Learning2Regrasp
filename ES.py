import torch
import numpy as np
import torch.nn as nn
import pytorch3d.transforms as torch_transform
from pose_check.utils.utils import dict2cuda
from torch.distributions import MultivariateNormal
from sklearn.cluster import AgglomerativeClustering

class CEM():

	"""
	Cross-entropy methods. Adapted to PyTorch
	"""

	def __init__(self,
	             num_params,
	             batch_size,
	             pop_size,
	             parents,
				 mu_init=None,
				 sigma_init=1e-3,
				 clip=0.1,
				 damp=0.1,
				 damp_limit=1e-5,
				 elitism=True,
				 device=torch.device('cuda')
				):

		# misc
		self.num_params = num_params
		self.batch_size = batch_size
		self.device = device
		# distribution parameters
		if mu_init is None:
			self.mu = torch.zeros([self.batch_size, self.num_params], device=device)
		else:
			self.mu = mu_init.clone()
		self.sigma = sigma_init
		self.damp = damp
		self.damp_limit = damp_limit
		self.tau = 0.95
		self.cov = self.sigma * torch.ones([self.batch_size, self.num_params], device=device)
		self.clip = clip 
		
		# elite stuff
		self.elitism = elitism
		self.elite = torch.sqrt(torch.tensor(self.sigma, device=device)) * torch.rand(self.batch_size, self.num_params, device=device)
		self.elite_score = None
		
		# sampling stuff
		self.pop_size = pop_size
		if parents is None or parents <= 0:
			self.parents = pop_size // 2
		else:
			self.parents = parents
		self.weights = torch.FloatTensor([np.log((self.parents + 1) / i)
								 for i in range(1, self.parents + 1)]).to(device)
		self.weights /= self.weights.sum()

	def ask(self, pop_size):
		"""
		Returns a list of candidates parameters
		"""
		epsilon = torch.randn(self.batch_size, pop_size, self.num_params, device=self.device)
		inds = self.mu.unsqueeze(1) + (epsilon * torch.sqrt(self.cov).unsqueeze(1)).clamp(-self.clip, self.clip)
		if self.elitism:
			inds[:, -1] = self.elite
		return inds

	def tell(self, solutions, scores):
		"""
		Updates the distribution
		returns the best solution
		:param solutions: (B, N, 6) 6d representation of transforms
		:param scores: (B, N)
		:return top_solution: (B, 6)
		"""
		assert len(scores.shape) == 2

		sorted_scores, idx_sorted = torch.sort(scores, dim=1, descending=True)

		old_mu = self.mu.clone()
		self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
		idx_sorted = idx_sorted[:, :self.parents]
		top_solutions = torch.gather(solutions, 1, idx_sorted.unsqueeze(2).expand(*idx_sorted.shape, solutions.shape[-1]))
		self.mu = self.weights @ top_solutions
		z = top_solutions - old_mu.unsqueeze(1)
		self.cov = 1 / self.parents * self.weights @ (
			z * z) + self.damp * torch.ones([self.batch_size, self.num_params], device=self.device)

		self.elite = top_solutions[:, 0, :]
		# self.elite_score = scores[:, idx_sorted[0]]

		return top_solutions[:, 0, :], sorted_scores[:, 0]

	def get_distrib_params(self):
		"""
		Returns the parameters of the distrubtion:
		the mean and sigma
		"""
		return self.mu.clone(), self.cov.clone()

class Searcher():
	def __init__(self,
				 action_dim,
				 pop_size=25,
				 parents=5,
				 sigma_init=1e-3,
				 clip=0.1,
				 damp=0.1,
				 damp_limit=0.05,
				 device=torch.device('cuda')):

		self.sigma_init = sigma_init
		self.clip=clip
		self.pop_size = pop_size
		self.damp = damp
		self.damp_limit = damp_limit
		self.parents = parents
		self.action_dim = action_dim
		self.device = device

	def search(self, action_init, support_ply, object_ply, critic, n_iter=3):
		'''

		:param action_init: (B, 4, 4)
		:param critic:
		:param n_iter:
		:param action_bound:
		:return:
		'''
		batch_size = action_init.shape[0]
		action_init = matrix2vectors(action_init) # (B, 6)

		cem = CEM(num_params=self.action_dim,
		          batch_size=batch_size,
		          pop_size=self.pop_size,
		          parents=self.parents,
		          mu_init=action_init,
		          sigma_init=self.sigma_init,
		          clip=self.clip,
		          damp=self.damp,
		          damp_limit=self.damp_limit,
		          elitism=True,
		          device=self.device
		          )

		best_actions = None
		best_scores = None

		with torch.no_grad():
			for iter in range(n_iter):
				actions = cem.ask(self.pop_size)
				Qs = critic(tr6d=actions.view(self.pop_size * batch_size, -1),
				            support_ply=support_ply,
				            object_ply=object_ply).view(batch_size, self.pop_size)
				good_actions, good_scores = cem.tell(actions, Qs)

				if best_scores is None:
					best_actions = good_actions
					best_scores = good_scores
				else:
					action_index = (best_scores < good_scores).squeeze()
					best_actions[action_index] = good_actions[action_index]
					print('before assign: ', best_scores)
					print('good scores: ', good_scores)
					best_scores = torch.max(best_scores, good_scores)
					print('after max: ', best_scores)

				if iter == n_iter - 1:
					best_actions = vectors2matrix(best_actions) # (B, 4, 4)
					return best_actions, best_scores

def calc_min_dist(p_a, p_b):
	'''

    :param p_a: (n, 3)
    :param p_b: (m, 3)
    :return:
    '''
	aa = np.sum(p_a ** 2, axis=1, keepdims=False)
	bb = np.sum(p_b ** 2, axis=1, keepdims=False)
	n = p_a.shape[0]
	m = p_b.shape[0]

	a_ = np.reshape(p_a, (n, 1, 1, 3))
	b_ = np.reshape(p_b, (1, m, 3, 1))
	ab_ = np.matmul(a_, b_)[..., 0, 0] # (n, m)
	aa_ = np.repeat(np.reshape(aa, (n, 1)), axis=1, repeats=m)
	bb_ = np.repeat(np.reshape(bb, (1, m)), axis=0, repeats=n)
	dist = np.sqrt(aa_+bb_-2*ab_)
	return dist

def heuristic_filter(points_a, points_b, thresh=0.018, d_th=0.65):
	dist = calc_min_dist(points_a, points_b)
	dist_a = np.min(dist, axis=1, keepdims=False)
	nearby_points = points_a[dist_a < thresh]
	if len(nearby_points) < 10:
		return False

	clustering = AgglomerativeClustering(n_clusters=None,
										 distance_threshold=d_th).fit(nearby_points)
	labels = clustering.labels_
	label_ids = np.unique(labels)

	if len(label_ids) < 2:
		return False
	for id_ in label_ids:
		if np.sum(labels == id_) < 5:
			return False
	return True

def do_filtering(support_ply, object_ply, ):
	scores = []
	sup_np = support_ply.detach().cpu().numpy()
	obj_np = object_ply.detach().cpu().numpy()
	B = sup_np.shape[0]

	for i in range(B):
		f = heuristic_filter(sup_np[i], obj_np[i])
		scores.append(float(f))
	scores = torch.from_numpy(np.array(scores)).to(support_ply.device)
	return scores


class Critic(object):
	def __init__(self, model: nn.Module, device, mini_batch=4, use_filter=False):
		self.model = model.to(device)
		self.model.eval()
		self.mini_batch = mini_batch
		self.use_filter = use_filter
		self.device = device

	def __call__(self, tr6d, support_ply, object_ply):
		'''

		:param pose_6d: (B*pop_size, 6)
		:param support: (N, 3)
		:param object: (N, 3)
		:return: scores: (B*pop_size)
		'''

		transform = vectors2matrix(tr6d) # (M, 4, 4)
		object_ply = apply_multi_transforms(transform, object_ply) # (B, N, 3)
		B = object_ply.shape[0]
		N1 = object_ply.shape[1]
		support_ply = support_ply.unsqueeze(0).repeat(B, 1, 1) # (B, N, 3)
		N2 = support_ply.shape[1]

		data = torch.cat([support_ply, object_ply], 1).permute(0, 2, 1) # (B, 3, 2*N)

		mask = torch.zeros((B, 1, N1+N2), device=data.device, dtype=data.dtype)
		mask[:, :, N2:] = 1

		data = torch.cat([data, mask], 1)
		sample = {'data': data}
		if 'cuda' in self.device:
			sample = dict2cuda(sample)

		ret = infer_mini_batch(self.model, sample, self.mini_batch)
		probs = torch.softmax(ret['preds'][0], 1)[:, 1] # (M, )

		if self.use_filter:
			scores = do_filtering(support_ply, object_ply)
			probs *= scores

		return probs

def sample_from_gaussian(d, batch_size, num_samples):
	m = MultivariateNormal(torch.zeros(d), torch.eye(d))
	z_noise = m.sample((batch_size, num_samples))
	z_noise = z_noise.permute(0, 2, 1) # (B, 3, M)
	return z_noise

def write_ply(points, colors, save_path):
	import os
	import os.path as osp
	import open3d as o3d

	if colors.max() > 1:
		div_ = 255.
	else:
		div_ = 1.
	os.makedirs(osp.dirname(save_path), exist_ok=True)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(points)
	pcd.colors = o3d.utility.Vector3dVector(colors / div_)
	o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)


class Actor(object):
	def __init__(self, model: nn.Module, device, z_dim, batch_size=1):
		self.model = model.to(device)
		self.z_dim = z_dim
		self.batch_size = batch_size
		assert batch_size == 1
		self.model.eval()
		self.device = device

	def __call__(self, support_ply, object_ply, n_samp=128):
		'''

		:param support: (N, 3)
		:param object: (N, 3)
		:return: scores: (M, 4, 4)
		'''
		support_ply = support_ply.unsqueeze(0).permute(0, 2, 1) # (1, 3, N)
		object_ply = object_ply.unsqueeze(0).permute(0, 2, 1) # (1, 3, N)


		sample = {'support': support_ply, 'object': object_ply}
		z_noise = sample_from_gaussian(self.z_dim,
		                               self.batch_size,
		                               n_samp)
		z_noise[:, :, 0] = 0
		sample['z_noise'] = z_noise
		if 'cuda' in self.device:
			sample = dict2cuda(sample)

		pred = self.model(sample)['pred'] # (1, M, 4, 4)

		return pred[0]

def matrix2vectors(matrix):
	'''

	:param matrix: (B, 4, 4)
	:return: (B, 6)
	'''
	assert matrix.shape[1:] == (4, 4)

	rot_vec = torch_transform.matrix_to_euler_angles(matrix[:, :3, :3], 'XYZ') # (B, 3)
	trs_vec = matrix[:, :3, 3] # (B, 3)
	return torch.cat([trs_vec, rot_vec], 1) # (B, 6)

def vectors2matrix(vec6d):
	'''

	:param vec6d: (B, 6)
	:return: (B, 4, 4)
	'''
	assert vec6d.shape[1: ] == (6, )
	B = vec6d.shape[0]
	rot = torch_transform.euler_angles_to_matrix(vec6d[:, 3:], 'XYZ') # (B, 3, 3)
	trs = vec6d[:, :3].unsqueeze(2) # (B, 3, 1)
	transform = torch.cat([rot, trs], dim=2) # (B, 3, 4)
	ones = torch.tensor([0, 0, 0, 1],
	                    device=transform.device).view(1, 1, 4)
	ones = ones.repeat((B, 1, 1))
	transform = torch.cat([transform, ones], dim=1) # (B, 4, 4)
	return transform

def apply_transform(transform, points):
	'''

	:param transform: (4, 4)
	:param points: (N, 3)
	:return:
	'''
	N = points.shape[0]
	ones = torch.ones((N, 1), device=points.device, dtype=points.dtype)
	points = torch.cat([points, ones], 1).unsqueeze(2) # (N, 4, 1)
	points_t = torch.matmul(transform.unsqueeze(0), points) # (N, 4, 1)
	points_t = points_t[..., :3, 0] # (N, 3)
	return points_t

def apply_multi_transforms(transforms, points):
	'''

	:param transforms:
	:param points:
	:return:
	'''
	ret = []
	for t in transforms:
		ret.append(apply_transform(t, points))
	return torch.stack(ret, 0)

def infer_mini_batch(model, data:dict, batch_size=16):
	B = data['data'].shape[0]
	assert B % batch_size == 0
	N = B // batch_size
	rets = {}
	for i in range(N):
		batch_data = {}
		for k, v in data.items():
			batch_data[k] = v[i*batch_size:(i+1)*batch_size]
		ret_i:dict = model(batch_data)
		for k, v in ret_i.items():
			rets[k] = rets.get(k, []) + [v]
	for k, v in rets.items():
		if isinstance(v[0], torch.Tensor):
			rets[k] = torch.cat(v, dim=0)
		elif isinstance(v[0], list):
			cat_num = len(v[0])
			ret_k = []
			for i in range(cat_num):
				cat_i = torch.cat([x[i] for x in v], dim=0)
				ret_k.append(cat_i)
			rets[k] = ret_k
		else:
			raise
	return rets
