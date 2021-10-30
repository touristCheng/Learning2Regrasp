import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from .pointnet_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class UniNet_MT_V2(nn.Module):
	def __init__(self, normal_channel=False, mask_channel=False, bootle_neck=256, only_test=False):
		super(UniNet_MT_V2, self).__init__()

		self.normal_channel = normal_channel
		self.mask_channel = mask_channel
		self.only_test = only_test

		additional_channel = 0
		if mask_channel:
			additional_channel += 1
		if normal_channel:
			additional_channel += 3

		self.sa1 = PointNetSetAbstractionMsg(512, [0.05, 0.1], [16, 32], additional_channel, [[8, 8, 16], [16, 16, 32]])
		self.sa2 = PointNetSetAbstractionMsg(128, [0.1, 0.2], [16, 32], 16+32, [[32, 32, 64], [32, 48, 64]])
		self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 64+64, [[64, 96, 128], [64, 96, 128]])
		self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 128+128, [[128, 128, 256], [128, 172, bootle_neck]])

		self.sa5 = PointNetSetAbstraction(None, None, None, bootle_neck+256 + 3, [256, 128], True)

		self.fp4 = PointNetFeaturePropagation(bootle_neck+256+128+128, [512, 256])
		self.fp3 = PointNetFeaturePropagation(64+64+256, [256, 128])
		self.fp2 = PointNetFeaturePropagation(16+32+128, [128, 96])
		self.fp1 = PointNetFeaturePropagation(96, [64, 64])

		self.contact_cls = nn.Sequential(*[
			nn.Linear(128, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.2), # drop 30% during training
			nn.Linear(64, 32),
			nn.Linear(32, 2)
		])

		self.stable_cls = nn.Sequential(*[
			nn.Linear(128, 64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.2), # drop 30% during training
			nn.Linear(64, 32),
			nn.Linear(32, 2)
		])

		self.offset_reg = nn.Sequential(*[nn.Conv1d(64, 32, kernel_size=(1, )),
		                                  nn.BatchNorm1d(32),
		                                  nn.ReLU(),
		                                  nn.Conv1d(32, 16, kernel_size=(1, )),
		                                  nn.Conv1d(16, 3, kernel_size=(1, ))
		                                  ])

	def forward(self, sample):
		xyz = sample['data']

		B, C, N = xyz.shape
		l0_xyz = xyz[:, :3, :]
		if self.normal_channel or self.mask_channel:
			extr_channel = xyz[:, 3:, :]
		else:
			extr_channel = None

		l1_xyz, l1_points = self.sa1(l0_xyz, extr_channel)
		l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
		l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
		l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

		l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)

		l5_points_flatten = l5_points.view(B, -1)

		l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
		l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
		l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
		l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

		stable_pred = self.stable_cls(l5_points_flatten)
		contact_pred = self.contact_cls(l5_points_flatten)
		offset_pred = self.offset_reg(l0_points)

		if self.only_test:
			return {'preds': [stable_pred, contact_pred, offset_pred]}
		else:
			loss_items = {'stable': [stable_pred, sample['stable'], 1.],
			              'contact': [contact_pred, sample['contact'], 1.],
			              'offset': [offset_pred, sample, 10., 2.]
			              }
			ret = self.calc_loss(loss_items)
			ret['preds'] = [stable_pred, contact_pred, offset_pred]
			return ret

	def calc_loss(self, loss_items: dict):
		stable_loss = F.cross_entropy(loss_items['stable'][0],
		                              loss_items['stable'][1],
		                              ignore_index=255)
		contact_loss = F.cross_entropy(loss_items['contact'][0],
		                               loss_items['contact'][1],
		                               ignore_index=255)


		gt_offset, ind = self.compute_vector_field(loss_items['offset'][1]) # (B, 3, N), (B, N)
		offset_loss = F.smooth_l1_loss(gt_offset, loss_items['offset'][0], reduce=False) # (B, N)

		end_points = loss_items['offset'][1]['data'][:, :3, :] + loss_items['offset'][0]

		var_loss = self.calc_variance(end_points=end_points,
		                              index=ind,
		                              num_contacts=loss_items['offset'][1]['total_contacts'])

		offset_loss = offset_loss[loss_items['contact'][1] == 1]
		offset_loss = torch.mean(offset_loss)
		offset_loss = torch.nan_to_num(offset_loss)

		total_loss = stable_loss * loss_items['stable'][2] + \
		             contact_loss * loss_items['contact'][2] + \
		             offset_loss * loss_items['offset'][2] + \
					 var_loss * loss_items['offset'][3]

		return {'loss': total_loss, 'loss_items': [stable_loss, contact_loss, offset_loss, var_loss]}

	def compute_vector_field(self, sample):
		'''

		:param points: (B, N, 3)
		:param contacts: (B, M, 3)
		:return:
		'''
		points = sample['data'].permute(0, 2, 1)[..., :3] # (B, N, 3)
		contacts = sample['contact_points'] # (B, 30, 3)
		num_contacts = sample['total_contacts'] # (B, )
		B = points.shape[0]
		N = points.shape[1]
		inds = []
		vects = []
		with torch.no_grad():
			for i in range(B):
				cnt = int(num_contacts[i])
				if cnt > 0:
					vec_field = contacts[i:i+1, :cnt] - points[i].unsqueeze(1) # (N, M, 3)
					nm_dist = torch.sum(vec_field**2, dim=2, keepdim=False) # (N, M)
					ind1 = torch.argmin(nm_dist, dim=1, keepdim=False) # (N, )
					ind0 = torch.arange(nm_dist.shape[0]) # (N, )
					select_vect = vec_field[ind0, ind1].permute(1, 0) # (3, N)
				else:
					select_vect = torch.zeros((3, N), device=points.device)
					ind1 = torch.zeros((N, ), device=points.device, dtype=torch.long)

				vects.append(select_vect)
				inds.append(ind1)

		return torch.stack(vects, 0), torch.stack(inds, 0)

	def calc_variance(self, end_points, index, num_contacts):
		B = num_contacts.shape[0]
		var_loss = torch.zeros((), device=end_points.device, dtype=torch.float)
		end_points = end_points.permute(0, 2, 1) # (B, N, 3)
		cnt = 0
		for i in range(B):
			if int(num_contacts[i]) == 0:
				continue
			cnt += 1
			endp_i = end_points[i] # (N, 3)
			cont_id = torch.unique(index[i])
			for j in cont_id:
				jth_ind = index[i] == int(j)
				mean_j = torch.mean(endp_i[jth_ind], dim=0, keepdim=True) # (1, 3)
				var_j = torch.sum((endp_i[jth_ind] - mean_j) ** 2, dim=1, keepdim=False) # (n, )
				var_loss = var_loss + torch.mean(var_j)

		var_loss =var_loss / float(cnt+1e-16)
		return var_loss


