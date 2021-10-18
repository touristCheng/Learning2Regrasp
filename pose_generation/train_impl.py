import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import os
import numpy as np
import open3d as o3d
from tensorboardX import SummaryWriter

from data_loader.stable_pose_dataset import StablePoseDataset
import argparse, os, sys, time, datetime
import os.path as osp
from utils.utils import get_linear_schedule_with_warmup, \
	get_step_schedule_with_warmup, dict2cuda, add_summary, \
	DictAverageMeter

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Deep stereo using adaptive cost volume.')
parser.add_argument('--root_path', type=str, help='path to root directory.')
parser.add_argument('--list_path', type=str, help='train scene list.', default='')
parser.add_argument('--save_path', type=str, help='path to save checkpoints.')
parser.add_argument('--restore_path', type=str, default='')
parser.add_argument('--net_arch', type=str, default='vnet2')

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_idx', type=str, default="50,100,160:0.5")
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--pose_num', type=int, default=128)
parser.add_argument('--rot_rp', type=str, default='6d')

parser.add_argument('--log_freq', type=int, default=1, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=2000, help='save checkpoint frequency.')

parser.add_argument('--sync_bn', action='store_true',help='Sync BN.')
parser.add_argument('--opt_level', type=str, default="O0")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--distributed', action='store_true')

args = parser.parse_args()

# num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = args.distributed

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.device("cuda")

if args.sync_bn:
	import apex
	import apex.amp as amp

def print_func(data: dict, prefix: str= ''):
	for k, v in data.items():
		if isinstance(v, dict):
			print_func(v, prefix + '.' + k)
		elif isinstance(v, list):
			print(prefix+'.'+k, v)
		else:
			print(prefix+'.'+k, v.shape)

def sample_from_gaussian(d, batch_size, num_samples):
	m = MultivariateNormal(torch.zeros(d), torch.eye(d))
	z_noise = m.sample((batch_size, num_samples))
	z_noise = z_noise.permute(0, 2, 1) # (B, 3, M)
	return z_noise

def add_point_cloud(pred_pc, gt_pc, sup_pc, logger, step, flag):
	t2n = lambda x: x.detach().cpu().numpy()
	M = pred_pc.shape[1]
	N1 = pred_pc.shape[2]
	pred_pc = t2n(pred_pc[0])
	gt_pc = t2n(gt_pc[0])

	sup_pc:np.array = t2n(sup_pc[0]).T # (N, 3)
	sup_pc = np.repeat(np.expand_dims(sup_pc, axis=0), repeats=M, axis=0)
	N2 = sup_pc.shape[1]

	pred_tags = ['pred_sample_{}'.format(x) for x in range(M)]
	gt_tags = ['gt_sample_{}'.format(x) for x in range(M)]

	pred_pc = np.concatenate([pred_pc, sup_pc], axis=1)
	gt_pc = np.concatenate([gt_pc, sup_pc], axis=1)

	gt_colors = np.zeros((M, N1+N2, 3))
	gt_colors[:, :N1, 1] = 255
	gt_colors[:, N1:, 0] = 100
	gt_colors[:, N1:, 1] = 100
	gt_colors[:, N1:, 2] = 200

	pred_colors = np.zeros((M, N1+N2, 3))
	pred_colors[:, :N1, 1] = 80
	pred_colors[:, N1:, 0] = 100
	pred_colors[:, N1:, 1] = 100
	pred_colors[:, N1:, 2] = 200

	add_summary([{'type': 'points', 'tags': pred_tags,
	              'vals': [pred_pc, pred_colors]},
	             {'type': 'points', 'tags': gt_tags,
	              'vals': [gt_pc, gt_colors]}],
	            logger=logger, step=step, flag=flag, max_disp=2)

def main(args, model, optimizer, scheduler, train_loader, train_sampler, start_step=0):

	train_step = start_step
	start_ep = start_step // len(train_loader)

	model.train()
	for ep in range(start_ep, args.epochs):
		np.random.seed()
		train_scores = DictAverageMeter()
		if train_sampler is not None:
			train_sampler.set_epoch(ep)

		for batch_idx, sample in enumerate(train_loader):
			tic = time.time()
			sample['z_noise'] = sample_from_gaussian(args.z_dim,
			                                         args.batch_size,
			                                         args.pose_num)
			sample_cuda = dict2cuda(sample)

			# print_func(sample_cuda)
			optimizer.zero_grad()
			ret = model(sample_cuda)
			loss = ret['loss']

			# print_func(outputs)
			if is_distributed and args.sync_bn:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			optimizer.step()
			scheduler.step()

			train_scores.update({'loss': loss.item()})

			train_step += 1
			if train_step % args.log_freq == 0:
				avg_stat = train_scores.mean()
				print("[Rank: {}] time={:.2f} Epoch {}/{}, Iter {}/{}, lr {:.6f}, stats: {}".format(
					args.local_rank, time.time() - tic,
					                 ep+1, args.epochs, batch_idx+1, len(train_loader),
					optimizer.param_groups[0]["lr"],
					avg_stat))
				if on_main:
					add_point_cloud(ret['pred_pc'], ret['selt_pc'], sample['support'],
					                logger=logger, step=train_step, flag='train')
					add_summary([{'type': 'scalars', 'tags': list(avg_stat.keys()),
					              'vals': list(avg_stat.values())}],
					              logger=logger, step=train_step, flag='train')

			if on_main and train_step % args.save_freq == 0:
				torch.save({"step": train_step,
				            "model": model.module.state_dict(),
				            "optimizer": optimizer.state_dict(),
				            "scheduler": scheduler.state_dict(),
				            },
				           "{}/model_{:08d}.ckpt".format(args.save_path, train_step))

def distribute_model(args):
	def sync():
		if not dist.is_available():
			return
		if not dist.is_initialized():
			return
		if dist.get_world_size() == 1:
			return
		dist.barrier()
	if is_distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(
			backend="nccl", init_method="env://"
		)
		sync()

	start_step = 0

	if args.net_arch == 'vnet2':
		from models.vnet import VNet
		print('use vnet2!')
	else:
		raise NotImplementedError

	model: torch.nn.Module = VNet(mask_channel=False, rot_rep=args.rot_rp,
	                              z_dim=args.z_dim, obj_feat=128, sup_feat=128, z_feat=64)
	if args.restore_path:
		checkpoint = torch.load(args.restore_path, map_location=torch.device("cpu"))
		model.load_state_dict(checkpoint['model'], strict=True)

	model.to(device)

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
	                       weight_decay=args.wd)

	train_set = StablePoseDataset(root_dir=args.root_path, list_path=args.list_path,
	                              pose_num=args.pose_num, pc_len=1024, use_aug=True)

	if is_distributed:
		if args.sync_bn:
			model = apex.parallel.convert_syncbn_model(model)
			model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, )
			print('Convert BN to Sync_BN successful.')

		model = torch.nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank,)

		train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
		                                                    rank=dist.get_rank())

	else:
		model = nn.DataParallel(model)
		train_sampler = None

	def worker_init_fn(worker_id):
		np.random.seed(np.random.get_state()[1][0] + worker_id)

	train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler,
	                          num_workers=args.num_workers, pin_memory=True,
	                          drop_last=True, shuffle=not is_distributed, worker_init_fn=worker_init_fn)

	milestones = list(map(float, args.lr_idx.split(':')[0].split(',')))
	assert np.max(milestones) <= 1.0, milestones
	milestones = list(map(lambda x: int(float(x) * float(len(train_loader) * args.epochs)), milestones))
	gamma = float(args.lr_idx.split(':')[1])
	warpup_iters = min(500, int(0.05*len(train_loader)))

	scheduler = get_step_schedule_with_warmup(optimizer=optimizer, milestones=milestones,
	                                          gamma=gamma, warmup_iters=warpup_iters)

	if args.restore_path:
		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])
		start_step = checkpoint['step']
		print("Restoring checkpoint {} ...".format(args.restore_path))

	return model, optimizer, scheduler, train_loader, train_sampler, start_step


if __name__ == '__main__':
	model, optimizer, scheduler, train_loader, train_sampler, start_step = distribute_model(args)
	on_main = (not is_distributed) or (dist.get_rank() == 0)
	if on_main:
		os.makedirs(args.save_path, exist_ok=True)
		logger = SummaryWriter(args.save_path)
		print(args)

	main(args=args, model=model, optimizer=optimizer, scheduler=scheduler,
	     train_loader=train_loader, train_sampler=train_sampler, start_step=start_step)
