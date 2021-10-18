import argparse
import gc
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from data_loader.pose_check_dataset import MultiTaskDatasetV2
from models.uninet_mt import UniNet_MT_V2
from utils.utils import get_step_schedule_with_warmup, dict2cuda, add_summary, \
	DictAverageMeter, calc_stat

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Deep stereo using adaptive cost volume.')
parser.add_argument('--root_path', type=str, help='path to root directory.')
parser.add_argument('--train_list', type=str, help='train scene list.', default='')
parser.add_argument('--val_list', type=str, help='val scene list.', default='')
parser.add_argument('--save_path', type=str, help='path to save checkpoints.')
parser.add_argument('--restore_path', type=str, default='')

parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_idx', type=str, default="50,100,160:0.5")
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--batch_size', type=int, default=32)

parser.add_argument('--log_freq', type=int, default=100, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=10000, help='save checkpoint frequency.')
parser.add_argument('--eval_freq', type=int, default=10000, help='evaluate frequency.')

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

def main(args, model, optimizer, scheduler, train_loader, val_loader, train_sampler, start_step=0):

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

			sample_cuda = dict2cuda(sample)

			# print_func(sample_cuda)
			optimizer.zero_grad()
			ret = model(sample_cuda)
			loss = ret['loss'].mean()
			preds = ret['preds']
			loss_items = [l.mean() for l in ret['loss_items']]

			# print_func(outputs)
			if is_distributed and args.sync_bn:
				with amp.scale_loss(loss, optimizer) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()

			optimizer.step()
			scheduler.step()

			if train_step % args.log_freq == 0:
				train_scores.update({'loss': float(loss),
				                     'contact_loss': float(loss_items[0]),
				                     'stable_loss': float(loss_items[1]),
				                     'offset_loss': float(loss_items[2]),
				                     'variance_loss': float(loss_items[3])
				                     })

				calc_stat(sample_cuda, preds[0], train_scores, label_type='stable')
				calc_stat(sample_cuda, preds[1], train_scores, label_type='contact')

				avg_stat = train_scores.mean()
				print("[Rank: {}] time={:.2f} Epoch {}/{}, Iter {}/{}, lr {:.6f}, stats: {}".format(
					args.local_rank, time.time() - tic,
					ep, args.epochs, batch_idx, len(train_loader),
					optimizer.param_groups[0]["lr"],
					avg_stat))
				if on_main:
					add_summary([{'type': 'scalars', 'tags': list(avg_stat.keys()),
					              'vals': list(avg_stat.values())}],
					            logger=logger, step=train_step, flag='train')

				del sample_cuda
				del avg_stat
				gc.collect()

			if on_main and train_step % args.save_freq == 0:
				torch.save({"step": train_step,
			                "model": model.module.state_dict(),
			                "optimizer": optimizer.state_dict(),
				            "scheduler": scheduler.state_dict(),
				            },
			                "{}/model_{:08d}.ckpt".format(args.save_path, train_step))

			if train_step % args.eval_freq == 0:
				with torch.no_grad():
					test(args, model, val_loader, train_step)
				model.train()

			train_step += 1

		del train_scores
		gc.collect()

def test(args, model, test_loader, train_step):
	model.eval_a_category()
	val_scores = DictAverageMeter()
	for batch_idx, sample in enumerate(test_loader):
		sample_cuda = dict2cuda(sample)
		ret = model(sample_cuda)
		preds = ret['preds']
		calc_stat(sample_cuda, preds[0], val_scores, label_type='stable')
		calc_stat(sample_cuda, preds[1], val_scores, label_type='contact')

	avg_stat = val_scores.mean()
	print("[Rank: {}] step {:06d}, stats: {}".format(args.local_rank, train_step, avg_stat))
	if on_main:
		add_summary([{'type': 'scalars', 'tags': list(avg_stat.keys()),
		              'vals': list(avg_stat.values())}],
		            logger=logger, step=train_step, flag='val')

	del sample_cuda
	del avg_stat
	del val_scores
	gc.collect()

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

	model: torch.nn.Module = UniNet_MT_V2(mask_channel=True, bootle_neck=256)
	if args.restore_path:
		checkpoint = torch.load(args.restore_path, map_location=torch.device("cpu"))
		model.load_state_dict(checkpoint['model'], strict=True)

	model.to(device)

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
	                       weight_decay=args.wd)

	train_set = MultiTaskDatasetV2(root_dir=args.root_path, list_path=args.train_list,
	                               use_aug=True, )
	print('train set ready.')
	val_set = MultiTaskDatasetV2(root_dir=args.root_path, list_path=args.val_list,
	                             use_aug=False,)
	print('val set ready.')
	if is_distributed:
		if args.sync_bn:
			model = apex.parallel.convert_syncbn_model(model)
			model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, )
			print('Convert BN to Sync_BN successful.')

		model = nn.parallel.DistributedDataParallel(
			model, device_ids=[args.local_rank], output_device=args.local_rank,)

		train_sampler = torch.utils.data.DistributedSampler(train_set, num_replicas=dist.get_world_size(),
		                                                    rank=dist.get_rank())
		val_sampler = torch.utils.data.DistributedSampler(val_set, num_replicas=dist.get_world_size(),
		                                                   rank=dist.get_rank())
	else:
		model = nn.DataParallel(model)
		train_sampler, val_sampler = None, None

	def worker_init_fn(worker_id):
		np.random.seed(np.random.get_state()[1][0] + worker_id)

	train_loader = DataLoader(train_set, args.batch_size, sampler=train_sampler,
	                          num_workers=args.num_workers, pin_memory=True,
	                          drop_last=True, shuffle=not is_distributed, worker_init_fn=worker_init_fn)
	val_loader = DataLoader(val_set, 128, sampler=val_sampler,
	                        num_workers=1, pin_memory=True,
	                        drop_last=False, shuffle=False, worker_init_fn=worker_init_fn)

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

	return model, optimizer, scheduler, train_loader, val_loader, train_sampler, start_step

if __name__ == '__main__':
	model, optimizer, scheduler, train_loader, val_loader, train_sampler, start_step = distribute_model(args)
	on_main = (not is_distributed) or (dist.get_rank() == 0)
	if on_main:
		os.makedirs(args.save_path, exist_ok=True)
		logger = SummaryWriter(args.save_path)
		print(args)

	main(args=args, model=model, optimizer=optimizer, scheduler=scheduler,
	     train_loader=train_loader, val_loader=val_loader, train_sampler=train_sampler, start_step=start_step)

