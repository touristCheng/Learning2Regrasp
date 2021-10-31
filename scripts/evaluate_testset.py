import os
from glob import glob
import json
from collections import OrderedDict
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

#######


g_ckpt="./checkpoints/generator/model_00368000.ckpt"
c_ckpt="./checkpoints/classifier/model_00160000.ckpt"

root_path="./dataset/test_plys"
all_test_list = glob('./dataset/data_list/test_groups/*.txt')

test_root_dir = './test_results/'

#######

mesh_dir = './dataset/test_urdf'
init_dir = './dataset/test_plys'
total_rounds = range(1, 6)

def eval_a_category(data_list, pred_dir, save_dir, mesh_dir, init_dir):
	data_pairs = open(data_list, 'r').readlines()
	data_pairs = [str(x).strip().split('-') for x in data_pairs]

	for sup_name, obj_name in data_pairs:
		sup_urdf = '{}/{}.urdf'.format(mesh_dir, sup_name)
		obj_urdf = '{}/{}.urdf'.format(mesh_dir, obj_name)
		init_sup_pose = '{}/{}_init_pose.txt'.format(init_dir, sup_name)
		init_obj_pose = '{}/{}_init_pose.txt'.format(init_dir, obj_name)
		pose_dir = '{}/{}-{}'.format(pred_dir, sup_name, obj_name)

		cmd = 'python3 evaluate_poses.py --obj_path {} --sup_path {} ' \
		      '--init_obj_pose {} --init_sup_pose {} --transforms {} --save_dir {}' \
			.format(obj_urdf, sup_urdf, init_obj_pose, init_sup_pose, pose_dir, save_dir)
		os.system(cmd)

def read_file(path):
	with open(path, 'r') as fd:
		data_list = json.load(fd, object_hook=OrderedDict)
		return data_list

def write_file(path, data_list):
	dir_name = osp.dirname(path)
	if dir_name:
		os.makedirs(dir_name, exist_ok=True)
	with open(path, 'w') as f:
		json.dump(data_list, f)

def plot(x_names, means, stds, ax, max_y=1, title=''):
	xs = np.arange(len(x_names))
	ax.bar(xs, means, yerr=stds, align='center', alpha=0.5, ecolor='green', capsize=3)
	ax.set_xticks(xs, )
	ax.set_xticklabels(x_names, rotation=75)
	ax.set_ylim(0, max_y)
	ax.yaxis.grid(True)
	ax.axhline(y=np.mean(means), color='r', linestyle='--')
	ax.set_ylabel(title)

	for x, val in zip(xs, means):
		ax.annotate("{:.3f}".format(val),
	                xy = (x, 0.1),             # top left corner of the histogram bar
	                xytext = (0,0.2),             # offsetting label position above its bar
	                textcoords = "offset points", # Offset (in points) from the *xy* value
	                ha = 'center', va = 'bottom'
	                )

def load_eval_results(acc_stats, cnt_stats, root_dir, cat_name, thresh):
	all_paths = glob('{}/*.json'.format(root_dir, ))
	for path in all_paths:
		# pair_name = osp.basename(path).split('.')[0]
		# cat_name = pair_name #

		data = read_file(path)[0]
		acc_val = float(data['acc'][str(thresh)])
		cnt_val = float(data['cnt'][str(thresh)])
		acc_stats[cat_name] = acc_stats.get(cat_name, []) + [acc_val]
		cnt_stats[cat_name] = cnt_stats.get(cat_name, []) + [cnt_val]

def eval_all():

	for path in all_test_list:
		pair_name = osp.basename(path).split('.')[0]

		for i in total_rounds:
			testset_root = '{}/round_{}'.format(test_root_dir, i)
			save_dir = '{}/results/{}'.format(testset_root, pair_name)
			pred_dir = '{}/{}'.format(testset_root, pair_name)
			eval_a_category(data_list=path,
			                pred_dir=pred_dir,
			                save_dir=save_dir,
			                mesh_dir=mesh_dir,
			                init_dir=init_dir)

def stats(thresh=0.8, ):
	acc_stats = {}
	cnt_stats = {}

	for path in all_test_list:
		pair_name = osp.basename(path).split('.')[0]
		for i in total_rounds:
			testset_root = '{}/round_{}'.format(test_root_dir, i)
			save_dir = '{}/results/{}'.format(testset_root, pair_name)
			load_eval_results(acc_stats=acc_stats, cnt_stats=cnt_stats,
			                  root_dir=save_dir, thresh=thresh, cat_name=pair_name)

	def process_stat(stat, ):
		x_names = sorted(list(stat.keys()))
		means = []
		stds = []
		for k in x_names:
			mean_k = np.mean(stat[k])
			std_k = np.std(stat[k])
			means.append(mean_k)
			stds.append(std_k)
		x_names = x_names + ['mean', ]
		mean_ = np.nanmean(means)
		std_ = np.nanstd(means)
		means.append(mean_)
		stds.append(std_)
		return x_names, means, stds

	x_names, acc_mean, acc_std = process_stat(acc_stats)
	_, cnt_mean, cnt_std = process_stat(cnt_stats)

	_, ax = plt.subplots(2, 1, figsize=(30, 10))
	plot(x_names=x_names, means=acc_mean, stds=acc_std, ax=ax[0], title='accuracy')
	plot(x_names=x_names, means=cnt_mean, stds=cnt_std, ax=ax[1], max_y=128, title='diversity #')

	plt.tight_layout()
	plt.savefig('{}/hist_{}.pdf'.format(test_root_dir, thresh))
	plt.show()

def infer_all():
	for path in all_test_list:
		pair_name = osp.basename(path).split('.')[0]
		for i in total_rounds:
			save_path="{}/round_{}/{}".format(test_root_dir, i, pair_name)
			os.makedirs(save_path, exist_ok=True)
			cmd = 'python3 inference.py --root_path {} --test_list {} --save_path {} ' \
			      '--generator_ckpt {} --stable_critic_ckpt {} --z_dim 3 --num_iter 1 ' \
				  '--pose_num 128 --rot_rp axis_angle --device cuda --render_ply' \
				  .format(root_path, path, save_path, g_ckpt, c_ckpt, )
			os.system(cmd)

def main():
	infer_all()
	eval_all()
	stats(0.8)


if __name__ == '__main__':
    main()