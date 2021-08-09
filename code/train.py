import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os
import random
import datetime
import logging
import sys
import importlib
import shutil
import zipfile
import emd_module as emd
import trans
from pathlib import Path
from tqdm import tqdm
from dataloader import DataLoader
from operations import furthest_point_sample

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = '../data'
# ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']
CATES = ['02691156']

def parse_args():
	'''PARAMETERS'''
	parser = argparse.ArgumentParser('CTF-Net')
	parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
	parser.add_argument('--mgpu', action='store_true', default=False, help='Whether to use all GPUs [default: False]')
	parser.add_argument('--batch_size', type=int, default=20, help='batch size in training [default: 24]')
	parser.add_argument('--model', default='model', help='model name [default: pointnet_cls]')
	parser.add_argument('--epoch',  default=200, type=int, help='number of epoch in training [default: 200]')
	parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
	parser.add_argument('--decay_rate', type=float, default=0.001, help='decay rate [default: 1e-4]')
	parser.add_argument('--input_scale_list',type=list,default=[2048,1024,512], help='number of input points in each scales')
	parser.add_argument('--output_scale_list', type=list, default=[128,512,2048], help='number of output points in each scales')
	parser.add_argument('--num_channel',type=int,default=3, help='number of channels')
	parser.add_argument('--weights',type=float,default=[3, 3, 1, 1, 1, 0.5],help='the loss weights of Regi, Orient, Comp, Consist, CR, RC')
	parser.add_argument('--comments', type=str, default='', help='Additional comments')
	return parser.parse_args()

def dist_emd(pc1, pc2):
	dist, assigment = emd.emdModule()(pc1, pc2, 0.001, 100)
	return torch.sqrt(dist).mean()

def dist_quat(q1, q2):
	minus = q1 - q2
	plus  = q1 + q2
	# batch dot product
	d_minus = torch.bmm(minus.view(args.batch_size, 1, -1), minus.view(args.batch_size, -1, 1))
	d_minus = torch.sqrt(d_minus)
	d_plus = torch.bmm(plus.view(args.batch_size, 1, -1), plus.view(args.batch_size, -1, 1))
	d_plus = torch.sqrt(d_plus)
	return torch.mean(torch.min(d_minus, d_plus))

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv2d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("Conv1d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find("BatchNorm1d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)

def main(args):
	def log_string(str):
		logger.info(str)
		print(str)

	def one_epoch(mode, epoch, dataloader, models, optimizers):
		loss_comp_sum, loss_regi_sum, loss_orient_sum, error_trans_sum, error_angle_sum = [],[],[],[],[]
		T_eye = torch.eye(4)
		T_eye = T_eye.reshape((1, 4, 4))
		T_eye = T_eye.repeat(args.batch_size, 1, 1)
		T_eye = T_eye.cuda()

		if epoch < 30:
			alpha1, alpha2 = 0.01, 0.02
		elif epoch < 50:
			alpha1, alpha2 = 0.05, 0.1
		else:
			alpha1, alpha2 = 0.1, 0.2

		regi_cr, regi_rc, comp_cr, comp_rc, orient_cr, orient_rc = models
		optimizer_regi, optimizer_comp, optimizer_orient = optimizers

		if mode == 'train':
			comp_cr = comp_cr.train()
			comp_rc = comp_rc.train()
			regi_cr = regi_cr.train()
			regi_rc = regi_rc.train()
			orient_cr = orient_cr.train()
			orient_rc = orient_rc.train()
		else:
			comp_cr = comp_cr.eval()
			comp_rc = comp_rc.eval()
			regi_cr = regi_cr.eval()
			regi_rc = regi_rc.eval()
			orient_cr = orient_cr.eval()
			orient_rc = orient_rc.eval()

		for batch_id, data in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
			if mode == 'train':
				optimizer_comp.zero_grad()
				optimizer_regi.zero_grad()
				optimizer_orient.zero_grad()
			'''
			# (1) Prepare data
			'''
			in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2 = data
			in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2 = in_pts1.float().cuda(), in_pts2.float().cuda(), gt_pts1.float().cuda(), gt_pts2.float().cuda(), gt_pts31.float().cuda(), gt_pts32.float().cuda(), gt_para12_r.float().cuda(), gt_para12_t.float().cuda(), gt_para21_r.float().cuda(), gt_para21_t.float().cuda(), gt_para_canonical_1.float().cuda(), gt_para_canonical_2.float().cuda()

			in_pts1_mid = furthest_point_sample(in_pts1, args.input_scale_list[1])
			in_pts2_mid = furthest_point_sample(in_pts2, args.input_scale_list[1])
			in_pts1_low = furthest_point_sample(in_pts1, args.input_scale_list[2])
			in_pts2_low = furthest_point_sample(in_pts2, args.input_scale_list[2])
			gt_pts1_mid = furthest_point_sample(gt_pts1, args.output_scale_list[1])
			gt_pts2_mid = furthest_point_sample(gt_pts2, args.output_scale_list[1])
			gt_pts31_mid = furthest_point_sample(gt_pts31, args.output_scale_list[1])
			gt_pts32_mid = furthest_point_sample(gt_pts32, args.output_scale_list[1])
			gt_pts1_low = furthest_point_sample(gt_pts1, args.output_scale_list[0])
			gt_pts2_low = furthest_point_sample(gt_pts2, args.output_scale_list[0])
			gt_pts31_low = furthest_point_sample(gt_pts31, args.output_scale_list[0])
			gt_pts32_low = furthest_point_sample(gt_pts32, args.output_scale_list[0])

			R21 = trans.quaternion2matrix_torch(gt_para21_r)
			T21 = trans.translation2matrix_torch(gt_para21_t)
			M21 = torch.bmm(T21, R21)
			R12 = trans.quaternion2matrix_torch(gt_para12_r)
			T12 = trans.translation2matrix_torch(gt_para12_t)
			M12 = torch.bmm(T12, R12)
			'''
			# (2) CR flow
			'''
			''' completion module (orient) '''
			out_cr_para_canonical_1 = orient_cr(in_pts1)
			out_cr_para_canonical_2 = orient_cr(in_pts2)

			loss_cr_orient = (dist_quat(out_cr_para_canonical_1, gt_para_canonical_1) + dist_quat(out_cr_para_canonical_2, gt_para_canonical_2)) / 2

			''' completion module (complete) '''
			R1_cr = trans.quaternion2matrix_torch(out_cr_para_canonical_1)
			in_pts1_low_transed = trans.transform_pts_torch(in_pts1_low, R1_cr)
			in_pts1_mid_transed = trans.transform_pts_torch(in_pts1_mid, R1_cr)
			in_pts1_transed = trans.transform_pts_torch(in_pts1, R1_cr)
			gt_pts1_low_transed = trans.transform_pts_torch(gt_pts1_low, R1_cr)
			gt_pts1_mid_transed = trans.transform_pts_torch(gt_pts1_mid, R1_cr)
			gt_pts1_transed = trans.transform_pts_torch(gt_pts1, R1_cr)
			R2_cr = trans.quaternion2matrix_torch(out_cr_para_canonical_2)
			in_pts2_low_transed = trans.transform_pts_torch(in_pts2_low, R2_cr)
			in_pts2_mid_transed = trans.transform_pts_torch(in_pts2_mid, R2_cr)
			in_pts2_transed = trans.transform_pts_torch(in_pts2, R2_cr)
			gt_pts2_low_transed = trans.transform_pts_torch(gt_pts2_low, R2_cr)
			gt_pts2_mid_transed = trans.transform_pts_torch(gt_pts2_mid, R2_cr)
			gt_pts2_transed = trans.transform_pts_torch(gt_pts2, R2_cr)

			out_cr_pts1_comp_low_transed, out_cr_pts1_comp_mid_transed, out_cr_pts1_comp_transed = comp_cr(in_pts1_transed, in_pts1_mid_transed, in_pts1_low_transed)
			out_cr_pts2_comp_low_transed, out_cr_pts2_comp_mid_transed, out_cr_pts2_comp_transed = comp_cr(in_pts2_transed, in_pts2_mid_transed, in_pts2_low_transed)

			dist1 = dist_emd(out_cr_pts1_comp_transed, gt_pts1_transed)
			dist2 = dist_emd(out_cr_pts2_comp_transed, gt_pts2_transed)
			dist3 = dist_emd(out_cr_pts1_comp_mid_transed, gt_pts1_mid_transed)
			dist4 = dist_emd(out_cr_pts2_comp_mid_transed, gt_pts2_mid_transed)
			dist5 = dist_emd(out_cr_pts1_comp_low_transed, gt_pts1_low_transed)
			dist6 = dist_emd(out_cr_pts2_comp_low_transed, gt_pts2_low_transed)
			loss_cr_comp = (dist1 + dist2) / 2 + alpha1 * (dist3 + dist4) / 2 + alpha2 * (dist5 + dist6) / 2

			R1_cr_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_cr_para_canonical_1))
			out_cr_pts1_comp = trans.transform_pts_torch(out_cr_pts1_comp_transed, R1_cr_inv)
			R2_cr_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_cr_para_canonical_2))
			out_cr_pts2_comp = trans.transform_pts_torch(out_cr_pts2_comp_transed, R2_cr_inv)

			''' registration module, pretrain using GT before epoch 30 '''
			if mode == 'train' and epoch < 30:
				in_cr_pts1_regi = torch.cat((in_pts1, gt_pts1), 1)
				in_cr_pts2_regi = torch.cat((in_pts2, gt_pts2), 1)
				gt_cr_pts12_regi = torch.cat((in_pts1, gt_pts1), 1)
				gt_cr_pts12_regi = trans.transform_pts_torch(gt_cr_pts12_regi, M12)
				gt_cr_pts21_regi = torch.cat((in_pts2, gt_pts2), 1)
				gt_cr_pts21_regi = trans.transform_pts_torch(gt_cr_pts21_regi, M21)
			else:
				in_cr_pts1_regi = torch.cat((in_pts1, out_cr_pts1_comp), 1)
				in_cr_pts2_regi = torch.cat((in_pts2, out_cr_pts2_comp), 1)
				gt_cr_pts12_regi = torch.cat((in_pts1, out_cr_pts1_comp), 1)
				gt_cr_pts12_regi = trans.transform_pts_torch(gt_cr_pts12_regi, M12)
				gt_cr_pts21_regi = torch.cat((in_pts2, out_cr_pts2_comp), 1)
				gt_cr_pts21_regi = trans.transform_pts_torch(gt_cr_pts21_regi, M21)
			in_cr_pts1_regi = furthest_point_sample(in_cr_pts1_regi, args.input_scale_list[0])
			in_cr_pts2_regi = furthest_point_sample(in_cr_pts2_regi, args.input_scale_list[0])

			out_cr_para21_r, out_cr_para21_t, out_cr_pts21_regi = regi_cr(in_cr_pts1_regi, in_cr_pts2_regi)
			out_cr_para12_r, out_cr_para12_t, out_cr_pts12_regi = regi_cr(in_cr_pts2_regi, in_cr_pts1_regi)

			loss_cr_regi_r = (dist_quat(out_cr_para21_r, gt_para21_r) + dist_quat(out_cr_para12_r, gt_para12_r)) / 2
			loss_cr_regi_t = (F.mse_loss(out_cr_para21_t, gt_para21_t) + F.mse_loss(out_cr_para12_t, gt_para12_t)) / 2
			loss_cr_regi = loss_cr_regi_r + loss_cr_regi_t

			'''
			# (3) RC flow
			'''
			''' registration module '''
			out_rc_para21_r, out_rc_para21_t, out_rc_pts21_regi = regi_rc(in_pts1, in_pts2)
			out_rc_para12_r, out_rc_para12_t, out_rc_pts12_regi = regi_rc(in_pts2, in_pts1)

			gt_rc_pts21_regi = trans.transform_pts_torch(in_pts2, M21)
			gt_rc_pts12_regi = trans.transform_pts_torch(in_pts1, M12)
			loss_rc_regi_r = (dist_quat(out_rc_para21_r, gt_para21_r) + dist_quat(out_rc_para12_r, gt_para12_r)) / 2
			loss_rc_regi_t = (F.mse_loss(out_rc_para21_t, gt_para21_t) + F.mse_loss(out_rc_para12_t, gt_para12_t)) / 2
			loss_rc_regi = loss_rc_regi_r + loss_rc_regi_t
			
			''' completion module (orient), pretrain using GT before epoch 30 '''
			if mode == 'train' and epoch < 30:
				in_rc_pts31_comp = torch.cat((in_pts1, gt_rc_pts21_regi), 1)
				in_rc_pts32_comp = torch.cat((in_pts2, gt_rc_pts12_regi), 1)
			else:
				in_rc_pts31_comp = torch.cat((in_pts1, out_rc_pts21_regi), 1)
				in_rc_pts32_comp = torch.cat((in_pts2, out_rc_pts12_regi), 1)
			in_rc_pts31_comp = furthest_point_sample(in_rc_pts31_comp, args.input_scale_list[0])
			in_rc_pts32_comp = furthest_point_sample(in_rc_pts32_comp, args.input_scale_list[0])
			in_rc_pts31_comp_mid = furthest_point_sample(in_rc_pts31_comp, args.input_scale_list[1])
			in_rc_pts32_comp_mid = furthest_point_sample(in_rc_pts32_comp, args.input_scale_list[1])
			in_rc_pts31_comp_low = furthest_point_sample(in_rc_pts31_comp, args.input_scale_list[2])
			in_rc_pts32_comp_low = furthest_point_sample(in_rc_pts32_comp, args.input_scale_list[2])

			out_rc_para_canonical_1 = orient_rc(in_rc_pts31_comp)
			out_rc_para_canonical_2 = orient_rc(in_rc_pts32_comp)

			loss_rc_orient = (dist_quat(out_rc_para_canonical_1, gt_para_canonical_1) + dist_quat(out_rc_para_canonical_2, gt_para_canonical_2)) / 2

			''' completion module (complete), pretrain using GT before epoch 30 '''
			R1_rc = trans.quaternion2matrix_torch(out_rc_para_canonical_1)
			R2_rc = trans.quaternion2matrix_torch(out_rc_para_canonical_2)
			in_rc_pts31_comp_low_transed = trans.transform_pts_torch(in_rc_pts31_comp_low, R1_rc)
			in_rc_pts31_comp_mid_transed = trans.transform_pts_torch(in_rc_pts31_comp_mid, R1_rc)
			in_rc_pts31_comp_transed = trans.transform_pts_torch(in_rc_pts31_comp, R1_rc)
			gt_pts31_low_transed = trans.transform_pts_torch(gt_pts31_low, R1_rc)
			gt_pts31_mid_transed = trans.transform_pts_torch(gt_pts31_mid, R1_rc)
			gt_pts31_transed = trans.transform_pts_torch(gt_pts31, R1_rc)
			in_rc_pts32_comp_low_transed = trans.transform_pts_torch(in_rc_pts32_comp_low, R2_rc)
			in_rc_pts32_comp_mid_transed = trans.transform_pts_torch(in_rc_pts32_comp_mid, R2_rc)
			in_rc_pts32_comp_transed = trans.transform_pts_torch(in_rc_pts32_comp, R2_rc)
			gt_pts32_low_transed = trans.transform_pts_torch(gt_pts32_low, R2_rc)
			gt_pts32_mid_transed = trans.transform_pts_torch(gt_pts32_mid, R2_rc)
			gt_pts32_transed = trans.transform_pts_torch(gt_pts32, R2_rc)

			out_rc_pts31_comp_low_transed, out_rc_pts31_comp_mid_transed, out_rc_pts31_comp_transed = comp_rc(in_rc_pts31_comp_transed, in_rc_pts31_comp_mid_transed, in_rc_pts31_comp_low_transed)
			out_rc_pts32_comp_low_transed, out_rc_pts32_comp_mid_transed, out_rc_pts32_comp_transed = comp_rc(in_rc_pts32_comp_transed, in_rc_pts32_comp_mid_transed, in_rc_pts32_comp_low_transed)

			dist7  = dist_emd(out_rc_pts31_comp_transed, gt_pts31_transed)
			dist8  = dist_emd(out_rc_pts32_comp_transed, gt_pts32_transed)
			dist9  = dist_emd(out_rc_pts31_comp_mid_transed, gt_pts31_mid_transed)
			dist10 = dist_emd(out_rc_pts32_comp_mid_transed, gt_pts32_mid_transed)
			dist11 = dist_emd(out_rc_pts31_comp_low_transed, gt_pts31_low_transed)
			dist12 = dist_emd(out_rc_pts32_comp_low_transed, gt_pts32_low_transed)
			loss_rc_comp = (dist7 + dist8) / 2 + alpha1 * (dist9 + dist10) / 2 + alpha2 * (dist11 + dist12) / 2

			R1_rc_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_rc_para_canonical_1))
			out_rc_pts31_comp = trans.transform_pts_torch(out_rc_pts31_comp_transed, R1_rc_inv)
			R2_rc_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_rc_para_canonical_2))
			out_rc_pts32_comp = trans.transform_pts_torch(out_rc_pts32_comp_transed, R2_rc_inv)

			out_rc_pts1_final = torch.cat((out_rc_pts31_comp, in_rc_pts31_comp), 1)
			out_rc_pts2_final = torch.cat((out_rc_pts32_comp, in_rc_pts32_comp), 1)

			''' calculate loss '''
			loss_orient = loss_cr_orient * args.weights[4] + loss_rc_orient * args.weights[5]
			loss_comp = loss_cr_comp * args.weights[4] + loss_rc_comp * args.weights[5]
			loss_regi = loss_cr_regi * args.weights[4] + loss_rc_regi * args.weights[5] * 6

			if epoch > 50:
				zero = torch.zeros(1).cuda()
				loss_orient_consist = (F.mse_loss(R1_cr, R1_rc) + F.mse_loss(R2_cr, R2_rc)) / 2
				R_cr_12 = trans.quaternion2matrix_torch(out_cr_para12_r)
				T_cr_12 = trans.translation2matrix_torch(out_cr_para12_t)
				M_cr_12 = torch.bmm(T_cr_12, R_cr_12)
				R_cr_21 = trans.quaternion2matrix_torch(out_cr_para21_r)
				T_cr_21 = trans.translation2matrix_torch(out_cr_para21_t)
				M_cr_21 = torch.bmm(T_cr_21, R_cr_21)
				R_rc_12 = trans.quaternion2matrix_torch(out_rc_para12_r)
				T_rc_12 = trans.translation2matrix_torch(out_rc_para12_t)
				M_rc_12 = torch.bmm(T_rc_12, R_rc_12)
				R_rc_21 = trans.quaternion2matrix_torch(out_rc_para21_r)
				T_rc_21 = trans.translation2matrix_torch(out_rc_para21_t)
				M_rc_21 = torch.bmm(T_rc_21, R_rc_21)
				loss_regi_consist_T = (F.mse_loss(torch.bmm(M_cr_12, M_cr_21), T_eye) + F.mse_loss(torch.bmm(M_rc_12, M_rc_21), T_eye)) / 2
				loss_regi_consist_R = (F.mse_loss(M_cr_12, M_rc_12) + F.mse_loss(M_cr_21, M_rc_21)) / 2
				loss_comp_consist = (F.mse_loss(dist_emd(torch.cat((in_pts1, out_cr_pts1_comp), 1), out_rc_pts1_final), zero) + F.mse_loss(dist_emd(torch.cat((in_pts2, out_cr_pts2_comp), 1), out_rc_pts2_final), zero)) / 2
				loss_consist = 3 * loss_orient_consist + 1 * loss_comp_consist + 3 * loss_regi_consist_T + 3 * loss_regi_consist_R
				loss_total = args.weights[0] * loss_regi + args.weights[1] * loss_orient + args.weights[2] * loss_comp + args.weights[3] * loss_consist
			else:
				loss_total = args.weights[0] * loss_regi + args.weights[1] * loss_orient + args.weights[2] * loss_comp

			if mode == 'train':
				loss_total.backward()
				optimizer_comp.step()
				optimizer_regi.step()
				optimizer_orient.step()

			loss_orient_sum.append(loss_orient.item())
			loss_comp_sum.append(loss_comp.item())
			loss_regi_sum.append(loss_regi.item())

			error_angle = 0
			''' rotation error cr '''
			diff12 = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_cr_para12_r), gt_para12_r)
			diff21 = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_cr_para21_r), gt_para21_r)
			diff12 = diff12.cpu().detach().numpy()
			diff21 = diff21.cpu().detach().numpy()
			for i in range(args.batch_size):
				error_axis12, error_angle12 = trans.quaternion2axisangle(diff12[i])
				error_axis21, error_angle21 = trans.quaternion2axisangle(diff21[i])
				error_angle += (np.abs(error_angle12) + np.abs(error_angle21)) / 2 * 180 / np.pi
			''' rotation error rc '''
			diff12 = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_rc_para12_r), gt_para12_r)
			diff21 = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_rc_para21_r), gt_para21_r)
			diff12 = diff12.cpu().detach().numpy()
			diff21 = diff21.cpu().detach().numpy()
			for i in range(args.batch_size):
				error_axis12, error_angle12 = trans.quaternion2axisangle(diff12[i])
				error_axis21, error_angle21 = trans.quaternion2axisangle(diff21[i])
				error_angle += (np.abs(error_angle12) + np.abs(error_angle21)) / 2 * 180 / np.pi
			error_angle_sum.append(error_angle / args.batch_size / 2)
			''' translation error '''
			error_cr_regi_t = loss_cr_regi_t
			error_rc_regi_t = loss_rc_regi_t
			error_trans = (error_cr_regi_t + error_rc_regi_t) / 2
			error_trans_sum.append(error_trans.item())

		log_string(mode+' loss orient : %f' % np.mean(loss_orient_sum))
		log_string(mode+' loss comp   : %f' % np.mean(loss_comp_sum))
		log_string(mode+' loss regi   : %f' % np.mean(loss_regi_sum))
		log_string(mode+' error angle : %f' % np.mean(error_angle_sum))
		log_string(mode+' error trans : %f' % np.mean(error_trans_sum))
		return np.mean(loss_orient_sum), np.mean(loss_comp_sum), np.mean(loss_regi_sum)


	if args.mgpu:
		GPU_LIST = [0, 1, 2, 3]
		args.batch_size = len(GPU_LIST) * args.batch_size
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu		

	'''CREATE DIR'''
	timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
	experiment_dir = Path(ROOT_DIR).joinpath('../output/')
	experiment_dir.mkdir(exist_ok=True)
	experiment_dir = experiment_dir.joinpath(args.model+'_'+timestr+'_'+args.comments)
	experiment_dir.mkdir(exist_ok=True)
	checkpoints_dir = experiment_dir.joinpath('checkpoints/')
	checkpoints_dir.mkdir(exist_ok=True)
	log_dir = experiment_dir.joinpath('logs/')
	log_dir.mkdir(exist_ok=True)

	'''BACKUP CODE'''
	zip_name = os.path.join(experiment_dir) + "/code.zip"
	filelist = []
	for root, dirs, files in os.walk(ROOT_DIR):
		for name in files:
			filelist.append(os.path.join(root, name))
	zip_code = zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED)
	for tar in filelist:
		arcname = tar[len(ROOT_DIR):]
		zip_code.write(tar, arcname)
	zip_code.close()

	'''LOG'''
	logger = logging.getLogger("Model")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	log_string('Parameters: ')
	log_string(args)

	'''DATA LOADING'''
	log_string('Load dataset ...')
	TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=args.input_scale_list[0], split='train', category=CATES)
	VAL_DATASET = DataLoader(root=DATA_PATH, npoint=args.input_scale_list[0], split='val', category=CATES)
	trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
	valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

	'''MODEL LOADING'''
	MODEL = importlib.import_module(args.model)
	shutil.copy('./%s.py' % args.model, str(experiment_dir))

	orient_cr = MODEL.orient_model().cuda()
	orient_rc = MODEL.orient_model().cuda()
	regi_cr = MODEL.regi_model().cuda()
	regi_rc = MODEL.regi_model().cuda()
	comp_cr = MODEL.comp_model(args.input_scale_list, args.output_scale_list, args.num_channel)
	comp_cr.apply(weights_init_normal).cuda()
	comp_rc = MODEL.comp_model(args.input_scale_list, args.output_scale_list, args.num_channel)
	comp_rc.apply(weights_init_normal).cuda()
	if args.mgpu:
		orient_cr = torch.nn.DataParallel(orient_cr, device_ids=GPU_LIST)
		orient_rc = torch.nn.DataParallel(orient_rc, device_ids=GPU_LIST)
		regi_cr = torch.nn.DataParallel(regi_cr, device_ids=GPU_LIST)
		regi_rc = torch.nn.DataParallel(regi_rc, device_ids=GPU_LIST)
		comp_cr = torch.nn.DataParallel(comp_cr, device_ids=GPU_LIST)
		comp_rc = torch.nn.DataParallel(comp_rc, device_ids=GPU_LIST)

	manual_seed = random.randint(1, 10000)
	torch.cuda.manual_seed_all(manual_seed)

	if args.mgpu:
		optimizer_comp = torch.optim.Adam(
			list(comp_cr.parameters()) + list(comp_rc.parameters()),
			lr=args.learning_rate * len(GPU_LIST),
			betas=(0.9, 0.999),
			eps=1e-05,
			weight_decay=args.decay_rate
		)
		optimizer_regi = torch.optim.Adam(
			list(regi_cr.parameters()) + list(regi_rc.parameters()),
			lr=args.learning_rate * len(GPU_LIST),
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=args.decay_rate
		)
		optimizer_orient = torch.optim.Adam(
			list(orient_cr.parameters()) + list(orient_rc.parameters()),
			lr=args.learning_rate * len(GPU_LIST),
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=args.decay_rate
		)
	else:
		optimizer_comp = torch.optim.Adam(
			list(comp_cr.parameters()) + list(comp_rc.parameters()),
			lr=args.learning_rate,
			betas=(0.9, 0.999),
			eps=1e-05,
			weight_decay=args.decay_rate
		)
		optimizer_regi = torch.optim.Adam(
			list(regi_cr.parameters()) + list(regi_rc.parameters()),
			lr=args.learning_rate,
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=args.decay_rate
		)
		optimizer_orient = torch.optim.Adam(
			list(orient_cr.parameters()) + list(orient_rc.parameters()),
			lr=args.learning_rate,
			betas=(0.9, 0.999),
			eps=1e-08,
			weight_decay=args.decay_rate
		)

	scheduler_comp = torch.optim.lr_scheduler.StepLR(optimizer_comp, step_size=40, gamma=0.2)
	scheduler_regi = torch.optim.lr_scheduler.StepLR(optimizer_regi, step_size=20, gamma=0.7)
	scheduler_orient = torch.optim.lr_scheduler.StepLR(optimizer_orient, step_size=20, gamma=0.7)

	global_epoch = 0
	loss_total_best = 1000

	'''TRANING'''
	logger.info('---Start training---')
	for epoch in range(args.epoch):
		log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
		loss_orient, loss_comp, loss_regi = one_epoch('train', epoch, trainDataLoader, [regi_cr, regi_rc, comp_cr, comp_rc, orient_cr, orient_rc], [optimizer_regi, optimizer_comp, optimizer_orient])
		scheduler_comp.step()
		scheduler_regi.step()
		scheduler_orient.step()

		with torch.no_grad():
			loss_orient, loss_comp, loss_regi = one_epoch('val', epoch, valDataLoader, [regi_cr, regi_rc, comp_cr, comp_rc, orient_cr, orient_rc], [optimizer_regi, optimizer_comp, optimizer_orient])

			loss_total = loss_comp + loss_regi
			if (loss_total <= loss_total_best):
				loss_total_best = loss_total
				logger.info('Saving model...')

				savepath = str(checkpoints_dir) + '/best_comp_cr.pth'
				log_string('Saving at %s'% savepath)
				if args.mgpu:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_comp,
						'model_state_dict': comp_cr.module.state_dict()
					}
				else:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_comp,
						'model_state_dict': comp_cr.state_dict()
					}
				torch.save(state, savepath)

				savepath = str(checkpoints_dir) + '/best_regi_cr.pth'
				log_string('Saving at %s'% savepath)
				if args.mgpu:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_regi,
						'model_state_dict': regi_cr.module.state_dict()
					}
				else:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_regi,
						'model_state_dict': regi_cr.state_dict()
					}
				torch.save(state, savepath)

				savepath = str(checkpoints_dir) + '/best_orient_cr.pth'
				log_string('Saving at %s'% savepath)
				if args.mgpu:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_orient,
						'model_state_dict': orient_cr.module.state_dict()
					}
				else:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_orient,
						'model_state_dict': orient_cr.state_dict()
					}
				torch.save(state, savepath)

				savepath = str(checkpoints_dir) + '/best_comp_rc.pth'
				log_string('Saving at %s'% savepath)
				if args.mgpu:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_comp,
						'model_state_dict': comp_rc.module.state_dict()
					}
				else:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_comp,
						'model_state_dict': comp_rc.state_dict()
					}
				torch.save(state, savepath)

				savepath = str(checkpoints_dir) + '/best_regi_rc.pth'
				log_string('Saving at %s'% savepath)
				if args.mgpu:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_regi,
						'model_state_dict': regi_rc.module.state_dict()
					}
				else:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_regi,
						'model_state_dict': regi_rc.state_dict()
					}
				torch.save(state, savepath)

				savepath = str(checkpoints_dir) + '/best_orient_rc.pth'
				log_string('Saving at %s'% savepath)
				if args.mgpu:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_orient,
						'model_state_dict': orient_rc.module.state_dict()
					}
				else:
					state = {
						'epoch': epoch + 1,
						'total loss': loss_orient,
						'model_state_dict': orient_rc.state_dict()
					}
				torch.save(state, savepath)
				log_string('Best total loss: %f' % loss_total_best)

			global_epoch += 1 

	logger.info('---End of training---')

if __name__ == '__main__':
	args = parse_args()
	main(args)
