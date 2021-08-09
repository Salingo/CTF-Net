import torch
import torch.nn.functional as F
import argparse
import numpy as np
import os
import logging
import sys
import importlib.util
import h5py
import trans
import emd_module as emd
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
	parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
	parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
	parser.add_argument('--input_scale_list',type=list,default=[2048,1024,512],help='number of points in each scales')
	parser.add_argument('--output_scale_list', type=list, default=[128,512,2048], help='number of output points in each scales')
	parser.add_argument('--num_channel',type=int,default=3,help='number of channels')
	parser.add_argument('--save_pts', action='store_true', default=False, help='Whether to save predicted pts files')
	parser.add_argument('--log_dir', type=str, default='', help='Experiment root')	
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

def main(args):
	def log_string(str):
		logger.info(str)
		print(str)

	'''HYPER PARAMETER'''
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	'''CREATE DIR'''
	experiment_dir = Path(ROOT_DIR).joinpath('../output/', args.log_dir)
	#experiment_dir = experiment_dir + args.log_dir
	output_dir = experiment_dir.joinpath('output/')
	output_dir.mkdir(exist_ok=True)

	'''LOG'''
	logger = logging.getLogger("Model")
	logger.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)
	log_string('Parameters: ')
	log_string(args)

	'''DATA LOADING'''
	log_string('Load dataset ...')
	TEST_DATASET = DataLoader(root=DATA_PATH, npoint=args.input_scale_list[0], split='test', category=CATES)
	testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, num_workers=0, shuffle=False)

	'''MODEL LOADING'''
	log_string('Load model ...')
	model_name = os.listdir(experiment_dir.joinpath('logs/'))[0].split('.')[0]
	spec = importlib.util.spec_from_file_location(model_name, os.path.join(ROOT_DIR,experiment_dir,model_name)+'.py')
	MODEL = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(MODEL)

	orient_cr = MODEL.orient_model().cuda()
	orient_rc = MODEL.orient_model().cuda()
	regi_cr = MODEL.regi_model().cuda()
	regi_rc = MODEL.regi_model().cuda()
	comp_cr = MODEL.comp_model(args.input_scale_list, args.output_scale_list, args.num_channel).cuda()
	comp_rc = MODEL.comp_model(args.input_scale_list, args.output_scale_list, args.num_channel).cuda()

	checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_comp_cr.pth')
	comp_cr.load_state_dict(checkpoint['model_state_dict'])
	checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_orient_cr.pth')
	orient_cr.load_state_dict(checkpoint['model_state_dict'])
	checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_regi_cr.pth')
	regi_cr.load_state_dict(checkpoint['model_state_dict'])
	checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_comp_rc.pth')
	comp_rc.load_state_dict(checkpoint['model_state_dict'])
	checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_orient_rc.pth')
	orient_rc.load_state_dict(checkpoint['model_state_dict'])
	checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_regi_rc.pth')
	regi_rc.load_state_dict(checkpoint['model_state_dict'])

	log_string('Testing...\n')
	with torch.no_grad():
		ptss_name, in_ptss1, in_ptss2, out_ptss1, out_ptss2, out_ptss31, out_ptss32, out_paras12_r, out_paras12_t, out_paras21_r, out_paras21_t, gt_ptss1, gt_ptss2, gt_ptss31, gt_ptss32, gt_paras12_r, gt_paras12_t, gt_paras21_r, gt_paras21_t = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
		error_total_sum, error_cr_comp_gen_emd_sum, error_rc_comp_gen_emd_sum, error_cr_comp_full_emd_sum, error_rc_comp_full_emd_sum, error_cr_regi_t_sum, error_rc_regi_t_sum, error_cr_angle_sum, error_rc_angle_sum = [],[],[],[],[],[],[],[],[]

		comp_cr = comp_cr.eval()
		regi_cr = regi_cr.eval()
		orient_cr = orient_cr.eval()
		comp_rc = comp_rc.eval()
		regi_rc = regi_rc.eval()
		orient_rc = orient_rc.eval()
		for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
			in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2 = data
			in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2 = in_pts1.float().cuda(), in_pts2.float().cuda(), gt_pts1.float().cuda(), gt_pts2.float().cuda(), gt_pts31.float().cuda(), gt_pts32.float().cuda(), gt_para12_r.float().cuda(), gt_para12_t.float().cuda(), gt_para21_r.float().cuda(), gt_para21_t.float().cuda(), gt_para_canonical_1.float().cuda(), gt_para_canonical_2.float().cuda()
			
			in_pts1_mid = furthest_point_sample(in_pts1, args.input_scale_list[1])
			in_pts2_mid = furthest_point_sample(in_pts2, args.input_scale_list[1])
			in_pts1_low = furthest_point_sample(in_pts1, args.input_scale_list[2])
			in_pts2_low = furthest_point_sample(in_pts2, args.input_scale_list[2])

			R21 = trans.quaternion2matrix_torch(gt_para21_r)
			T21 = trans.translation2matrix_torch(gt_para21_t)
			M21 = torch.bmm(T21, R21)
			R12 = trans.quaternion2matrix_torch(gt_para12_r)
			T12 = trans.translation2matrix_torch(gt_para12_t)
			M12 = torch.bmm(T12, R12)

			'''
			# CR flow
			'''
			out_cr_para_canonical_1 = orient_cr(in_pts1)
			out_cr_para_canonical_2 = orient_cr(in_pts2)

			R1 = trans.quaternion2matrix_torch(out_cr_para_canonical_1)
			in_pts1_transed = trans.transform_pts_torch(in_pts1, R1)
			gt_pts1_transed = trans.transform_pts_torch(gt_pts1, R1)
			R2 = trans.quaternion2matrix_torch(out_cr_para_canonical_2)
			in_pts2_transed = trans.transform_pts_torch(in_pts2, R2)
			gt_pts2_transed = trans.transform_pts_torch(gt_pts2, R2)
			in_pts1_low_transed = trans.transform_pts_torch(in_pts1_low, R1)
			in_pts1_mid_transed = trans.transform_pts_torch(in_pts1_mid, R1)
			in_pts2_low_transed = trans.transform_pts_torch(in_pts2_low, R2)
			in_pts2_mid_transed = trans.transform_pts_torch(in_pts2_mid, R2)		
			out_cr_pts1_comp_low_transed, out_cr_pts1_comp_mid_transed, out_cr_pts1_comp_transed = comp_cr(in_pts1_transed, in_pts1_mid_transed, in_pts1_low_transed)
			out_cr_pts2_comp_low_transed, out_cr_pts2_comp_mid_transed, out_cr_pts2_comp_transed = comp_cr(in_pts2_transed, in_pts2_mid_transed, in_pts2_low_transed)
			dist1_emd = dist_emd(out_cr_pts1_comp_transed, gt_pts1_transed)
			dist2_emd = dist_emd(out_cr_pts2_comp_transed, gt_pts2_transed)
			error_cr_comp_gen_emd = (dist1_emd + dist2_emd) / 2

			R1_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_cr_para_canonical_1))
			out_cr_pts1_comp = trans.transform_pts_torch(out_cr_pts1_comp_transed, R1_inv)
			R2_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_cr_para_canonical_2))
			out_cr_pts2_comp = trans.transform_pts_torch(out_cr_pts2_comp_transed, R2_inv)

			dist3_emd = dist_emd(torch.cat((out_cr_pts1_comp, in_pts1), 1), torch.cat((gt_pts1, in_pts1), 1))
			dist4_emd = dist_emd(torch.cat((out_cr_pts2_comp, in_pts2), 1), torch.cat((gt_pts2, in_pts2), 1))
			error_cr_comp_full_emd = (dist3_emd + dist4_emd) / 2

			in_cr_pts1_regi = torch.cat((in_pts1, out_cr_pts1_comp), 1)
			in_cr_pts2_regi = torch.cat((in_pts2, out_cr_pts2_comp), 1)
			in_cr_pts1_regi = furthest_point_sample(in_cr_pts1_regi, args.input_scale_list[0])
			in_cr_pts2_regi = furthest_point_sample(in_cr_pts2_regi, args.input_scale_list[0])

			out_cr_para21_r, out_cr_para21_t, out_cr_pts21_regi = regi_cr(in_cr_pts1_regi, in_cr_pts2_regi)
			out_cr_para12_r, out_cr_para12_t, out_cr_pts12_regi = regi_cr(in_cr_pts2_regi, in_cr_pts1_regi)

			gt_cr_pts12_regi = torch.cat((in_pts1, out_cr_pts1_comp), 1)
			gt_cr_pts12_regi = trans.transform_pts_torch(gt_cr_pts12_regi, M12)
			gt_cr_pts21_regi = torch.cat((in_pts2, out_cr_pts2_comp), 1)
			gt_cr_pts21_regi = trans.transform_pts_torch(gt_cr_pts21_regi, M21)
			error_cr_regi_t = (F.mse_loss(out_cr_para21_t, gt_para21_t) + F.mse_loss(out_cr_para12_t, gt_para12_t)) / 2
			error_cr_regi_r = (dist_quat(out_cr_para21_r, gt_para21_r) + dist_quat(out_cr_para12_r, gt_para12_r)) / 2

			out_cr_pts1_final = torch.cat((out_cr_pts1_comp, in_pts1), 1)
			out_cr_pts2_final = torch.cat((out_cr_pts2_comp, in_pts2), 1)

			'''
			# RC flow
			'''
			out_rc_para21_r, out_rc_para21_t, out_rc_pts21_regi = regi_rc(in_pts1, in_pts2)
			out_rc_para12_r, out_rc_para12_t, out_rc_pts12_regi = regi_rc(in_pts2, in_pts1)

			gt_rc_pts21_regi = trans.transform_pts_torch(in_pts2, M21)
			gt_rc_pts12_regi = trans.transform_pts_torch(in_pts1, M12)
			error_rc_regi_t = (F.mse_loss(out_rc_para21_t, gt_para21_t) + F.mse_loss(out_rc_para12_t, gt_para12_t)) / 2
			error_rc_regi_r = (dist_quat(out_rc_para21_r, gt_para21_r) + dist_quat(out_rc_para12_r, gt_para12_r)) / 2

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

			R1 = trans.quaternion2matrix_torch(out_rc_para_canonical_1)
			in_rc_pts31_comp_transed = trans.transform_pts_torch(in_rc_pts31_comp, R1)
			gt_pts31_transed = trans.transform_pts_torch(gt_pts31, R1)
			R2 = trans.quaternion2matrix_torch(out_rc_para_canonical_2)
			in_rc_pts32_comp_transed = trans.transform_pts_torch(in_rc_pts32_comp, R2)
			gt_pts32_transed = trans.transform_pts_torch(gt_pts32, R2)
			in_rc_pts31_comp_low_transed = trans.transform_pts_torch(in_rc_pts31_comp_low, R1)
			in_rc_pts31_comp_mid_transed = trans.transform_pts_torch(in_rc_pts31_comp_mid, R1)	
			in_rc_pts32_comp_low_transed = trans.transform_pts_torch(in_rc_pts32_comp_low, R2)
			in_rc_pts32_comp_mid_transed = trans.transform_pts_torch(in_rc_pts32_comp_mid, R2)
			out_rc_pts31_comp_low_transed, out_rc_pts31_comp_mid_transed, out_rc_pts31_comp_transed = comp_rc(in_rc_pts31_comp_transed, in_rc_pts31_comp_mid_transed, in_rc_pts31_comp_low_transed)
			out_rc_pts32_comp_low_transed, out_rc_pts32_comp_mid_transed, out_rc_pts32_comp_transed = comp_rc(in_rc_pts32_comp_transed, in_rc_pts32_comp_mid_transed, in_rc_pts32_comp_low_transed)

			dist7_emd = dist_emd(out_rc_pts31_comp_transed, gt_pts31_transed)
			dist8_emd = dist_emd(out_rc_pts32_comp_transed, gt_pts32_transed)
			error_rc_comp_gen_emd = (dist7_emd + dist8_emd) / 2
			
			R1_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_rc_para_canonical_1))
			out_rc_pts31_comp = trans.transform_pts_torch(out_rc_pts31_comp_transed, R1_inv)
			R2_inv = trans.quaternion2matrix_torch(trans.quaternion_inv_torch(out_rc_para_canonical_2))
			out_rc_pts32_comp = trans.transform_pts_torch(out_rc_pts32_comp_transed, R2_inv)

			dist9_emd = dist_emd(torch.cat((out_rc_pts31_comp, in_rc_pts31_comp), 1), torch.cat((gt_pts31, in_rc_pts31_comp), 1))
			dist10_emd = dist_emd(torch.cat((out_rc_pts32_comp, in_rc_pts32_comp), 1), torch.cat((gt_pts32, in_rc_pts32_comp), 1))
			error_rc_comp_full_emd = (dist9_emd + dist10_emd) / 2

			out_rc_pts1_final = torch.cat((out_rc_pts31_comp, in_rc_pts31_comp), 1)
			out_rc_pts2_final = torch.cat((out_rc_pts32_comp, in_rc_pts32_comp), 1)

			error_cr_angle = 0
			error_rc_angle = 0

			error_cr_comp_gen_emd_sum.append(error_cr_comp_gen_emd.item())
			error_cr_comp_full_emd_sum.append(error_cr_comp_full_emd.item())
			error_cr_regi_t_sum.append(error_cr_regi_t.item())
			error_rc_comp_gen_emd_sum.append(error_rc_comp_gen_emd.item())
			error_rc_comp_full_emd_sum.append(error_rc_comp_full_emd.item())
			error_rc_regi_t_sum.append(error_rc_regi_t.item())
			error_total = (error_cr_regi_r + error_cr_regi_t + error_cr_comp_gen_emd + error_rc_regi_r + error_rc_regi_t + error_rc_comp_gen_emd) / 2
			error_total_sum.append(error_total.item())

			diff12_cr = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_cr_para12_r), gt_para12_r)
			diff21_cr = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_cr_para21_r), gt_para21_r)
			diff12_cr = diff12_cr.cpu().detach().numpy()
			diff21_cr = diff21_cr.cpu().detach().numpy()
			for i in range(args.batch_size):
				error_axis12, error_angle12 = trans.quaternion2axisangle(diff12_cr[i])
				error_axis21, error_angle21 = trans.quaternion2axisangle(diff21_cr[i])
				error_cr_angle += (np.abs(error_angle12) + np.abs(error_angle21)) / 2 * 180 / np.pi
			error_cr_angle_sum.append(error_cr_angle / args.batch_size)
			diff12_rc = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_rc_para12_r), gt_para12_r)
			diff21_rc = trans.quat_multiply_torch(trans.quaternion_inv_torch(out_rc_para21_r), gt_para21_r)
			diff12_rc = diff12_rc.cpu().detach().numpy()
			diff21_rc = diff21_rc.cpu().detach().numpy()
			for i in range(args.batch_size):
				error_axis12, error_angle12 = trans.quaternion2axisangle(diff12_rc[i])
				error_axis21, error_angle21 = trans.quaternion2axisangle(diff21_rc[i])
				error_rc_angle += (np.abs(error_angle12) + np.abs(error_angle21)) / 2 * 180 / np.pi
			error_rc_angle_sum.append(error_rc_angle / args.batch_size)
				
			pts_name = TEST_DATASET.get_name(batch_id)

			in_pts1, in_pts2, out_cr_para21_r, out_cr_para12_r, out_cr_para21_t, out_cr_para12_t, out_cr_pts1_final, out_cr_pts2_final, out_rc_pts1_final, out_rc_pts2_final = in_pts1.cpu().numpy()[0], in_pts2.cpu().numpy()[0], out_cr_para21_r.cpu().numpy()[0], out_cr_para12_r.cpu().numpy()[0], out_cr_para21_t.cpu().numpy()[0], out_cr_para12_t.cpu().numpy()[0], out_cr_pts1_final.cpu().numpy()[0], out_cr_pts2_final.cpu().numpy()[0], out_rc_pts1_final.cpu().numpy()[0], out_rc_pts2_final.cpu().numpy()[0]

			if args.save_pts:
				ptss_name.append(pts_name)
				in_ptss1.append(in_pts1)
				in_ptss2.append(in_pts2)
				out_ptss1.append(out_cr_pts1_final)
				out_ptss2.append(out_cr_pts2_final)
				out_ptss31.append(out_rc_pts1_final)
				out_ptss32.append(out_rc_pts2_final)
				out_paras12_r.append(out_cr_para12_r)
				out_paras12_t.append(out_cr_para12_t)
				out_paras21_r.append(out_cr_para21_r)
				out_paras21_t.append(out_cr_para21_t)

		log_string('CR flow - generate emd: %f' % np.mean(error_cr_comp_gen_emd_sum))
		log_string('RC flow - generate emd: %f' % np.mean(error_rc_comp_gen_emd_sum))
		log_string('CR flow - full emd: %f' % np.mean(error_cr_comp_full_emd_sum))
		log_string('RC flow - full emd: %f' % np.mean(error_rc_comp_full_emd_sum))
		log_string('CR flow - trans error: %f' % np.mean(error_cr_regi_t_sum))
		log_string('RC flow - trans error: %f' % np.mean(error_rc_regi_t_sum))
		log_string('CR flow - angle error: %f' % np.mean(error_cr_angle_sum))
		log_string('RC flow - angle error: %f' % np.mean(error_rc_angle_sum))

		if args.save_pts:
			order = np.argsort(error_total_sum)
			# print(np.array(loss_total_all)[order])
			ptss_name = np.array(ptss_name,'S')[order]
			in_ptss1 = np.array(in_ptss1)[order]
			in_ptss2 = np.array(in_ptss2)[order]
			out_ptss1 = np.array(out_ptss1)[order]
			out_ptss2 = np.array(out_ptss2)[order]
			out_ptss31 = np.array(out_ptss31)[order]
			out_ptss32 = np.array(out_ptss32)[order]
			out_paras12_r = np.array(out_paras12_r)[order]
			out_paras12_t = np.array(out_paras12_t)[order]
			out_paras21_r = np.array(out_paras21_r)[order]
			out_paras21_t = np.array(out_paras21_t)[order]

			with h5py.File(output_dir+'output.h5', 'w') as f:
				f.create_dataset(name="pts_name", data=ptss_name, compression="gzip")
				f.create_dataset(name="in_pts1", data=in_ptss1, compression="gzip")
				f.create_dataset(name="in_pts2", data=in_ptss2, compression="gzip")
				f.create_dataset(name="out_pts1", data=out_ptss1, compression="gzip")
				f.create_dataset(name="out_pts2", data=out_ptss2, compression="gzip")
				f.create_dataset(name="out_pts31", data=out_ptss31, compression="gzip")
				f.create_dataset(name="out_pts32", data=out_ptss32, compression="gzip")
				f.create_dataset(name="out_para12_r", data=out_paras12_r, compression="gzip")
				f.create_dataset(name="out_para12_t", data=out_paras12_t, compression="gzip")
				f.create_dataset(name="out_para21_r", data=out_paras21_r, compression="gzip")
				f.create_dataset(name="out_para21_t", data=out_paras21_t, compression="gzip")

if __name__ == '__main__':
	args = parse_args()
	main(args)
