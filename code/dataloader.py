import numpy as np
import warnings
import os
from torch.utils.data import Dataset
import trans
import h5py
warnings.filterwarnings('ignore')

def pc_normalize(pc):
	pmax = np.max(pc, axis=0)
	pmin = np.min(pc, axis=0)
	centroid = (pmax + pmin) / 2.0
	return - centroid

class DataLoader(Dataset):
	def __init__(self, root, npoint=2048, split='train', category=['02691156']):
		self.npoints = npoint

		in_pts1 = np.zeros(shape=(0, 2048, 3))
		in_pts2 = np.zeros(shape=(0, 2048, 3))
		gt_pts1 = np.zeros(shape=(0, 2048, 3))
		gt_pts2 = np.zeros(shape=(0, 2048, 3))
		gt_pts3 = np.zeros(shape=(0, 2048, 3))
		pts_name = np.zeros(shape=(0))
		for cate in category:
			with h5py.File(os.path.join(root, split+'_'+cate+'.h5'), 'r') as f:
				in_pts1 = np.concatenate((in_pts1, np.array(f['in1'])[:,:,:3]), axis=0).astype(np.float32)
				in_pts2 = np.concatenate((in_pts2, np.array(f['in2'])[:,:,:3]), axis=0).astype(np.float32)
				gt_pts1 = np.concatenate((gt_pts1, np.array(f['gt1'])[:,:,:3]), axis=0).astype(np.float32)
				gt_pts2 = np.concatenate((gt_pts2, np.array(f['gt2'])[:,:,:3]), axis=0).astype(np.float32)
				gt_pts3 = np.concatenate((gt_pts3, np.array(f['gt12'])[:,:,:3]), axis=0).astype(np.float32)
				pts_name = np.concatenate((pts_name, np.array(f['name'])[:]), axis=0)
			print(os.path.join(root, split+'_'+cate+'.h5'), ' LOADED!')
		in_pts1[:,:,[1,2]] = in_pts1[:,:,[2,1]]
		in_pts2[:,:,[1,2]] = in_pts2[:,:,[2,1]]
		gt_pts1[:,:,[1,2]] = gt_pts1[:,:,[2,1]]
		gt_pts2[:,:,[1,2]] = gt_pts2[:,:,[2,1]]
		gt_pts3[:,:,[1,2]] = gt_pts3[:,:,[2,1]]
		self.in_ptss1 = np.array(in_pts1)
		self.in_ptss2 = np.array(in_pts2)
		self.gt_ptss1 = np.array(gt_pts1)
		self.gt_ptss2 = np.array(gt_pts2)
		self.gt_ptss3 = np.array(gt_pts3)
		self.ptss_name = np.array(pts_name)
		print('The size of %s data is %d'%(split, len(self.ptss_name)))

	def __getitem__(self, index):
		in_pts1 = self.in_ptss1[index,:int(self.npoints),:3].copy()
		in_pts2 = self.in_ptss2[index,:int(self.npoints),:3].copy()
		gt_pts1 = self.gt_ptss1[index,:int(self.npoints),:3].copy()
		gt_pts2 = self.gt_ptss2[index,:int(self.npoints),:3].copy()
		gt_pts3 = self.gt_ptss3[index,:int(self.npoints),:3].copy()

		np.random.seed()
		angle1 = np.pi / 2 * np.power(np.random.uniform(-1, 1), 3)
		angle2 = np.pi / 2 * np.power(np.random.uniform(-1, 1), 3)
		theta1 = np.random.uniform(0, np.pi * 2)
		phi1 = np.random.uniform(0, np.pi / 2)
		x1 = np.cos(theta1) * np.sin(phi1)
		y1 = np.sin(theta1) * np.sin(phi1)
		z1 = np.cos(phi1)
		axis1 = np.array([x1, y1, z1])
		theta2 = np.random.uniform(0, np.pi * 2)
		phi2 = np.random.uniform(0, np.pi / 2)
		x2 = np.cos(theta2) * np.sin(phi2)
		y2 = np.sin(theta2) * np.sin(phi2)
		z2 = np.cos(phi2)
		axis2 = np.array([x2, y2, z2])

		trans1 = pc_normalize(in_pts1)
		trans2 = pc_normalize(in_pts2)
		centerpoint1 = - trans1
		centerpoint2 = - trans2
		quater1 = trans.axisangle2quaternion(axis=axis1, angle=angle1)
		quater2 = trans.axisangle2quaternion(axis=axis2, angle=angle2)
		matrix1_t = trans.translation2matrix(trans1)
		matrix2_t = trans.translation2matrix(trans2)
		matrix1_r = trans.quaternion2matrix(quater1)
		matrix2_r = trans.quaternion2matrix(quater2)
		matrix1 = np.matmul(matrix1_r, matrix1_t)
		matrix2 = np.matmul(matrix2_r, matrix2_t)

		in_pts1 = trans.transform_pts(in_pts1, matrix1)
		gt_pts1 = trans.transform_pts(gt_pts1, matrix1)
		gt_pts31 = trans.transform_pts(gt_pts3, matrix1)
		centerpoint2 = trans.transform_pts(centerpoint2, matrix1)
		in_pts2 = trans.transform_pts(in_pts2, matrix2)
		gt_pts2 = trans.transform_pts(gt_pts2, matrix2)
		gt_pts32 = trans.transform_pts(gt_pts3, matrix2)
		centerpoint1 = trans.transform_pts(centerpoint1, matrix2)

		gt_para12_r = trans.quat_multiply(quater2, trans.quaternion_inv(quater1))
		gt_para12_t = centerpoint1[0]
		gt_para21_r = trans.quat_multiply(quater1, trans.quaternion_inv(quater2))
		gt_para21_t = centerpoint2[0]
		gt_para_canonical_1 = trans.quaternion_inv(quater1)
		gt_para_canonical_2 = trans.quaternion_inv(quater2)

		return in_pts1, in_pts2, gt_pts1, gt_pts2, gt_pts31, gt_pts32, gt_para12_r, gt_para12_t, gt_para21_r, gt_para21_t, gt_para_canonical_1, gt_para_canonical_2

	def __len__(self):
		return len(self.ptss_name)

	def get_name(self, index):
		return self.ptss_name[index].decode('utf-8')
