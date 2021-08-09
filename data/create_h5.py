import os
import numpy as np
import h5py

def create_h5(cate, lists, path_in, path_out):
	points_in1 = []
	points_in2 = []
	points_gt1 = []
	points_gt2 = []
	points_gt12 = []
	names = []
	for i in range(len(lists)):
		if lists[i].split('/')[0] == cate:
			for n in range(10):
				point_in1 = np.loadtxt(path_in+lists[i].split('/')[1].rstrip()+'_in'+str(n)+'_p0.pts', delimiter=' ')
				point_in2 = np.loadtxt(path_in+lists[i].split('/')[1].rstrip()+'_in'+str(n)+'_p1.pts', delimiter=' ')
				point_gt1 = np.loadtxt(path_in+lists[i].split('/')[1].rstrip()+'_gt'+str(n)+'_c0.pts', delimiter=' ')
				point_gt2 = np.loadtxt(path_in+lists[i].split('/')[1].rstrip()+'_gt'+str(n)+'_c1.pts', delimiter=' ')
				point_gt12 = np.loadtxt(path_in+lists[i].split('/')[1].rstrip()+'_gt'+str(n)+'_12.pts', delimiter=' ')
				points_in1.append(point_in1)
				points_in2.append(point_in2)
				points_gt1.append(point_gt1)
				points_gt2.append(point_gt2)
				points_gt12.append(point_gt12)
				names.append(lists[i].split('/')[1].rstrip()+'_'+str(n))
	with h5py.File(path_out+'.h5', 'w') as f:
		f.create_dataset(name="in1", data=np.array(points_in1).astype(np.float32), compression="gzip")
		f.create_dataset(name="in2", data=np.array(points_in2).astype(np.float32), compression="gzip")
		f.create_dataset(name="gt1", data=np.array(points_gt1).astype(np.float32), compression="gzip")
		f.create_dataset(name="gt2", data=np.array(points_gt2).astype(np.float32), compression="gzip")
		f.create_dataset(name="gt12", data=np.array(points_gt12).astype(np.float32), compression="gzip")
		f.create_dataset(name="name", data=np.array(names,'S'), compression="gzip")


if __name__ == '__main__':
	CATEGORIES = ['02691156','02933112','02958343','03001627','03636649','04256520','04379243','04530566']
	POINTS_PATH = './dataset/'
	train_list = []
	val_list = []
	test_list = []
	with open('./train_list.txt','r') as f:
		train_list = f.readlines()
	with open('./val_list.txt','r') as f:
		val_list = f.readlines()
	with open('./test_list.txt','r') as f:
		test_list = f.readlines()

	for cate in CATEGORIES:
		print(cate+' train \n')
		create_h5(cate, train_list, POINTS_PATH+CATE+'/train/', './train_'+CATE)
		print(cate+' test \n')
		create_h5(cate, test_list, POINTS_PATH+CATE+'/test/', './test_'+CATE)
		print(cate+' val \n')
		create_h5(cate, val_list, POINTS_PATH+CATE+'/val/', './val_'+CATE)
	