import os
import numpy as np
import h5py

def pc_normalize(pc):
	#norm by bbox
	max = np.max(pc,axis=0)
	min = np.min(pc,axis=0)
	centroid = max + min
	pc = pc - centroid / 2.0
	m = np.max(abs(pc)) * 2.0
	pc = pc / m
	return pc

def generate_partials(pc, IN_PC_SIZE, MIN_DISTANCE):
	dist = 0
	pc1 = np.zeros((1))
	pc2 = np.zeros((1))
	while(dist < MIN_DISTANCE):
		np.random.seed()
		theta = 2 * np.pi * np.random.random()
		a = 2 * np.random.random() - 1
		r = 0.75
		phi = np.arccos(a)
		x = r * np.cos(theta) * np.sin(phi)
		y = r * np.sin(theta) * np.sin(phi)
		z = r * np.cos(phi)
		center1 = np.array([x, y, z])
		radius1 = 0.3 + np.random.random()

		theta = 2 * np.pi * np.random.random()
		a = 2 * np.random.random() - 1
		r = 0.75
		phi = np.arccos(a)
		x = r * np.cos(theta) * np.sin(phi)
		y = r * np.sin(theta) * np.sin(phi)
		z = r * np.cos(phi)
		center2 = np.array([x, y, z])
		radius2 = 0.3 + np.random.random()
		
		center_points = pc[:,:3] - np.repeat(np.reshape(center1,(1,3)), IN_PC_SIZE, axis=0)
		valid_idx1 = np.where(np.linalg.norm(center_points, axis=1) < radius1)
		center_points = pc[:,:3] - np.repeat(np.reshape(center2,(1,3)), IN_PC_SIZE, axis=0)
		valid_idx2 = np.where(np.linalg.norm(center_points, axis=1) < radius2)

		idxall = [i for i in range(IN_PC_SIZE)]
		idxall = np.array(idxall)
		remain_idx1 = np.setdiff1d(idxall, valid_idx1)
		remain_idx2 = np.setdiff1d(idxall, valid_idx2)
		remain_idx12 = np.setdiff1d(idxall, np.union1d(valid_idx1, valid_idx2))
		pc1 = pc[valid_idx1]
		pc2 = pc[valid_idx2]
		pc_remain1 = pc[remain_idx1]
		pc_remain2 = pc[remain_idx2]
		pc_remain12 = pc[remain_idx12]
		centroid1 = np.mean(pc1[:,:3], axis=0)
		centroid2 = np.mean(pc2[:,:3], axis=0)
		dist = np.linalg.norm(centroid1 - centroid2)
	return [pc1, pc2, pc_remain1, pc_remain2, pc_remain12]
	
def generate_data(PATH_LOAD, PATH_SAVE, IN_PC_SIZE, OUT_PC_SIZE, MIN_DISTANCE, NUM_SAMPLES):
	if not os.path.exists(PATH_SAVE):
		os.makedirs(PATH_SAVE)
	files = os.listdir(PATH_LOAD)
	for i in range(int(len(files))):
		print(i,' ',files[i])
		pc = np.zeros((1))
		with h5py.File(os.path.join(PATH_LOAD+files[i]), 'r') as f:
			pc = np.array(f['data'])
		pc[:,:3] = pc_normalize(pc[:,:3]) # normalize corrdinates
		pc[:,3:] = pc[:,3:] / 255.0 # normalize colors
		FLAG_WRONG = False
		for n in range(NUM_SAMPLES):
			points = generate_partials(pc, IN_PC_SIZE, MIN_DISTANCE)
			count = 0
			while(points[0].shape[0] < 4096 or points[1].shape[0] < 4096 or points[2].shape[0] < OUT_PC_SIZE or points[3].shape[0] < OUT_PC_SIZE or points[4].shape[0] < OUT_PC_SIZE):
				points = generate_partials(pc, IN_PC_SIZE, MIN_DISTANCE)
				count += 1
				if count > 10000:
					with open('failed.txt', 'a') as f:
						f.writelines(CATES+'/'+files[i]+' is failed\n')
					FLAG_WRONG = True
					break
			if FLAG_WRONG:
				break
			np.savetxt(PATH_SAVE+files[i].split('.')[0]+'_in'+str(n)+'_p0.pts', points[0][:OUT_PC_SIZE], fmt='%.8f')
			np.savetxt(PATH_SAVE+files[i].split('.')[0]+'_in'+str(n)+'_p1.pts', points[1][:OUT_PC_SIZE], fmt='%.8f')
			np.savetxt(PATH_SAVE+files[i].split('.')[0]+'_gt'+str(n)+'_c0.pts', points[2][:OUT_PC_SIZE], fmt='%.8f')
			np.savetxt(PATH_SAVE+files[i].split('.')[0]+'_gt'+str(n)+'_c1.pts', points[3][:OUT_PC_SIZE], fmt='%.8f')
			np.savetxt(PATH_SAVE+files[i].split('.')[0]+'_gt'+str(n)+'_12.pts', points[4][:OUT_PC_SIZE], fmt='%.8f')

if __name__ == '__main__':
	#CATEGORIES: '02691156','02933112','02958343','03001627','03636649','04256520','04379243','04530566'
	CATE = '02691156'
	IN_PC_SIZE = 16384
	OUT_PC_SIZE = 2048
	MIN_DISTANCE = 0.3
	NUM_SAMPLES = 10

	PATH_LOAD = "./virtualscan/"+CATE+"/train/" # complete point cloud data from virtual scanning, each contains 16384 points
	PATH_SAVE = "./dataset/"+CATE+"/train/" 
	generate_data(PATH_LOAD, PATH_SAVE, IN_PC_SIZE, OUT_PC_SIZE, MIN_DISTANCE, NUM_SAMPLES)

	PATH_LOAD = "./virtualscan/"+CATE+"/val/" # complete point cloud data from virtual scanning, each contains 16384 points
	PATH_SAVE = "./dataset/"+CATE+"/val/" 
	generate_data(PATH_LOAD, PATH_SAVE, IN_PC_SIZE, OUT_PC_SIZE, MIN_DISTANCE, NUM_SAMPLES)

	PATH_LOAD = "./virtualscan/"+CATE+"/test/" # complete point cloud data from virtual scanning, each contains 16384 points
	PATH_SAVE = "./dataset/"+CATE+"/test/" 
	generate_data(PATH_LOAD, PATH_SAVE, IN_PC_SIZE, OUT_PC_SIZE, MIN_DISTANCE, NUM_SAMPLES)
