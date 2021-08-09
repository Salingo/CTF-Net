import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trans
from pointnet_util import PointNetEncoder

class Convlayer(nn.Module):
	def __init__(self, point_scales, nchannel=3):
		super(Convlayer, self).__init__()
		self.point_scales = point_scales
		self.conv1 = torch.nn.Conv2d(1, 64, (1, nchannel))
		self.conv2 = torch.nn.Conv2d(64, 64, 1)
		self.conv3 = torch.nn.Conv2d(64, 128, 1)
		self.conv4 = torch.nn.Conv2d(128, 256, 1)
		self.conv5 = torch.nn.Conv2d(256, 512, 1)
		self.conv6 = torch.nn.Conv2d(512, 1024, 1)
		self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.bn5 = nn.BatchNorm2d(512)
		self.bn6 = nn.BatchNorm2d(1024)
	def forward(self, x):
		x = torch.unsqueeze(x, 1)
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x_128 = F.relu(self.bn3(self.conv3(x)))
		x_256 = F.relu(self.bn4(self.conv4(x_128)))
		x_512 = F.relu(self.bn5(self.conv5(x_256)))
		x_1024 = F.relu(self.bn6(self.conv6(x_512)))
		x_128 = torch.squeeze(self.maxpool(x_128), 2)
		x_256 = torch.squeeze(self.maxpool(x_256), 2)
		x_512 = torch.squeeze(self.maxpool(x_512), 2)
		x_1024 = torch.squeeze(self.maxpool(x_1024), 2)
		L = [x_1024, x_512, x_256, x_128]
		x = torch.cat(L, 1)
		return x

class STN3d(nn.Module):
	def __init__(self, nchannel=3):
		super(STN3d, self).__init__()
		self.conv1 = torch.nn.Conv1d(nchannel, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)
		self.conv3 = torch.nn.Conv1d(128, 1024, 1)
		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 9)
		self.relu = nn.ReLU()
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(128)
		self.bn3 = nn.BatchNorm1d(1024)
		self.bn4 = nn.BatchNorm1d(512)
		self.bn5 = nn.BatchNorm1d(256)
	def forward(self, x):
		x = x.transpose(2, 1)
		batchsize = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, 1024)
		x = F.relu(self.bn4(self.fc1(x)))
		x = F.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)
		iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden
		x = x.view(-1, 3, 3)
		return x

class Latentfeature(nn.Module):
	def __init__(self, num_resolution, input_scale_list, nchannel=3, use_stn=False):
		super(Latentfeature, self).__init__()
		self.num_resolution = num_resolution
		self.input_scale_list = input_scale_list
		self.num_channels = nchannel
		self.use_stn = use_stn
		if use_stn:
			self.stn = STN3d(nchannel)
		self.Convlayers = nn.ModuleList([Convlayer(point_scales=self.input_scale_list[i], nchannel=nchannel)for i in range(self.num_resolution)])
		self.conv1 = torch.nn.Conv1d(num_resolution, 1, 1)
		self.bn1 = nn.BatchNorm1d(1)
	def forward(self, x):
		outs = []
		for i in range(self.num_resolution):
			if self.use_stn:
				trans = self.stn(x[i])
				if self.num_channels > 3:
					x[i], feature = x[i].split(3, dim=2)
				x[i] = torch.bmm(x[i], trans)
				if self.num_channels > 3:
					x[i] = torch.cat([x[i], feature], dim=2)
			outs.append(self.Convlayers[i](x[i]))
		latentfeature = torch.cat(outs, 2)
		latentfeature = latentfeature.transpose(1, 2)
		latentfeature = F.relu(self.bn1(self.conv1(latentfeature)))
		latentfeature = torch.squeeze(latentfeature, 1)
		return latentfeature

class comp_model(nn.Module):
	def  __init__(self, input_scale_list, output_scale_list, nchannel=3):
		super(comp_model, self).__init__()
		num_resolution = 3
		self.output_scale_list = output_scale_list
		self.latentfeature = Latentfeature(num_resolution, input_scale_list, nchannel)
		self.fc_fine_1 = nn.Linear(1920, 1024)
		self.fc_mid_1 = nn.Linear(1024, 512)
		self.fc_low_1 = nn.Linear(512, 256)
		self.fc_fine_2 = nn.Linear(1024, self.output_scale_list[2] * 16)
		self.fc_mid_2 = nn.Linear(512, self.output_scale_list[1] * 16)
		self.fc_low_2 = nn.Linear(256, self.output_scale_list[0] * 3)
		self.conv_mid_1 = torch.nn.Conv1d(int(self.output_scale_list[1] / self.output_scale_list[0] * 16), int(self.output_scale_list[1] / self.output_scale_list[0] * 3), 1)
		self.conv_fine_1 = torch.nn.Conv1d(int(self.output_scale_list[2] / self.output_scale_list[1] * 16), int(self.output_scale_list[2] / self.output_scale_list[1] * 16), 1)
		self.conv_fine_2 = torch.nn.Conv1d(int(self.output_scale_list[2] / self.output_scale_list[1] * 16), int(self.output_scale_list[2] / self.output_scale_list[1] * 8), 1)
		self.conv_fine_3 = torch.nn.Conv1d(int(self.output_scale_list[2] / self.output_scale_list[1] * 8), int(self.output_scale_list[2] / self.output_scale_list[1] * 3), 1)
	def forward(self, x1, x2, x3):
		x = self.latentfeature([x1,x2,x3])
		x_fine = F.relu(self.fc_fine_1(x)) #1024
		x_mid = F.relu(self.fc_mid_1(x_fine)) #512
		x_low = F.relu(self.fc_low_1(x_mid))  #256
		pc_low_feat = self.fc_low_2(x_low)
		pc_low_xyz = pc_low_feat.reshape(-1, self.output_scale_list[0], 3) # scale_0 x 3 (output_0)
		pc_mid_feat = F.relu(self.fc_mid_2(x_mid))
		pc_mid_feat = pc_mid_feat.reshape(-1, int(self.output_scale_list[1] / self.output_scale_list[0] * 16), self.output_scale_list[0])
		pc_mid_xyz = self.conv_mid_1(pc_mid_feat) # ((scale_1 / scale_0) * 3) x scale_0
		pc_fine_feat = F.relu(self.fc_fine_2(x_fine))
		pc_fine_feat = pc_fine_feat.reshape(-1, int(self.output_scale_list[2] / self.output_scale_list[1] * 16), self.output_scale_list[1])
		pc_fine_feat = F.relu(self.conv_fine_1(pc_fine_feat))
		pc_fine_feat = F.relu(self.conv_fine_2(pc_fine_feat))
		pc_fine_xyz = self.conv_fine_3(pc_fine_feat) # ((scale_2 / scale_1) * 3) x scale_1
		pc_low_xyz_expand = torch.unsqueeze(pc_low_xyz, 2)
		pc_mid_xyz = pc_mid_xyz.transpose(1, 2)
		pc_mid_xyz = pc_mid_xyz.reshape(-1, self.output_scale_list[0], int(self.output_scale_list[1] / self.output_scale_list[0]), 3)
		pc_mid_xyz = pc_low_xyz_expand + pc_mid_xyz
		pc_mid_xyz = pc_mid_xyz.reshape(-1, self.output_scale_list[1], 3) # scale_1 x 3 (output_1)
		pc_mid_xyz_expand = torch.unsqueeze(pc_mid_xyz, 2)
		pc_fine_xyz = pc_fine_xyz.transpose(1, 2)
		pc_fine_xyz = pc_fine_xyz.reshape(-1, self.output_scale_list[1], int(self.output_scale_list[2] / self.output_scale_list[1]), 3)
		pc_fine_xyz = pc_mid_xyz_expand + pc_fine_xyz
		pc_fine_xyz = pc_fine_xyz.reshape(-1, self.output_scale_list[2], 3) # scale_2 x 3 (output_2)
		return pc_low_xyz, pc_mid_xyz, pc_fine_xyz 

class orient_model(nn.Module):
	def __init__(self):
		super(orient_model, self).__init__()
		self.encoder = PointNetEncoder(global_feat=True, feature_transform=False, channel=3)
		self.fc0 = nn.Linear(1024, 1024)
		self.bn0 = nn.BatchNorm1d(1024)
		self.fc1 = nn.Linear(1024, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.drop1 = nn.Dropout(0.4)
		self.fc2 = nn.Linear(512, 128)
		self.bn2 = nn.BatchNorm1d(128)
		self.drop2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(128, 4) # para_rotation
	def forward(self, xyz):
		xr = self.encoder(xyz.transpose(2, 1))
		xr = F.relu(self.bn0(self.fc0(xr)))
		xr = self.drop1(F.relu(self.bn1(self.fc1(xr))))
		xr = self.drop2(F.relu(self.bn2(self.fc2(xr))))
		para_r = F.tanh(self.fc3(xr))
		R = trans.quaternion2matrix_torch(para_r)
		return para_r

class regi_model(nn.Module):
	def __init__(self):
		super(regi_model, self).__init__()
		self.encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
		self.fc0 = nn.Linear(1024, 512)
		self.bn0 = nn.BatchNorm1d(512)
		self.fc1 = nn.Linear(1024, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.drop1 = nn.Dropout(0.4)
		self.fc2 = nn.Linear(512, 128)
		self.bn2 = nn.BatchNorm1d(128)
		self.drop2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(128, 4) # para_rotation
		self.fc4 = nn.Linear(1024, 512)
		self.bn4 = nn.BatchNorm1d(512)
		self.drop4 = nn.Dropout(0.4)
		self.fc5 = nn.Linear(512, 128)
		self.bn5 = nn.BatchNorm1d(128)
		self.drop5 = nn.Dropout(0.5)
		self.fc6 = nn.Linear(128, 3) #para_translation
	def forward(self, xyz1, xyz2):
		fea1 = self.encoder(xyz1.transpose(2, 1))
		fea1 = F.relu(self.bn0(self.fc0(fea1)))
		fea2 = self.encoder(xyz2.transpose(2, 1))
		fea2 = F.relu(self.bn0(self.fc0(fea2)))
		xr = torch.cat((fea1, fea2), 1)
		xr = self.drop1(F.relu(self.bn1(self.fc1(xr))))
		xr = self.drop2(F.relu(self.bn2(self.fc2(xr))))
		para_r21 = F.tanh(self.fc3(xr))
		R = trans.quaternion2matrix_torch(para_r21)
		transformed_xyz2 = trans.transform_pts_torch(xyz2, R)
		fea3 = self.encoder(transformed_xyz2.transpose(2, 1))
		fea3 = F.relu(self.bn0(self.fc0(fea3)))
		xt = torch.cat((fea1, fea3), 1)
		xt = self.drop4(F.relu(self.bn4(self.fc4(xt))))
		xt = self.drop5(F.relu(self.bn5(self.fc5(xt))))
		para_t21 = F.tanh(self.fc6(xt))
		T = trans.translation2matrix_torch(para_t21)
		transformed_xyz2 = trans.transform_pts_torch(transformed_xyz2, T)
		return para_r21, para_t21, transformed_xyz2
