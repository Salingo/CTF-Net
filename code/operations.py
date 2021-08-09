import torch
import numpy as np
import sampling

class GatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx):
        r"""
        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor
        idx : torch.Tensor
            (B, npoint) tensor of the features to gather
        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        features = features.contiguous()
        idx = idx.contiguous()
        idx = idx.to(dtype=torch.int32)

        B, npoint = idx.size()
        _, C, N = features.size()

        output = torch.empty(
            B, C, npoint, dtype=features.dtype, device=features.device)
        output = sampling.gather_forward(
            B, C, N, npoint, features, idx, output
        )

        ctx.save_for_backward(idx)
        ctx.C = C
        ctx.N = N
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, = ctx.saved_tensors
        B, npoint = idx.size()

        grad_features = torch.zeros(
            B, ctx.C, ctx.N, dtype=grad_out.dtype, device=grad_out.device)
        grad_features = sampling.gather_backward(
            B, ctx.C, ctx.N, npoint, grad_out.contiguous(), idx, grad_features
        )

        return grad_features, None

gather_points = GatherFunction.apply

class FurthestPointSampling(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.LongTensor
            (B, npoint) tensor containing the indices

        """
        B, N, _ = xyz.size()

        idx = torch.empty([B, npoint], dtype=torch.int32, device=xyz.device)
        temp = torch.full([B, N], 1e10, dtype=torch.float32, device=xyz.device)

        sampling.furthest_sampling(
            B, N, npoint, xyz, temp, idx
        )
        ctx.mark_non_differentiable(idx)
        return idx

__furthest_point_sample = FurthestPointSampling.apply

def furthest_point_sample(xyz, npoint, NCHW=False):
    """
    :param
        xyz (B, C, N) or (B, N, C)
        npoint a constant
    :return
        torch.LongTensor
            (B, npoint) tensor containing the indices
        torch.FloatTensor
            (B, npoint, C) or (B, C, npoint) point sets"""
    # assert(xyz.dim() == 3), "input for furthest sampling must be a 3D-tensor, but xyz.size() is {}".format(xyz.size())
    # need transpose
    if NCHW:
        xyz = xyz.transpose(2, 1).contiguous()

    # assert(xyz.size(2) == 3), "furthest sampling is implemented for 3D points"
    idx = __furthest_point_sample(xyz[:, :, :3].contiguous(), npoint)
    sampled_pc = gather_points(xyz.transpose(2, 1).contiguous(), idx)
    if not NCHW:
        sampled_pc = sampled_pc.transpose(2, 1).contiguous()
    return sampled_pc
    # return idx, sampled_pc

# Verification
if __name__ == "__main__":
    import h5py 
    
    nchannel = 6
    pts_in = np.zeros(shape=(0, 2048, nchannel))
    with h5py.File('../../data/shapenet_color/test_03001627.h5', 'r') as f:
        pts_in = np.concatenate((pts_in, np.array(f['in1'])), axis=0).astype(np.float32)
    pts_in[:,:,[1,2]] = pts_in[:,:,[2,1]]
    pts_in = pts_in[-5:]
    pts_in_torch = torch.tensor(pts_in).cuda()
    _, pts_list = furthest_point_sample(pts_in_torch, 1024)
    pts_list = pts_list.cpu()
    for i in range(5):
        np.savetxt(str(i) + '.xyz', pts_list[i])
        np.savetxt('origin' + str(i) +'.xyz', pts_in[i])
            