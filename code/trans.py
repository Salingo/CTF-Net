import numpy as np
import torch

def transform_pts(points, transform):
    # batch transform
    if len(transform.shape) == 3:
        rot = transform[:, :3, :3]
        trans = transform[:, :3, 3]
    # single transform
    else:
        rot = transform[:3, :3]
        trans = transform[:3, 3]
    point = np.matmul(points, np.transpose(rot)) + np.expand_dims(trans, axis=-2)
    return point

def transform_pts_torch(points, transform):
    # batch transform
    if len(transform.shape) == 3:
        rot = transform[:, :3, :3]
        trans = transform[:, :3, 3]
    # single transform
    else:
        rot = transform[:3, :3]
        trans = transform[:3, 3]
    point = torch.matmul(points, torch.transpose(rot,-2,-1)) + torch.unsqueeze(trans, -2)
    return point

def translation2matrix(trans):
    T = np.eye(4, dtype=np.float32)
    T[0,3] = trans[0]
    T[1,3] = trans[1]
    T[2,3] = trans[2]
    return T

def translation2matrix_torch(trans):
    batch, _ = trans.shape
    T = torch.eye(4)
    T = T.reshape((1, 4, 4))
    T = T.repeat(batch, 1, 1)
    T = T.cuda()
    T[:,0,3] = trans[:,0]
    T[:,1,3] = trans[:,1]
    T[:,2,3] = trans[:,2]
    return T

def quaternion2matrix(quaternion):
    x, y, z, w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x
    R = np.array([[1.0 - yy2 - zz2,       xy2 - wz2,       zx2 + wy2, 0.0],
                  [      xy2 + wz2, 1.0 - xx2 - zz2,       yz2 - wx2, 0.0],
                  [      zx2 - wy2,       yz2 + wx2, 1.0 - xx2 - yy2, 0.0],
                  [            0.0,             0.0,             0.0, 1.0]])
    return R

def quaternion2matrix_torch(quaternion):
    batch, _ = quaternion.shape
    x, y, z, w = quaternion[:,0], quaternion[:,1], quaternion[:,2], quaternion[:,3]
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x

    R = torch.eye(4)
    R = R.reshape((1, 4, 4))
    R = R.repeat(batch, 1, 1)
    R = R.cuda()
    R[:,0,0] = 1.0 - yy2 - zz2
    R[:,0,1] = xy2 - wz2
    R[:,0,2] = zx2 + wy2
    R[:,1,0] = xy2 + wz2
    R[:,1,1] = 1.0 - xx2 - zz2
    R[:,1,2] = yz2 - wx2
    R[:,2,0] = zx2 - wy2
    R[:,2,1] = yz2 + wx2
    R[:,2,2] = 1.0 - xx2 - yy2
    return R

def quat2mat3x3(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def quat_multiply(q1, q2):
    x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
    x2, y2, z2, w2 = np.split(q2, 4, axis=-1)
    return np.concatenate([x1*w2 + y1*z2 - z1*y2 + w1*x2,
                          -x1*z2 + y1*w2 + z1*x2 + w1*y2,
                           x1*y2 - y1*x2 + z1*w2 + w1*z2,
                          -x1*x2 - y1*y2 - z1*z2 + w1*w2], axis=-1)

def quat_multiply_torch(q1, q2):
    x1, y1, z1, w1 = torch.split(q1, 1, dim=-1)
    x2, y2, z2, w2 = torch.split(q2, 1, dim=-1)
    return torch.cat([x1*w2 + y1*z2 - z1*y2 + w1*x2,
                     -x1*z2 + y1*w2 + z1*x2 + w1*y2,
                      x1*y2 - y1*x2 + z1*w2 + w1*z2,
                     -x1*x2 - y1*y2 - z1*z2 + w1*w2], dim=-1)

def quaternion_inv(q):
    Q = np.zeros(4)
    Q[:3]= q[:3]
    Q[3]= -q[3]
    return Q

def quaternion_inv_torch(q):
    Q = torch.zeros_like(q)
    Q[:,:3]= q[:,:3]
    Q[:,3]= -q[:,3]
    return Q

def wrap_angle(theta):
    result = ((theta + np.pi) % (2 * np.pi)) - np.pi
    if result == -np.pi:
        result = np.pi
    return result

def quaternion2axisangle(q):
    tolerance = 1e-17
    norm = np.linalg.norm(q[:3])
    if norm < tolerance:
        # Here there are an infinite set of possible axes, use what has been specified as an undefined axis.
        axis = np.array([0,0,0])
    else:
        axis = q[:3] / norm
    angle = wrap_angle(2.0 * np.arctan2(norm, q[3]))
    return (axis, angle)

def axisangle2quaternion(axis, angle):
    mag_sq = np.dot(axis, axis)
    if mag_sq == 0.0:
        raise ZeroDivisionError("Provided rotation axis has no length")
    axis = axis / np.sqrt(mag_sq)
    angle = angle / 2.0
    r = np.cos(angle)
    i = axis * np.sin(angle)
    return np.array([i[0], i[1], i[2], r])


def matrix2quaternion(M, isprecise=False):
    q = np.empty((4, ))
    i, j, k = 0, 1, 2
    if M[1, 1] > M[0, 0]:
        i, j, k = 1, 2, 0
    if M[2, 2] > M[i, i]:
        i, j, k = 2, 0, 1
    t = M[i, i] - (M[j, j] + M[k, k]) + 1
    q[i] = t
    q[j] = M[i, j] + M[j, i]
    q[k] = M[k, i] + M[i, k]
    q[3] = M[k, j] - M[j, k]
    q = q[[3, 0, 1, 2]]
    
    if q[0] < 0.0:
        np.negative(q, q)
    xyzw_q = np.array([q[1], q[2], q[3], q[0]])
    xyzw_q = xyzw_q / np.linalg.norm(xyzw_q)
    return xyzw_q

if __name__ == '__main__':
    q = np.array([-0.420084, -0.560112, 0.70014, 0.140028])
    mat = quaternion2matrix(q)
    new_q = matrix2quaternion(mat)
    print(new_q)