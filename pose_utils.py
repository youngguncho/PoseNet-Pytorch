import torch
import numpy as np


def quat_to_euler(q, is_degree=False):
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.rad2deg(np.arctan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.rad2deg(np.arcsin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.rad2deg(np.arctan2(t3, t4))

    if is_degree:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)

    return [roll, pitch, yaw]


def array_dist(pred, target):
    return np.linalg.norm(pred - target, 2)


def position_dist(pred, target):
    return np.linalg.norm(pred-target, 2)


def rotation_dist(pred, target):
    pred = quat_to_euler(pred)
    target = quat_to_euler(target)

    return np.linalg.norm(pred-target, 2)


def fit_gaussian(pose_quat):
    pose_quat = pose_quat.detach().cpu().numpy()

    num_data, _ = pose_quat.shape

    # Convert quat to euler
    pose_euler = []
    for i in range(0, num_data):
        pose = pose_quat[i, :3]
        quat = pose_quat[i, 3:]
        euler = quat_to_euler(quat)
        pose_euler.append(np.concatenate((pose, euler)))

    # Calculate mean and variance
    pose_mean = np.mean(pose_euler, axis=0)
    mat_var = np.zeros((6, 6))
    for i in range(0, num_data):
        pose_diff = pose_euler[i] - pose_mean
        mat_var += pose_diff * np.transpose(pose_diff)

    mat_var = mat_var / num_data
    pose_var = mat_var.diagonal()
    print(pose_var)

    return pose_mean, pose_var








