# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import os, sys, pickle, math
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from load_data import load_kitti_lidar_data, load_kitti_poses, load_kitti_calib
from utils import *

import logging
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

import open3d as o3d

class map_t:
    def __init__(s, resolution=0.5):
        s.resolution = resolution
        s.xmin, s.xmax = -700, 700
        s.zmin, s.zmax = -500, 900
        # s.xmin, s.xmax = -400, 1100
        # s.zmin, s.zmax = -300, 1200

        s.szx = int(np.ceil((s.xmax - s.xmin) / s.resolution + 1))
        s.szz = int(np.ceil((s.zmax - s.zmin) / s.resolution + 1))

        # binarized map and log-odds
        s.cells = np.zeros((s.szx, s.szz), dtype=np.int8)
        s.log_odds = np.zeros(s.cells.shape, dtype=np.float64)

        # value above which we are not going to increase the log-odds,
        # and similarly we will not decrease log-odds of a cell below -max
        s.log_odds_max = 5e6
        # number of observations received for each cell
        s.num_obs_per_cell = np.zeros(s.cells.shape, dtype=np.uint64)

        # we call a cell occupied if the probability of
        # occupancy P(m_i | ... ) is >= occupied_prob_thresh
        s.occupied_prob_thresh = 0.6
        s.log_odds_thresh = np.log(s.occupied_prob_thresh / (1 - s.occupied_prob_thresh))

    def grid_cell_from_xz(s, x, z):
        """
        x and z can be 1-dimensional arrays, compute the cell indices in the map corresponding
        to these (x,y) locations. You should return an array of shape 2 x len(x). Be
        careful to handle instances when x/z go outside the map bounds, you can use
        np.clip to handle these situations.
        """
        x = np.atleast_1d(x)
        z = np.atleast_1d(z)
        
        idx_x = np.floor((x - s.xmin) / s.resolution).astype(int)
        idx_z = np.floor((z - s.zmin) / s.resolution).astype(int)
        
        idx_x = np.clip(idx_x, 0, s.szx - 1)
        idx_z = np.clip(idx_z, 0, s.szz - 1)
        
        indices = np.vstack((idx_x, idx_z))
        return indices
    
class slam_t:
    """
    s is the same as s. In Python it does not really matter
    what we call s, s is shorter. As a general comment, (I believe)
    you will have fewer bugs while writing scientific code if you
    use the same/similar variable names as those in the mathematical equations.
    """
    def __init__(s, resolution=0.5, Q=1e-3*np.eye(3), resampling_threshold=0.3):
        s.lidar_log_odds_occ = np.log(9)
        s.lidar_log_odds_free = np.log(1/9.)

        # dynamics noise for the state (x, z, yaw)
        s.Q = Q

        # we resample particles if the effective number of particles
        # falls below s.resampling_threshold*num_particles
        s.resampling_threshold = resampling_threshold

        # initialize the map
        s.map = map_t(resolution)

    def read_data(s, src_dir, idx):
        """
        src_dir: location of the "data" directory
        """
        logging.info('> Reading data')
        s.idx = idx
        s.lidar_dir = src_dir + f'odometry/{s.idx}/velodyne/'
        s.poses = load_kitti_poses(src_dir + f'poses/{s.idx}.txt')
        s.lidar_files = sorted(os.listdir(src_dir + f'odometry/{s.idx}/velodyne/'))
        s.calib = load_kitti_calib(src_dir + f'calib/{s.idx}/calib.txt')

    def init_particles(s, n=100, p=None, w=None):
        """
        n: number of particles
        p: xy yaw locations of particles (3xn array)
        w: weights (array of length n)
        """
        s.n = n
        s.p = deepcopy(p) if p is not None else np.zeros((3, s.n))
        s.w = deepcopy(w) if w is not None else np.ones(n) / n

    @staticmethod
    def stratified_resampling(p, w):
        """
        Resampling step of the particle filter.
        """
        n = w.shape[0]
        new_p = np.empty_like(p)
        w_sum = np.cumsum(w)
        w_sum[-1] = 1.0
        
        positions = (np.arange(n) + np.random.rand(n)) / n
        indices = np.searchsorted(w_sum, positions)
        
        new_p = p[:, indices]
        new_w = np.full(n, 1.0 / n)
        
        return new_p, new_w

    @staticmethod
    def log_sum_exp(w):
        return w.max() + np.log(np.exp(w-w.max()).sum())

    def lidar2world(s, p, points):
        """
        Transforms LiDAR points to world coordinates.

        The particle state p is now interpreted as [x, z, theta], where:
        - p[0]: x translation
        - p[1]: z translation
        - p[2]: rotation in the x-z plane

        The input 'points' is an (N, 3) array of LiDAR points in xyz.
        """
        N = points.shape[0]
        points_h = np.vstack((points.T, np.ones((1, N))))  # (4, N)
        points_cam = s.calib @ points_h  # (3, N)
        p_x, p_z, theta = p[0], p[1], p[2]

        x_cam = points_cam[0, :]
        z_cam = points_cam[2, :]
        y_cam = points_cam[1, :]

        # rotation matrix
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        
        cam_coords = np.vstack((x_cam, z_cam))  # (2, N)
        world_coords_2d = R @ cam_coords + np.array([[p_x], [p_z]])  # (2, N)
        world_points = np.vstack((world_coords_2d[0, :], y_cam, world_coords_2d[1, :]))  # (3, N)

        return world_points.T

    @staticmethod
    def pose_to_state(pose):
        x = pose[0, 3]
        z = pose[2, 3]
        theta = np.arctan2(pose[2, 0], pose[0, 0])
        return np.array([x, z, theta])

    def get_control(s, t):
        """
        Use the pose at time t and t-1 to calculate what control the robot could have taken
        at time t-1 at state (x,y,th)_{t-1} to come to the current state (x,y,th)_t. We will
        assume that this is the same control that the robot will take in the function dynamics_step
        below at time t, to go to time t-1. need to use the smart_minus_2d
        function to get the difference of the two poses and we will simply
        set this to be the control.
        Extracts control in the state space [x, z, rotation] from consecutive poses.
        [x, z, theta]
        theta is the rotation around the Y-axis
              | cos  0  -sin |
        R_y = |  0   1    0  |
              |+sin  0   cos |
        R31 = +sin
        R11 =  cos
        yaw = atan2(R_31, R_11)
        """
        if t == 0:
            return np.zeros(3)
        
        state_prev = slam_t.pose_to_state(s.poses[t-1])
        state_curr = slam_t.pose_to_state(s.poses[t])
        
        control = smart_minus_2d(state_curr, state_prev)
        return control

    def dynamics_step(s, t):
        """
        Compute the control using get_control and perform that control on each particle to get the updated locations of the particles in the particle filter
        """
        #### TODO: XXXXXXXXXXX
        # Get the control from the odometry (based on consecutive poses).
        control = s.get_control(t)  # (3,)
        n = s.n
        noise_samples = np.random.multivariate_normal(np.zeros(3), s.Q, n)  # (n, 3)
    
        for i in range(n):
            noisy_control = control + noise_samples[i]
            s.p[:, i] = smart_plus_2d(s.p[:, i], noisy_control)

    @staticmethod
    def update_weights(w, obs_logp):
        """
        Given the observation log-probability and the weights of particles w, calculate the
        new weights as discussed in the writeup. Make sure that the new weights are normalized
        """
        log_w = np.log(w + 1e-8)
        log_w_new = log_w + obs_logp
        max_log = np.max(log_w_new)
        w_new = np.exp(log_w_new - max_log)
        w_new /= np.sum(w_new)
        return w_new

    def observation_step(s, t):
        """
        This function does the following things
            1. updates the particles using the LiDAR observations
            2. updates map.log_odds and map.cells using occupied cells as shown by the LiDAR data
        you can also store a thresholded version of the map here for plotting later
        """
        lidar_file = s.lidar_dir + s.lidar_files[t]
        points_xyz = clean_point_cloud(load_kitti_lidar_data(lidar_file))[:, :3]

        obs0 = np.zeros(s.n)
        for i in range(s.n):
            idx = s.map.grid_cell_from_xz(*s.lidar2world(s.p[:,i], points_xyz)[:, [0,2]].T)
            lo = np.clip(s.map.log_odds[idx[0], idx[1]], -30, 30)
            occ = (lo>0.5).sum()
            free= (lo< -0.5).sum()
            obs0[i] = occ - 0.5*free
        w0 = s.__class__.update_weights(s.w, obs0)

        best = s.p[:, np.argmax(w0)]
        idx_best = s.map.grid_cell_from_xz(*s.lidar2world(best, points_xyz)[:, [0,2]].T)
        for ix, iz in zip(*idx_best):
            s.map.log_odds[ix, iz] = np.clip(s.map.log_odds[ix,iz] + s.lidar_log_odds_occ, -30,30)
            s.map.cells[ix, iz] = 1

        obs1 = np.zeros(s.n)
        for i in range(s.n):
            idx = s.map.grid_cell_from_xz(*s.lidar2world(s.p[:,i], points_xyz)[:, [0,2]].T)
            obs1[i] = np.sum(s.map.cells[idx[0], idx[1]])
        s.w = s.__class__.update_weights(w0, obs1)  



    def resample_particles(s):
        """
        Resampling is a (necessary) but problematic step which introduces a lot of variance
        in the particles. We should resample only if the effective number of particles
        falls below a certain threshold (resampling_threshold). A good heuristic to
        calculate the effective particles is 1/(sum_i w_i^2) where w_i are the weights
        of the particles, if this number of close to n, then all particles have about
        equal weights and we do not need to resample
        """
        e = 1/np.sum(s.w**2)
        logging.debug('> Effective number of particles: {}'.format(e))
        if e/s.n < s.resampling_threshold:
            s.p, s.w = s.stratified_resampling(s.p, s.w)
            logging.debug('> Resampling')
