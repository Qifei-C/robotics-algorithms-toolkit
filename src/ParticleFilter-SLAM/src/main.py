# Pratik Chaudhari (pratikac@seas.upenn.edu)
# Minku Kim (minkukim@seas.upenn.edu)

import click, tqdm, random

from tqdm import tqdm
from slam import *

def run_dynamics_step(src_dir, log_dir, idx, t0=0, draw_fig=False, plot_mode="separate"):
    """
    This function is for you to test your dynamics update step. It will create
    two figures after you run it. The first one is the robot location trajectory
    using odometry information obtained form the lidar. The second is the trajectory
    using the PF with a very small dynamics noise. The two figures should look similar.
    """
    slam = slam_t(Q=1e-8*np.eye(3))
    slam.read_data(src_dir, idx)

    # Trajectory using odometry (xz and yaw) in the lidar data
    d = slam.poses
    pose = np.column_stack([d[:,0,3], d[:,1,3], d[:,2,3]]) # X Y Z
    
    if plot_mode == "separate":
        plt.figure(1)
        plt.clf()
        plt.title("Trajectory using onboard odometry")
        plt.plot(pose[:, 0], pose[:, 2], "k-", label="Odometry")
        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.grid(True)
        plt.legend()
        odom_path = os.path.join(log_dir, f"odometry_{idx}.jpg")
        logging.info("> Saving odometry plot in " + odom_path)
        plt.savefig(odom_path)
        
    else:
        plt.figure(1, figsize=(8, 8))
        plt.clf()
        plt.title("Trajectory: Odometry vs PF")
        plt.plot(pose[:, 0], pose[:, 2], "k-", label="Odometry")
        

    # dynamics propagation using particle filter
    # n: number of particles, w: weights, p: particles (3 dimensions, n particles)
    # S covariance of the xyth location
    # particles are initialized at the first xyth given by the lidar
    # for checking in this function
    
    n = 3
    w = np.ones(n)/float(n)
    p = np.zeros((3,n), dtype=np.float64)
    slam.init_particles(n,p,w)
    slam.p[:,0] = deepcopy(pose[0])

    print('> Running prediction')
    t0 = 0
    T = len(d)
    ps = deepcopy(slam.p)

    for t in tqdm.tqdm(range(t0+1,T)):
        slam.dynamics_step(t)
        ps = np.hstack((ps, slam.p))
            
    if plot_mode == "separate":
        plt.figure(2)
        plt.clf()
        plt.title("Trajectory using PF (Q≈0)")

    plt.plot(ps[0], ps[1], "*c", label="PF (Q≈0)")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.grid(True)
    plt.legend()

    # ---------- plot ---------- #
    if plot_mode == "separate":
        pf_path = os.path.join(log_dir, f"dynamics_only_{idx}.jpg")
        logging.info("> Saving plot in " + pf_path)
        plt.savefig(pf_path)
    else:
        comb_path = os.path.join(log_dir, f"traj_combined_{idx}.jpg")
        logging.info("> Saving combined plot in " + comb_path)
        plt.savefig(comb_path)

    plt.close("all")

def run_observation_step(src_dir, log_dir, idx, is_online=False):
    """
    This function is for you to debug your observation update step
    It will create three particles np.array([[0.2, 2, 3],[0.4, 2, 5],[0.1, 2.7, 4]])
    * Note that the particle array has the shape 3 x num_particles so
    the first particle is at [x=0.2, y=0.4, z=0.1]
    This function will build the first map and update the 3 particles for one time step.
    After running this function, you should get that the weight of the second particle is the largest since it is the closest to the origin [0, 0, 0]
    """
    slam = slam_t(resolution=0.5)
    slam.read_data(src_dir, idx)

    # t=0 sets up the map using the yaw of the lidar, do not use yaw for
    # other timestep
    # initialize the particles at the location of the lidar so that we have some
    # occupied cells in the map to calculate the observation update in the next step
    t0 = 0
    d = slam.poses
    pose = np.column_stack([d[t0,0,3], d[t0,1,3], np.arctan2(-d[t0,2,0], d[t0,0,0])])
    logging.debug('> Initializing 1 particle at: {}'.format(pose))
    slam.init_particles(n=1,p=pose.reshape((3,1)),w=np.array([1]))

    slam.observation_step(t=0)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

    # reinitialize particles, this is the real test
    logging.info('\n')
    n = 3
    w = np.ones(n)/float(n)
    p = np.array([[2, 0.2, 3],[2, 0.4, 5],[2.7, 0.1, 4]])
    slam.init_particles(n, p, w)

    slam.observation_step(t=1)
    logging.info('> Particles\n: {}'.format(slam.p))
    logging.info('> Weights: {}'.format(slam.w))

def run_slam(src_dir, log_dir, idx):
    """
    This function runs slam. We will initialize the slam just like the observation_step
    before taking dynamics and observation updates one by one. You should initialize
    the slam with n=50 particles, you will also have to change the dynamics noise to
    be something larger than the very small value we picked in run_dynamics_step function
    above.
    """
    slam = slam_t(resolution=0.5, Q=np.diag([1e-4,1e-4,1e-5]))
    slam.read_data(src_dir, idx)
    T = len(slam.lidar_files)

    # again initialize the map to enable calculation of the observation logp in
    # future steps, this time we want to be more careful and initialize with the
    # correct lidar scan
    
    # initialize say n = 50 particles
    n_particles = 50
    
    # run dynamics, observation and resampling steps for each timepoint
    initial_pose = slam.poses[0]
    x0 = initial_pose[0, 3]
    z0 = initial_pose[2, 3]
    theta0 = np.arctan2(initial_pose[2, 0], initial_pose[0, 0])
    initial_state = np.array([x0, z0, theta0])
    
    poses = np.array([
        initial_pose[0,3],
        initial_pose[2,3],
        np.arctan2(initial_pose[2,0], initial_pose[0,0])
    ])
    
    particles = np.tile(initial_state.reshape(3, 1), (1, n_particles))
    weights = np.ones(n_particles) / n_particles
    slam.init_particles(n_particles, particles, weights)
    
    
    # save data to be plotted later
    best_trajectory = [initial_state.copy()]
    
    odometry_trajectory = []
    odometry_trajectory.append(np.array([initial_pose[0, 3], initial_pose[2, 3]]))
    
    entropy_list = []  
    
    # loop
    for t in tqdm(range(1, T), desc="SLAM Progress"):
        slam.dynamics_step(t)
        slam.observation_step(t)
        slam.resample_particles()
        
        best_idx = np.argmax(slam.w)
        best_state = slam.p[:, best_idx]
        best_trajectory.append(best_state.copy())
        
        current_pose = slam.poses[t]
        odometry_trajectory.append([current_pose[0, 3], current_pose[2, 3]])
        
        

    best_trajectory = np.array(best_trajectory)  # shape: (T, 3)
    odometry_trajectory = np.array(odometry_trajectory)
    binary_map = (slam.map.log_odds > slam.map.log_odds_thresh).astype(np.uint8)
    
    # Plot
    plt.figure
    plt.imshow(binary_map.T, origin='lower', extent=[slam.map.xmin, slam.map.xmax, slam.map.zmin, slam.map.zmax], cmap='jet', vmin=0, vmax=1, interpolation='nearest')
    plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'b-', label='Estimated Trajectory')
    plt.plot(odometry_trajectory[:, 0], odometry_trajectory[:, 1], 'r-', label='Odometry Trajectory')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.title('SLAM Estimated Trajectory')
    plt.grid(True)
    plt.legend()

    
    # Save the plot
    traj_plot_file = os.path.join(log_dir, 'slam_trajectory.jpg')
    plt.savefig(traj_plot_file)
    plt.show()
    
    return best_trajectory


from load_data import (
    show_kitti_lidar,
    trajectory2d,
    trajectory3d
)

@click.command()
@click.option('--src_dir', default='./KITTI/', help='data directory', type=str)
@click.option('--log_dir', default='logs', help='directory to save logs', type=str)
@click.option('--idx', default='00', help='dataset number', type=str)
@click.option('--mode', default='slam',
              help='choices: dynamics OR observation OR slam', type=str)
@click.option('--visualize', is_flag=True, help='whether to show visualization after run')
def main(src_dir, log_dir, idx, mode, visualize):
    # Run python main.py --help to see how to provide command line arguments
    
    if mode not in ['slam', 'dynamics', 'observation']:
        raise ValueError(f'Unknown argument --mode {mode}')
        sys.exit(1)

    np.random.seed(42)
    random.seed(42)


    if mode == 'dynamics':
        run_dynamics_step(src_dir, log_dir, idx, plot_mode="combined" if visualize else "separate")
        if visualize:
            trajectory2d(os.path.join(src_dir, f'odometry/{idx}/velodyne/'))
        sys.exit(0)

    elif mode == 'observation':
        run_observation_step(src_dir, log_dir, idx)
        if visualize:
            bin_file = os.path.join(src_dir, f'odometry/{idx}/velodyne/{idx}.bin')
            pc = load_kitti_lidar_data(bin_file)
            show_kitti_lidar(pc)
        sys.exit(0)

    else:  
        best_traj = run_slam(src_dir, log_dir, idx)
        if visualize:
            trajectory2d(os.path.join(src_dir, f'poses/{idx}.txt'))
            trajectory3d(os.path.join(src_dir, f'poses/{idx}.txt'))
        return best_traj

if __name__=='__main__':
    main()