import numpy as np
from scipy import io
from quaternion import Quaternion
import matplotlib.pyplot as plt
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

# -------------------- Helper functions ---------------------- #

def _quaternion_mean(q_list, q_init, max_iter=10, tolerance=1e-3):
    mean_q = q_init
    for _ in range(max_iter):
        error_list = []
        for q in q_list:
            qw = q * mean_q.inv()
            qw.normalize()
            error_list.append(qw.axis_angle())
        errors = np.array(error_list)
        mean_error_vec = np.mean(errors, axis=0)
        norm_err = np.linalg.norm(mean_error_vec)
        if norm_err < tolerance:
            break
        delta_mean = Quaternion()
        delta_mean.from_axis_angle(mean_error_vec)
        mean_q = delta_mean * mean_q
        mean_q.normalize()
    return mean_q


def mean_sigma_points_optimized(sigma_points, x_prev):
    q_list = []
    for i in range(12):
        qtmp = Quaternion(sigma_points[0, i], sigma_points[1:4, i])
        q_list.append(qtmp)

    q_init = Quaternion(x_prev[0], x_prev[1:4])
    q_init.normalize()

    q_mean_obj = _quaternion_mean(q_list, q_init)
    q_mean = q_mean_obj.q

    omega_mean = np.mean(sigma_points[4:, :], axis=1)

    return np.hstack((q_mean, omega_mean))


def compute_residuals(sigma_points, x_mean):
    # -------------- (6 x 12）--------------
    gravity = Quaternion(0, [0, 0, 9.81])
    N = sigma_points.shape[1]  # 12
    predicted_obs = np.zeros((6, N))

    for i in range(N):
        qi_k = Quaternion(sigma_points[0, i], sigma_points[1:4, i])
        g_prime = qi_k.inv() * gravity * qi_k
        predicted_obs[0:3, i] = g_prime.vec()
        predicted_obs[3:, i] = sigma_points[4:, i]

    z_mean = np.mean(predicted_obs, axis=1)

    # -------------- W --------------

    q_mean_obj = Quaternion(x_mean[0], x_mean[1:4])
    q_mean_obj.normalize()
    omega_mean = x_mean[4:]

    W = np.zeros((6, N))   # 6 x 12
    for i in range(N):
        qi = Quaternion(sigma_points[0, i], sigma_points[1:4, i])
        q_res = qi * q_mean_obj.inv()
        q_res.normalize()
        axis_ang = q_res.axis_angle()

        # omega residual
        omega_res = sigma_points[4:, i] - omega_mean
        W[:, i] = np.hstack((axis_ang, omega_res))

    # -------------- Z --------------
    Z = predicted_obs - z_mean.reshape(6, 1)

    return W, Z, z_mean


# -------------------- Main functions ---------------------- #

def estimate_rot(data_num=1):
    imu = io.loadmat('imu/imuRaw' + str(data_num) + '.mat')
    accel = imu['vals'][0:3, :]
    gyro  = imu['vals'][3:6, :]
    T = accel.shape[1]
    ts = imu['ts'].flatten()

    # bias paras

    beta_accel = np.array([511.79280087, 500.62235235, 453.41056868])
    alpha_accel = np.array([28.97524257, 37.87650269, 50.])
    beta_gyro = np.array([375.24392142, 376.51331073, 370.14066682])
    alpha_gyro = np.array([200.35589163, 200.76215466, 200.88302403])

    def imu_read(raw, sensitivity, bias):
        return (raw - bias.reshape(3, 1)) * 3300 / (1023 * sensitivity.reshape(3, 1))

    accel_cal = imu_read(accel, alpha_accel, beta_accel) * np.array([-1, -1, 1]).reshape(3, 1)
    gyro_cal  = imu_read(gyro[[1, 2, 0], :], alpha_gyro, beta_gyro)

    # ukf paras
    x_km1 = np.array([1, 0, 0, 0, 0, 0, 0], dtype=np.float64) 
    P_km1 = np.eye(6) * 1.0
    Q = np.eye(6) * 0.1
    R = np.eye(6) * 0.1

    roll_list, pitch_list, yaw_list = [], [], []

    # -------------- loop --------------
    for i in range(T - 1):
        dt = ts[i+1] - ts[i]

        # sigma point
        S = np.linalg.cholesky(np.sqrt(6) * (P_km1 + Q))
        Wp = np.hstack((S, -S))  # 6 x 12
        sigma_points = np.zeros((7, 12))

        q_km1 = Quaternion(x_km1[0], x_km1[1:4])
        q_km1.normalize()
        for j in range(12):
            qw = Quaternion()
            qw.from_axis_angle(Wp[0:3, j])
            q_sigma = q_km1 * qw
            q_sigma.normalize()
            sigma_points[0:4, j] = q_sigma.q
            sigma_points[4:, j] = x_km1[4:] + Wp[3:, j]

        # propagate
        sigma_points_next = np.zeros_like(sigma_points)
        for j in range(12):
            dq = Quaternion()
            dq.from_axis_angle(sigma_points[4:, j] * dt)
            qq = Quaternion(sigma_points[0, j], sigma_points[1:4, j])
            qq_prop = qq * dq
            qq_prop.normalize()
            sigma_points_next[0:4, j] = qq_prop.q
            sigma_points_next[4:, j] = sigma_points[4:, j]

        # mean
        x_pred = mean_sigma_points_optimized(sigma_points_next, x_km1)

        # residuam
        W, Z, z_mean = compute_residuals(sigma_points_next, x_pred)

        # cov
        P_pred = (W @ W.T) / 12.0  
        Pzz = (Z @ Z.T) / 12.0
        Pxz = (W @ Z.T) / 12.0

        # kalman gain
        K = Pxz @ np.linalg.inv(Pzz + R)

        # update
        obs = np.hstack((accel_cal[:, i], gyro_cal[:, i]))
        innovation = obs - z_mean
        update_vec = K @ innovation
        d_quat = Quaternion()
        d_quat.from_axis_angle(update_vec[:3])
        q_pred = Quaternion(x_pred[0], x_pred[1:4])
        q_upd = d_quat * q_pred
        q_upd.normalize()

        x_upd = x_pred.copy()
        x_upd[:4] = q_upd.q
        x_upd[4:] = x_pred[4:] + update_vec[3:]

        P_upd = P_pred - K @ (Pzz + R) @ K.T

        x_km1 = x_upd
        P_km1 = P_upd

        
        qq_final = Quaternion(x_upd[0], x_upd[1:4])
        euler = qq_final.euler_angles()
        roll_list.append(euler[0])
        pitch_list.append(euler[1])
        yaw_list.append(euler[2])

    return np.array(roll_list), np.array(pitch_list), np.array(yaw_list)

if __name__ == '__main__':
    roll, pitch, yaw = estimate_rot(1)
    plt.figure()
    plt.plot(roll, label='roll')
    plt.plot(pitch, label='pitch')
    plt.plot(yaw, label='yaw')
    plt.legend()
    plt.title("Estimated Attitude - Optimized")
    plt.show()