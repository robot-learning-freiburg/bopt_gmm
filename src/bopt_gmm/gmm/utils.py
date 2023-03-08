from numpy import linalg as la
import math
import numpy as np


def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    identity = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += identity * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


def rollout(gmm, start, steps=100, dt=1/30, stack_vels=False):
    traj  = [start]
    vels  = []
    point = start
    position_dim = tuple(range(gmm.n_dims // 2))

    print(f'Position Dim: {position_dim} {gmm.n_dims}')

    for _ in range(steps):
        vel   = gmm.predict(point, position_dim, full=True)
        point = point + dt * vel
        vels.append(vel)
        traj.append(point)
    
    if stack_vels:
        return np.hstack((traj[:-1], vels))
    return np.vstack(traj)


def f_translation(pos):
    out = np.eye(4)
    out[:3, 3] = pos
    return out

def f_rot_trans(rot, pos):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3,  3] = pos
    return out

def real_quat_from_matrix(frame):
    tr = frame[0,0] + frame[1,1] + frame[2,2]

    if tr > 0:
        S = math.sqrt(tr+1.0) * 2 # S=4*qw
        qw = 0.25 * S
        qx = (frame[2,1] - frame[1,2]) / S
        qy = (frame[0,2] - frame[2,0]) / S
        qz = (frame[1,0] - frame[0,1]) / S
    elif frame[0,0] > frame[1,1] and frame[0,0] > frame[2,2]:
        S  = math.sqrt(1.0 + frame[0,0] - frame[1,1] - frame[2,2]) * 2 # S=4*qx
        qw = (frame[2,1] - frame[1,2]) / S
        qx = 0.25 * S
        qy = (frame[0,1] + frame[1,0]) / S
        qz = (frame[0,2] + frame[2,0]) / S
    elif frame[1,1] > frame[2,2]:
        S  = math.sqrt(1.0 + frame[1,1] - frame[0,0] - frame[2,2]) * 2 # S=4*qy
        qw = (frame[0,2] - frame[2,0]) / S
        qx = (frame[0,1] + frame[1,0]) / S
        qy = 0.25 * S
        qz = (frame[1,2] + frame[2,1]) / S
    else:
        S  = math.sqrt(1.0 + frame[2,2] - frame[0,0] - frame[1,1]) * 2 # S=4*qz
        qw = (frame[1,0] - frame[0,1]) / S
        qx = (frame[0,2] + frame[2,0]) / S
        qy = (frame[1,2] + frame[2,1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))


def draw_gmm(vis, namespace, gmm, dimensions=None, visual_scaling=1.0, frame=None):
    dimensions = list(gmm.semantic_dims().keys()) if dimensions is None else dimensions

    vis.begin_draw_cycle(f'{namespace}/labels', 
                         f'{namespace}/inference', 
                         f'{namespace}/variance')
    for d in dimensions:
        if '|' in d:
            dims_in, dims_out = d.split('|')
            dims_in  = gmm.semantic_dims()[dims_in]
            dims_out = gmm.semantic_dims()[dims_out]

            poses = []
            for k, (mu_k_in, mu_k_out, sigma_io, sigma_ii) in enumerate(zip(gmm.mu(dims_in), gmm.mu(dims_out), gmm.sigma(dims_in, dims_out), gmm.sigma(dims_in, dims_in))):
                w, v = np.linalg.eig(gmm._cvar[k])
                # sigma_inv = np.linalg.inv(sigma_io)

                mat = sigma_io.dot(np.linalg.pinv(sigma_ii))

                # pose = np.eye(4)
                # pose[:3, :3] = v.astype(float) #  (v * w).astype(float) * visual_scaling
                # pose[:3,  3] = mu_k
                # poses.append(pose)

                print(v, w)
                print(v[:, 0].T.dot(v[:, 0]), 
                      v[:, 0].T.dot(v[:, 1]),
                      v[:, 0].T.dot(v[:, 2]),)

                vis.draw_text(f'{namespace}/labels', mu_k_in, f'{d} {k}', height=0.04)
                vis.draw_vector(f'{namespace}/inference', mu_k_in, (mat.T[0][:3] + mu_k_out) * visual_scaling, r=1, g=0, b=0, frame=frame)
                vis.draw_vector(f'{namespace}/inference', mu_k_in, (mat.T[1][:3] + mu_k_out) * visual_scaling, r=0, g=1, b=0, frame=frame)
                vis.draw_vector(f'{namespace}/inference', mu_k_in, (mat.T[2][:3] + mu_k_out) * visual_scaling, r=0, g=0, b=1, frame=frame)


            # vis.draw_poses(f'{namespace}/inference', np.eye(4), 1.0, 0.005, poses, r=1, g=1, b=1, a=1, frame=frame)
                # vis.draw_ellipsoid(f'{namespace}/inference', f_rot_trans(v, mu_k), w * visual_scaling, frame=frame)
        else:
            dims  = gmm.semantic_dims()[d]

            for k, (mu_k, sigma_k) in enumerate(zip(gmm.mu(dims), gmm.sigma(dims, dims))):
                w, v = np.linalg.eig(sigma_k)
                
                vis.draw_ellipsoid(f'{namespace}/variance', f_rot_trans(v, mu_k), w / visual_scaling, frame=frame)
    vis.render(f'{namespace}/labels', 
               f'{namespace}/inference', 
               f'{namespace}/variance')


def draw_gmm_stats(vis, namespace, gmm, observation, frame=None):
    activations = gmm.get_weights(observation, gmm.state_dim)
    
    if activations.sum() == 0:
        print(f'Zero activation {activations.sum()}')
        return

    activations = np.nan_to_num(activations / activations.sum()).flatten()

    vis.begin_draw_cycle(f'{namespace}/stats')
    for mu_k in [gmm.mu(v) for k, v in gmm.semantic_obs_dims().items() if len(v) == 3]:
        for k, a in enumerate(activations):
            vis.draw_vector(f'{namespace}/stats', mu_k[k], (0, 0, a), frame=frame)
    
    vis.render(f'{namespace}/stats')


if __name__ == "__main__":
    for i in range(10):
        for j in range(2, 100):
            A = np.random.randn(j, j)
            B = nearestPD(A)
            assert isPD(B)
    print("unit test passed!")
