import hydra
import iai_bullet_sim    as ibs
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import rospy

from argparse import ArgumentParser
from pathlib  import Path
from roebots  import ROSVisualizer

from bopt_gmm      import common, \
                          gmm as libgmm
from bopt_gmm.envs import ENV_TYPES


if __name__ == '__main__':
    rospy.init_node('ic_runner')

    vis = ROSVisualizer('vis')

    parser = ArgumentParser(description='Run models under given initial conditions.')
    parser.add_argument('hy', help='Hydra config for the scenario.')
    parser.add_argument('ic', help='CSV file with initial conditions.')
    parser.add_argument('--filter', default=None, help='Equality filter for columns query.')
    parser.add_argument('--show-gui', action='store_true', help='Show interactive GUI')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()

    df  = pd.read_csv(args.ic)

    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)

    class ContactEnv(ENV_TYPES[cfg.env.type]):
        def __init__(self, cfg, show_gui):
            super().__init__(cfg, show_gui)

            self.handle   = self.door.links['handle_link']
            self.c_normals = []
            self.c_force   = []

        def reset(self, initial_conditions=None):
            self.c_normals = []
            self.c_force   = []
            
            return super().reset(initial_conditions)

        def step(self, action):
            out = super().step(action)

            contacts = self.handle.get_contacts(self.robot)
            if len(contacts) > 0:
                rel_normal = ibs.Vector3(*np.mean([self.frame.pose.inv().dot(c.posOnA) for c in contacts], axis=0))
                self.c_normals.append(rel_normal)
                self.c_force.append(ibs.Vector3(*np.mean([self.frame.pose.inv().dot(-c.normalOnB * c.normalForce * 0.001) for c in contacts], axis=0)))

            return out
        
    env = ContactEnv(cfg.env, args.show_gui)
    missing_fields = [f for f in env.config_space if f not in df.columns]
    if len(missing_fields) > 0:
        print(f'File {args.ic} does not contain all fields required for environment "{cfg.env.type}". Missing: {missing_fields}')
        exit(-1)


    field_idcs = {f: i for i, f in enumerate(df.columns) if f in env.config_space}

    if args.filter is not None:
        c, v = args.filter.split('=')
        df   = df[df[c] == int(v)].reset_index(drop=True)

    gmm_path = Path(cfg.bopt_agent.gmm.model)
    gmm      = libgmm.GMM.load_model(cfg.bopt_agent.gmm.model)
    agent    = common.AgentWrapper(gmm, cfg.bopt_agent.gripper_command)

    n_successes = 0
    n_episodes  = 0
    successes   = []
    contacts    = []
    normals     = []
    forces      = []

    for i_row in range(len(df)):
        ic = {f: df.iloc[i_row, i] for f, i in field_idcs.items()}


        _, _, info = common.run_episode(env, agent, cfg.bopt_agent.num_episode_steps, initial_conditions=ic)

        n_episodes  += 1
        n_successes += int(info['success'])

        successes.append(int(info['success']))

        if len(env.c_normals) > 0:
            contacts.append(1)
            normals.append(env.c_normals)
            forces.append(env.c_force)
        else:
            contacts.append(0)
            normals.append([ibs.Vector3.zero()])
            forces.append([ibs.Vector3.zero()])

        print(f'Success VS original: {int(info["success"])} | {df.success[i_row]}')
        print(f'       Made contact: {len(env.c_normals) > 0}')
        print(f'   Current accuracy: {n_successes/n_episodes} ({n_successes} / {n_episodes})')

    c_in_failure = [c for s, c in zip(successes, contacts) if s == 0]
    print(f'Contacts in failures: {sum(c_in_failure) / (n_episodes - n_successes)}')

    vis.begin_draw_cycle('success points', 'success force', 'failure points', 'failure force')

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    normals   = [np.vstack(p) for p in normals]
    forces    = [np.vstack(p) for p in forces]
    successes =  np.array(successes)

    s_points  = np.vstack([n for s, n in zip(successes, normals) if s == 1])
    s_normals = np.vstack([f for s, f in zip(successes, forces)  if s == 1])
    f_points  = np.vstack([n for s, n in zip(successes, normals) if s == 0])
    f_normals = np.vstack([f for s, f in zip(successes, forces)  if s == 0])

    vis.draw_points('success points', np.eye(4), 0.005, s_points, r=0, g=0, b=1)
    vis.draw_points('failure points', np.eye(4), 0.005, f_points, r=1, g=0, b=0)
    
    # hstack and reshape
    s_lines = np.hstack((s_points, s_points + s_normals)).reshape((len(s_points) * 2, 3))
    f_lines = np.hstack((f_points, f_points + f_normals)).reshape((len(f_points) * 2, 3))

    vis.draw_lines('success force', np.eye(4), 0.001, s_lines, r=0, g=0, b=1)
    vis.draw_lines('failure force', np.eye(4), 0.001, f_lines, r=1, g=0, b=0)

    vis.render()

    rospy.sleep(0.1)


    # # Plot the points with different colors
    # ax.scatter(s_points[0], s_points[1], s_points[2], c='b', marker='.', label='Success')
    # ax.scatter(f_points[0], f_points[1], f_points[2], c='r', marker='.', label='Failure')

    # # Set the labels and the legend
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # ax.set_xlim(-1.1, 1.1)
    # ax.set_ylim(-1.1, 1.1)
    # ax.set_zlim(-1.1, 1.1)

    # ax.legend()

    # Show the plot
    # plt.show()