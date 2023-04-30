import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Plot success or failure by initial conditions.')
    parser.add_argument('initial_conditions', nargs='+', help='Files to process')
    args = parser.parse_args()

    data = []

    for p in args.initial_conditions:
        df = pd.read_csv(p)
        data.append(df[['ee_pose_x', 'ee_pose_y', 'ee_pose_z', 'success']].to_numpy())

    data = np.vstack(data).T
    c    = data[3]

    # Create a figure and a 3D Axes object
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points with different colors
    ax.scatter(data[0][c==0], data[1][c==0], data[2][c==0], c='r', marker='o', label='Failure')
    ax.scatter(data[0][c==1], data[1][c==1], data[2][c==1], c='b', marker='o', label='Success')

    # Set the labels and the legend
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_zlabel('Position Z')

    a_min, a_max = data[:3].min(), data[:3].max()

    ax.set_xlim(a_min, a_max)
    ax.set_ylim(a_min, a_max)
    ax.set_zlim(a_min, a_max)

    ax.legend()

    # Show the plot
    plt.show()
