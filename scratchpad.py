import matplotlib.pyplot as plt

import numpy as np

def rotating_3d_plot():
    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    plt.style.use("_mpl-gallery")

    def certainty(q, min_q):
        return np.where(q >= min_q, np.exp(-min_q*q), np.nan)
        # return np.sqrt((1 / np.exp(q)) * (1 - min_q))
        # return np.where(q >= min_q, np.sqrt((1 - q) * (1 - min_q)), 0)

    Q = np.arange(0, 1, 0.001)
    Q_min = np.arange(0, 1, 0.001)
    Q, Q_min = np.meshgrid(Q, Q_min)
    Z = certainty(Q, Q_min)

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,6))
    ax.plot_surface(Q, Q_min, Z, vmin=Z.min())
    ax.set_xlabel("quality")
    ax.set_ylabel("min quality")
    ax.set_zlabel("certainty")
    ax.set_box_aspect(None, zoom=0.85)

    # Rotate the axes and update
    for angle in range(0, 360*4 + 1):
        # Normalize the angle to the range [-180, 180] for display
        angle_norm = (angle + 180) % 360 - 180


        azim = angle_norm

        # Update the axis view and title
        ax.view_init(30, azim, 0)

        plt.draw()
        plt.pause(.001)

if __name__ == "__main__":
    rotating_3d_plot()
