import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

#######################################################
################    Training Curves    ################
#######################################################
mean_reward = pd.read_csv('data/policy-v6/mean_reward.csv')
RMSE_position = pd.read_csv('data/policy-v6/RMSE_position.csv')

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Training Steps [-]')
ax1.set_ylabel('Mean Episode Reward', color=color)
ax1.plot(mean_reward['Step'], mean_reward['Value'], color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid()

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel(r'Mean Episode $RMSE||p_{ref}-p||_2$', color=color)  # we already handled the x-label with ax1
ax2.plot(RMSE_position['Step'], RMSE_position['Value'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped


########################################################
###############    Disturbance Curves    ###############
########################################################

pos_dist_conv = pd.read_csv('data/position_disturbance_conventional.csv', delimiter=" ", names=["x", "y", "z"])
pos_dist_aug = pd.read_csv('data/position_disturbance_augmented.csv', delimiter=" ", names=["x", "y", "z"])

time = np.linspace(0., 25., len(pos_dist_conv))
ref = np.zeros((pos_dist_conv.shape[0], ))

fig1, ax = plt.subplots(3, 1)
ax[0].plot(time, pos_dist_conv["x"], label="Conventional")
ax[0].plot(time, pos_dist_aug["x"], label="Augmented")
ax[0].plot(time, ref, label="Reference")
ax[0].scatter([0., 25.], [0., 0.])
ax[0].set_ylabel("X-position [m]")
ax[0].grid()
ax[0].legend()

ax[1].plot(time, pos_dist_conv["y"], time, pos_dist_aug["y"], time, ref)
ax[1].scatter([0., 25.], [0., 0.])
ax[1].set_ylabel("Y-position [m]")
ax[1].grid()

ax[2].plot(time, pos_dist_conv["z"], time, pos_dist_aug["z"], time, ref)
ax[2].scatter([0., 25.], [0., 0.])
ax[2].set_ylabel("Z-position [m]")
ax[2].set_xlabel("Time [s]")
ax[2].grid()

ax = plt.figure().add_subplot(projection='3d')
ax.plot(pos_dist_conv["x"], pos_dist_conv["y"], pos_dist_conv["z"], label="Conventional")
ax.plot(pos_dist_aug["x"], pos_dist_aug["y"], pos_dist_aug["z"], label="Augmented")
ax.set_xlabel("X-position [m]")
ax.set_ylabel("Y-position [m]")
ax.set_zlabel("Z-position [m]")
ax.legend()

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
