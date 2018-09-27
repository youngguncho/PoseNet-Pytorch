import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# model = 'KingsCollege'
# model = 'Street'
# model = 'urban08'
model = 'NCLT'


pose_true = pd.read_csv('summary_' + model + '/pose_true.csv')
pose_estim = pd.read_csv('summary_' + model + '/pose_estim.csv')

position_true = pose_true.iloc[:, 0:3].values
position_estim = pose_estim.iloc[:, 0:3].values


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


ax.scatter(position_true[:,0], position_true[:,1], position_true[:,2], c='r', marker='o', label = 'turth')
ax.scatter(position_estim[:,0], position_estim[:,1], position_estim[:,2], c='b', marker='o', label = 'estimation')

#for i in range(position_true.__len__()):
#    position_set = np.vstack((position_true[i,:], position_estim[i,:]))
#    ax.plot(position_set[:,0], position_set[:,1], position_set[:,2], color='green', linewidth=0.5)

ax.legend()

# plt.axis('scaled')
plt.show();
