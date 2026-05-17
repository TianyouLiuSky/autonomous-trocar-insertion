import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

BOX_MM = 24.0

CORNERS = np.array([
    [0,      0,      0     ],
    [BOX_MM, 0,      0     ],
    [0,      BOX_MM, 0     ],
    [BOX_MM, BOX_MM, 0     ],
    [0,      0,      BOX_MM],
    [BOX_MM, 0,      BOX_MM],
    [0,      BOX_MM, BOX_MM],
    [BOX_MM, BOX_MM, BOX_MM],
])

rob = np.array([
    [15.3093,-106.6798,-15.3836],
    [15.3093,-106.6798,-15.3836],
    [38.8343,-106.6798,-15.3836],
    [15.3143, -96.7288,-15.3836],
    [39.3043, -96.7288,-15.3836],
    [15.6283,-106.5448,  8.2894],
    [38.8353,-106.6778,  8.6094],
    [15.3143, -96.7238,  8.6114],
    [39.3043, -96.7238,  8.6114],
])

# use sample 2 (index 1) as home, skip sample 1 (index 0)
home = rob[1]
rob_off = rob[1:] - home   # shape (8,3), in mm

cmd = CORNERS              # shape (8,3), in mm

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(*cmd.T,     c='red',  s=80, label='commanded (in robot frame)')
ax.scatter(*rob.T, c='blue', s=80, label='observed (in camera frame)')

ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.legend()
plt.tight_layout()
plt.show()