import matplotlib.pyplot as plt
import numpy as np

import ppafm.ocl.oclUtils as oclu
from ppafm.ocl.field import DataGrid

oclu.init_env(i_platform=0)


def test_clamp_visual():

    triangle_np = np.concatenate([np.linspace(-3, 10, 100), np.linspace(10, -3, 100)[1:]])
    triangle_data_grid = DataGrid(triangle_np[None, None], lvec=np.concatenate([np.zeros((1, 3)), np.eye(3)], axis=0))

    minimum = -1
    maximum = 5
    width = 0.5
    triangle_clamp_hard = triangle_data_grid.clamp(minimum=minimum, maximum=maximum, clamp_type="hard", in_place=False).array[0, 0]
    triangle_clamp_soft = triangle_data_grid.clamp(minimum=minimum, maximum=maximum, clamp_type="soft", soft_clamp_width=width, in_place=False).array[0, 0]

    plt.plot(triangle_np)
    plt.plot(triangle_clamp_hard)
    plt.plot(triangle_clamp_soft)

    plt.legend(["Original", "Hard clamp", "Soft clamp"])
    plt.savefig("clamps.png")
    plt.show()


if __name__ == "__main__":
    test_clamp_visual()
