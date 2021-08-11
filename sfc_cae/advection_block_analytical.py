import os
import numpy as np
import scipy
import numpy.linalg as la
import scipy.linalg as sl
import scipy.sparse.linalg as spl
import scipy.linalg as sl
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.optimize as sop
import progressbar
# making slopes
from matplotlib.pyplot import LinearLocator
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import cmocean

# create an animation
from matplotlib import animation
from IPython.display import HTML


def square_wave_pseudo(center_x, center_y, dx, dy, d, grid_need, n, sigma, Lx = 10, Ly = 10):
    center_grid_X = center_x // dx
    center_grid_Y = center_y // dy
    grid_temp = np.zeros((n, n))
    half = grid_need // 2
    if grid_need % 2 == 0:
        left = int(max(0, center_grid_X - half))
        right = int(min(n, center_grid_X + half))
        down = int(max(0, center_grid_Y - half))
        up = int(min(n, center_grid_Y + half))
    elif grid_need % 2 == 1:
        left = int(max(0, center_grid_X - half))
        right = int(min(n, center_grid_X + half + 1))
        down = int(max(0, center_grid_Y - half))
        up = int(min(n, center_grid_Y + half + 1))

    grid_temp[left:right, down: up] = 1
    return grid_temp
    
def gaussian_wave(center_x, center_y, dx, dy, d, grid_need, n, sigma, Lx = 10, Ly = 10):
    xx, yy = np.meshgrid(np.linspace(0, Lx, n), np.linspace(0, Ly, n))
    dist = (xx - center_x) ** 2 + (yy - center_y) **2
    dist /= sigma
    dist = np.exp(-dist)
    return dist

class run_simulation_advection():
    def __init__(self, Lx = 10, Ly = 10, d = 2.5, n = 128, t_end = 0.4, dt = 0.01, init_func = square_wave_pseudo):
        self.dx = Lx / n
        self.dy = Ly / n
        self.grid_need = int(d / (Lx / n))
        
        self.func = init_func
        self.middle_x, self.middle_y = Lx/2, Ly/2
        self.steps = int(t_end / dt)
        self.Lx = Lx
        self.Ly = Ly
        self.d = d
        self.n = n
        self.t_end = t_end
        self.dt = dt
        self.simulation_times = 0
        # self.cmap = cmocean.tools.crop_by_percent(cmocean.cm.ice, 5, which='min', N=None)
        self.cmap = cmocean.cm.ice
        self.sigma = 2 * (np.random.random(1) + 1) ** 2
        self.full_stage = np.zeros((self.steps + 1, self.n, self.n))
        
    # update time stages using exact solution
    def time_update_exact(self):
        cnt_progress = 0
        bar=progressbar.ProgressBar(maxval=self.steps + 1)
        for i in range(self.steps + 1):
            self.x0 += self.U * self.dt
            self.y0 += self.V * self.dt
            self.full_stage[i] = self.func(self.x0, self.y0, self.dx, self.dy, self.d, self.grid_need, self.n, self.sigma, self.Lx, self.Ly)
            cnt_progress+=1
            bar.update(cnt_progress)
        bar.finish()
        
    def __call__(self, view_anime = True):
        self.simulation_times += 1
        print("simulation %d starting..."%self.simulation_times)
        self.x0, self.y0 = np.random.rand(2) * (self.Lx - self.d) + self.d/2
        print("(x0, y0):", (self.x0, self.y0))
        end_x0 = self.Lx - self.x0
        end_y0 = self.Ly - self.y0
        self.U, self.V = (end_x0 - self.x0) / self.t_end, (end_y0 - self.y0) / self.t_end
        truncate = 0
        while(end_x0 >= self.Lx + self.d/2 or end_x0 <= -self.d/2 or end_y0 >= self.Ly + self.d/2 or end_y0 <= -self.d/2):
            truncate += 1
            new_t = self.t_end + truncate * self.dt
            self.U, self.V = (end_x0 - self.x0) / new_t, (end_y0 - self.y0) / new_t
            end_x0 = self.x0 + self.U * new_t
            end_y0 = self.y0 + self.V * new_t
        print("(U, V):", (self.U, self.V))

        # iteration
        self.time_update_exact()
        
    def update_grid(self, n_step: int):
        arr = self.full_stage[n_step].reshape((self.n, self.n))
        vmax = min(np.max(arr) + 0.001, 1 + 1e-5)
        vmin = max(np.min(arr) - 0.001, -1e-5)
        self.cax.set_data(arr)
        self.cax.set_clim(vmin, vmax)
        return self.cax,
   
    def generate_anime(self):     
        # set label and locator
        ele_x = int(self.Lx / 5)
        ele_y = int(self.Ly / 5)
        ele_i = int(self.n / 5)
        ele_j = int(self.n / 5)
        x_label = [ele_x * i for i in range(5)]
        y_label = [ele_y * i for i in range(5)]
        x_label.append(self.Lx)
        y_label.append(self.Ly)
        i_loc = [ele_i * i for i in range(5)]
        j_loc = [ele_j * i for i in range(5)]
        i_loc.append(self.n - 1)
        j_loc.append(self.n - 1)
        fig, ax = plt.subplots(figsize=(10,8))
        # ax.set_title("advection on a %d * %d square grid" % (self.n, self.n), fontsize = 25)
        ax.xaxis.set_major_locator(plt.FixedLocator(i_loc))
        ax.xaxis.set_major_formatter(plt.FixedFormatter(x_label))
        ax.yaxis.set_major_locator(plt.FixedLocator(j_loc))
        ax.yaxis.set_major_formatter(plt.FixedFormatter(y_label))
        self.cax = ax.imshow(self.full_stage[0].reshape((self.n, self.n)), cmap = self.cmap, origin = 'lower', vmin = max(np.min(self.full_stage[0]) - 0.001, -1e-5), vmax = min(np.max(self.full_stage[0]) + 0.001, 1 + 1e-5))
        cb = fig.colorbar(self.cax)
        anim = animation.FuncAnimation(fig, self.update_grid, frames = np.arange(1, self.steps + 1))
        return anim
        
    def show_step(self, step):
        # set label and locator
        ele_x = int(self.Lx / 5)
        ele_y = int(self.Ly / 5)
        ele_i = int(self.n / 5)
        ele_j = int(self.n / 5)
        x_label = [ele_x * i for i in range(5)]
        y_label = [ele_y * i for i in range(5)]
        x_label.append(self.Lx)
        y_label.append(self.Ly)
        i_loc = [ele_i * i for i in range(5)]
        j_loc = [ele_j * i for i in range(5)]
        i_loc.append(self.n - 1)
        j_loc.append(self.n - 1)
        fig, ax = plt.subplots(figsize=(10,8))
        # ax.set_title("advection on a %d * %d square grid" % (self.n, self.n), fontsize = 25)
        ax.xaxis.set_major_locator(plt.FixedLocator(i_loc))
        ax.xaxis.set_major_formatter(plt.FixedFormatter(x_label))
        ax.yaxis.set_major_locator(plt.FixedLocator(j_loc))
        ax.yaxis.set_major_formatter(plt.FixedFormatter(y_label))
        self.cax = ax.imshow(self.full_stage[step].reshape((self.n, self.n)), cmap = self.cmap, origin = 'lower', vmin = max(np.min(self.full_stage[step]) - 0.001, -1e-5), vmax = min(np.max(self.full_stage[step]) + 0.001, 1 + 1e-5))
        cb = fig.colorbar(self.cax)
        plt.show()
    
    
    def clear_run_times(self):
        self.simulation_times = 0
        
    def output_snapshots(self):
        command = F'mkdir -p ./output/simulation_{self.simulation_times}'
        os.system(command)
        for i in range(self.steps + 1):
            np.savetxt("./output/simulation_%d/step_%d.txt"%(self.simulation_times, i), self.full_stage[i].flatten()); 
