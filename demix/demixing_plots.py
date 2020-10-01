from matplotlib import pyplot as plt
import numpy as np
from _datetime import datetime

DATE_TIME_FORMAT = "%d%m%Y_%H%M%S"
LINE_WIDTH = 0.3
TRACES_X_LABEL = "frames"


def get_time_stamp():
    return datetime.now().strftime(DATE_TIME_FORMAT)


def save_plot(name, path, show):
    full_name = name + ".png"
    plt.savefig(path + full_name)
    print("Plot saved: " + full_name)
    if show:
        plt.show(block=False)


def plot_final_traces(beta_hat2, ref_im, mov_dims, X2, name, path, show=True):
    num_traces = beta_hat2.shape[0]
    plt.figure(figsize=(25, 3 * num_traces))
    for idx in range(num_traces):
        plt.subplot(num_traces, 2, 2 * idx + 1, xlabel=TRACES_X_LABEL)
        plt.plot(beta_hat2[idx, :], linewidth=LINE_WIDTH)
        plt.subplot(num_traces, 2, 2 * idx + 2)
        lower, upper = np.percentile(ref_im.flatten(), [1, 99])
        plt.imshow(ref_im.transpose(1, 0), cmap='gray', interpolation='none', clim=[lower, upper])
        cell_loc = X2[:, idx].reshape(mov_dims)
        cell_loc = np.ma.masked_where(abs(cell_loc) < 1e-8, cell_loc)
        plt.imshow(cell_loc.transpose(1, 0), cmap='jet', alpha=0.5)
    save_plot(name, path, show)


def plot_intermediate_traces(a, c, ref_im, mov_dims, name, path, show=True):
    cell_ct = c.shape[1]
    plt.figure(figsize=(25, 3 * cell_ct))
    for cell_num in range(cell_ct):
        plt.subplot(cell_ct, 2, 2 * cell_num + 1, xlabel=TRACES_X_LABEL)
        plt.plot(c[:, cell_num], linewidth=LINE_WIDTH)
        plt.title(cell_num, size=16)
        plt.subplot(cell_ct, 2, 2 * cell_num + 2)
        lower, upper = np.percentile(ref_im.flatten(), [1, 99])
        plt.imshow(ref_im, cmap='gray', interpolation='none', clim=[lower, upper])
        cell_loc = a[:, cell_num].reshape(mov_dims)
        cell_loc = np.ma.masked_where(cell_loc == 0, cell_loc)
        plt.imshow(cell_loc, cmap='jet', alpha=0.5)
        plt.colorbar()
    save_plot(name, path, show)


def plot_bg_traces(fb, ff, mov_dims, name, path, show=True):
    bg_rank = fb.shape[1]
    plt.figure(figsize=(25, 3 * bg_rank))
    for bkgd_num in range(bg_rank):
        plt.subplot(bg_rank, 2, 2 * bkgd_num + 1, xlabel=TRACES_X_LABEL)
        plt.plot(ff[:, bkgd_num], linewidth=LINE_WIDTH)
        bkgd_comp = fb[:, bkgd_num].reshape(mov_dims)
        plt.subplot(bg_rank, 2, 2 * bkgd_num + 2)
        plt.imshow(bkgd_comp)
        plt.colorbar()
    save_plot(name, path, show)


def plot_nmf_traces(rlt, ref_im, mov_dims, name, path, show=True):
    ref_im = ref_im
    cell_ct = rlt["fin_rlt"]["c"].shape[1]
    plt.figure(figsize=(25, 3 * cell_ct))
    for cell_num in range(cell_ct):
        plt.subplot(cell_ct, 2, 2 * cell_num + 1, xlabel=TRACES_X_LABEL)
        plt.plot(rlt["fin_rlt"]["c"][:, cell_num], linewidth=LINE_WIDTH)
        plt.title(cell_num, size=24)

        plt.subplot(cell_ct, 2, 2 * cell_num + 2)

        lower, upper = np.percentile(ref_im.flatten(), [1, 99])
        plt.imshow(ref_im.transpose(1,0), cmap='gray', interpolation='none', clim=[lower, upper])

        cell_loc = rlt["fin_rlt"]["a"][:, cell_num].reshape(mov_dims[0], mov_dims[1])
        cell_loc = np.ma.masked_where(cell_loc == 0, cell_loc)
        plt.imshow(cell_loc.transpose(1,0), cmap='jet', alpha=0.5)
    save_plot(name, path, show)


def plot_super_pixels(rlt, ref_im, name, path, show=True):
    num_pass = len(rlt["superpixel_rlt"])
    scale = np.maximum(1, (
            rlt["superpixel_rlt"][0]["connect_mat_1"].shape[1] / rlt["superpixel_rlt"][0]["connect_mat_1"].shape[0]))
    plt.figure(figsize=(10 * scale * num_pass, 10))

    plt.subplot(1, num_pass + 2, 1)
    plt.imshow(ref_im.transpose(1, 0))
    for p in range(num_pass):
        connect_mat_1 = rlt["superpixel_rlt"][p]["connect_mat_1"]
        pure_pix = rlt["superpixel_rlt"][p]["pure_pix"]
        brightness_rank = rlt["superpixel_rlt"][p]["brightness_rank"]
        ax1 = plt.subplot(1, num_pass + 2, p + 2)
        dims = connect_mat_1.shape
        connect_mat_1_pure = connect_mat_1.copy()
        connect_mat_1_pure = connect_mat_1_pure.reshape(np.prod(dims), order="F")
        connect_mat_1_pure[~np.in1d(connect_mat_1_pure, pure_pix)] = 0
        connect_mat_1_pure = connect_mat_1_pure.reshape(dims, order="F")

        ax1.imshow(connect_mat_1_pure, cmap="nipy_spectral_r")

        for ii in range(len(pure_pix)):
            pos = np.where(connect_mat_1_pure[:, :] == pure_pix[ii])
            pos0 = pos[0]
            pos1 = pos[1]
            ax1.text((pos1)[np.array(len(pos1) / 3, dtype=int)], (pos0)[np.array(len(pos0) / 3, dtype=int)],
                     f"{brightness_rank[ii] + 1}",
                     verticalalignment='bottom', horizontalalignment='right', color='black',
                     fontsize=15)

        ax1.set(title="pass " + str(p + 1))
        ax1.title.set_fontsize(15)
        ax1.title.set_fontweight("bold")
    save_plot(name, path, show)


def plot_average(mov,name, path, show=True):
    plt.imshow(np.std(mov, axis=2))
    save_plot(name, path, show)

