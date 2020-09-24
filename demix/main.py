import time, os, itertools
import matplotlib.pyplot as plt
import numpy as np
import sys
import superpixel_analysis as sup
from skimage import io
from scipy.ndimage import center_of_mass, filters, gaussian_filter
from sklearn.decomposition import TruncatedSVD
import torch
import scipy.io
import util_plot
import pickle
from datetime import datetime
import demixing_plots as plots

# TODO: add arg parser
sys.path.append(os.getcwd())
data_path = str(sys.argv[1])
cut_off_point = float(sys.argv[2])
corr_th_fix = float(sys.argv[3])
patch_size_edge = int(sys.argv[4])
default_bg_rank = int(float(sys.argv[5]))
trunc_start = int(sys.argv[6])
window_length = int(sys.argv[7])
th_lvl = int(sys.argv[8])
pass_num = int(sys.argv[9])
bg_mask = bool(int(sys.argv[10]))
merge_corr_thr = float(sys.argv[11])
sup_only = bool(int(sys.argv[12]))
remove_dimmest = int(sys.argv[13])
residual_cut = float(sys.argv[14])
update_ac_maxiter = int(sys.argv[15])
update_ac_tol = float(sys.argv[16])
update_ac_merge_overlap_thr = float(sys.argv[17])
update_ac_keep_shape = bool(int(sys.argv[18]))
bg_reg_lr = float(sys.argv[19])
bg_reg_max_iterations = int(sys.argv[20])
demix_all_flag = bool(int(sys.argv[21]))
hp_spacing = int(sys.argv[22])
edge_trim = int(float(sys.argv[23]))
binning_flag = bool(int(sys.argv[24]))
nmf_cells = [int(idx) for idx in list(sys.argv[25])]
output_path = sys.argv[26]
find_rois_flag = bool(int(sys.argv[27]))
calculate_traces_flag = bool(int(sys.argv[28]))


plot_path = output_path + '/plots/'


NO_RLT_FILE_MSG = "rlt.tif could not be found. Please check that the correct output directory was selected and that " \
                  "you have already found the ROIs."


# TODO: add as params
suffix = ''
optopatch_stim = False


def remove_previous_outputs(path, suffix):
    files_to_remove = ['spatial_footprints', 'cell_spatial_footprints', 'temporal_traces', 'cell_traces',
                       'residual_var', 'ref_im', 'rlt', 'ref']
    for file in files_to_remove:
        full_path = path + "/" + file + suffix + '.tif'
        if os.path.exists(full_path):
            os.remove(full_path)


def load_movie(path):
    """
    Reads in motion corrected movie.
    """
    if os.path.isfile(path + '/motion_corrected.tif'):
        mov = io.imread(path + '/motion_corrected.tif').transpose(1, 2, 0)
    elif os.path.isfile(path + '/denoised.tif'):
        mov = io.imread(path + '/denoised.tif')
    else:
        raise ValueError('No valid input file')
    return mov


def add_blood_mask(mov, noise):
    if os.path.isfile(output_path + '/bloodmask.tif'):
        bloodmask = np.squeeze(io.imread(output_path + '/bloodmask.tif'))
        mov = mov * np.repeat(np.expand_dims(noise * bloodmask, 2), mov.shape[2], axis=2)
    else:
        mov = mov * np.repeat(np.expand_dims(noise, 2), mov.shape[2], axis=2)
    return mov


def bin_and_trim_movie(mov, binning_flag):
    if binning_flag:
        movB = mov.reshape(int(mov.shape[0] / 2), 2, int(mov.shape[1] / 2), 2, mov.shape[2])
        movB = np.mean(np.mean(movB, axis=1), axis=2)
    else:
        movB = mov
    movB = movB[edge_trim:-edge_trim, edge_trim:-edge_trim, :]
    return movB


def load_bg_mask(mov, mov_b, path):
    # TODO: Add bg support
    # import manually initialized background components
    ff_ini = io.imread(path + '/ff.tif')
    fb_ini = io.imread(path + '/fb.tif')

    # bin the spatial components
    fb_ini = fb_ini.reshape(mov.shape[1], mov.shape[0], -1).transpose(1, 0, 2)

    print(fb_ini.shape)

    plt.figure(figsize=(30, 20))

    for i in range(6):
        plt.subplot(3, 4, 2 * i + 1)
        plt.plot(ff_ini[:2000, i])
        plt.subplot(3, 4, 2 * i + 2)
        plt.imshow(fb_ini[:, :, i])
        plt.show()
    bkg_components = range(3)
    fb_ini = fb_ini[:, :, bkg_components].reshape(mov_b.shape[0] * mov_b.shape[1], len(bkg_components))
    ff_ini = ff_ini[:, bkg_components]
    return fb_ini, ff_ini


def get_valid_window(movie, start, length):
    total_frames = movie.shape[2]
    if start < 1 or demix_all_flag:
        start = 1
    if length > total_frames - start or demix_all_flag:
        length = total_frames - start - 10
    first_frame = start
    last_frame = start + length
    return first_frame, last_frame


def tv_norm(image):
    return np.sum(np.abs(image[:, :-1] - image[:, 1:])) + np.sum(np.abs(image[:-1, :] - image[1:, :]))


def get_cell_locations(X2, n_cells, mov_dims):
    cell_locations = center_of_mass(X2[:, 0].reshape(mov_dims).transpose(1, 0))
    for idx in range(n_cells - 1):
        cell_locations = np.vstack((cell_locations,
                                    center_of_mass(X2[:, idx + 1].reshape(mov_dims).transpose(1, 0))))
    return cell_locations


def get_cell_demixing_matrix(X2, n_cells):
    return np.linalg.inv(np.array(X2[:, :n_cells].T @ X2[:, :n_cells])) @ X2[:, :n_cells].T


def save_outputs(path, suffix, X2, n_cells, beta_hat2, res, ref_im, rlt, mov_dims):
    remove_previous_outputs(path, suffix)
    io.imsave(path + '/spatial_footprints' + suffix + '.tif', X2)
    io.imsave(path + '/cell_spatial_footprints' + suffix + '.tif', X2[:, :n_cells])
    io.imsave(path + '/temporal_traces' + suffix + '.tif', beta_hat2)
    io.imsave(path + '/cell_traces' + suffix + '.tif', beta_hat2[:n_cells, :])
    io.imsave(path + '/residual_var' + suffix + '.tif', res)
    with open(path + '/ref' + suffix + '.tif', 'wb') as f:
        pickle.dump([mov_dims, ref_im], f)
    with open(path + '/rlt' + suffix + '.tif', 'wb') as f:
        pickle.dump(rlt, f)
    io.imsave(path + '/cell_locations' + suffix + '.tif', np.array(get_cell_locations(X2, n_cells, mov_dims)))
    if n_cells > 1:
        io.imsave(path + '/cell_demixing_matrix' + suffix + '.tif', get_cell_demixing_matrix(X2, n_cells))
    print('Data files saved!')


def initialize_bg_parameters(mov_b, first_frame, last_frame):
    dims = mov_b.shape[:2]
    T = last_frame - first_frame
    movVec = mov_b.reshape(np.prod(dims), -1, order="F")
    mov_min = movVec.min()
    if mov_min < 0:
        mov_min_pw = movVec.min(axis=1, keepdims=True)
        movVec -= mov_min_pw
    normalize_factor = np.std(movVec, axis=1, keepdims=True) * T
    return dims, movVec, normalize_factor


def initialize_bg(a, movVec, first_frame, last_frame, bg_rank):
    bg_comp_pos = np.where(a.sum(axis=1) == 0)[0]
    y_temp = movVec[bg_comp_pos, first_frame:last_frame]
    fb = np.zeros([movVec.shape[0], bg_rank])
    y_temp = y_temp - y_temp.mean(axis=1, keepdims=True)
    svd = TruncatedSVD(n_components=bg_rank, n_iter=7, random_state=0)
    fb[bg_comp_pos, :] = svd.fit_transform(y_temp)
    ff = svd.components_.T
    ff = ff - ff.mean(axis=0, keepdims=True)
    return fb, ff


def load_optopatch_stim(mov_b, data_path, plot_path):
    # TODO: add edge trim and binning flag
    trend = io.imread(data_path + '/trend.tif')
    plt.imshow(np.mean(trend, axis=2))
    plots.save_plot('Reloaded_Trend', plot_path, show=False)
    trendB = trend.reshape(int(trend.shape[0] / 2), 2, int(trend.shape[1] / 2), 2, trend.shape[2])
    trendB = np.mean(np.mean(trendB, axis=1), axis=2)
    Y = (mov_b + trendB).transpose(1, 0, 2).reshape(mov_b.shape[0] * mov_b.shape[1], mov_b.shape[2])
    print('Trend reloaded!')
    return Y


def initialize_regression_params(mov_b, a, fb, n_cells, bg_rank):
    Y = mov_b.transpose(1, 0, 2).reshape(mov_b.shape[0] * mov_b.shape[1], mov_b.shape[2])
    X = np.hstack((a, fb))
    X = X / np.ptp(X, axis=0)
    X2 = np.zeros((X.shape[0], n_cells + bg_rank))
    X2[:, :n_cells] = X[:, :n_cells]
    return X, X2, Y


def bg_regression(bg_rank, X, mov_b, n_cells, max_iters, lr, X2, path):
    plt.figure(figsize=(25, 3 * bg_rank))
    plt.suptitle('New Background Components')
    for b in range(bg_rank):
        bg_im = X[:, -(b + 1)].reshape(mov_b.shape[-2::-1])
        plt.subplot(bg_rank, 2, (bg_rank - b) * 2 - 1)
        plt.imshow(bg_im)
        plt.title(str(tv_norm(bg_im)))
        plt.colorbar()
        weights = torch.zeros((n_cells,), requires_grad=True, dtype=torch.double)
        image = torch.from_numpy(bg_im)
        for idx in range(max_iters):
            test_im = image - torch.reshape(torch.from_numpy(X[:, :n_cells]) @ weights, mov_b.shape[-2::-1])
            tv = torch.sum(torch.abs(test_im[:, :-1] - test_im[:, 1:])) + torch.sum(
                torch.abs(test_im[:-1, :] - test_im[1:, :]))

            tv.backward()

            with torch.no_grad():
                weights -= lr * weights.grad

            weights.grad.zero_()

        opt_weights = weights.data.numpy()

        X2[:, -(b + 1)] = np.maximum(X[:, -(b + 1)] - np.squeeze(X[:, :n_cells] @ opt_weights), 0)

        plt.subplot(bg_rank, 2, (bg_rank - b) * 2)
        plt.imshow(X2[:, -(b + 1)].reshape(mov_b.shape[-2::-1]), vmin=0, vmax=1)
        plt.title(str(tv_norm(X2[:, -(b + 1)].reshape(mov_b.shape[-2::-1]).T)))
        plt.colorbar()
    plots.save_plot('Temporal_Correlations', path, show=False)


def find_optimal_traces(X2, Y):
    beta_hat2 = np.linalg.lstsq(X2, Y)[0]
    res = np.mean(np.square(Y - X2 @ beta_hat2), axis=0)
    return beta_hat2, res


def find_rois(mov_b, first_frame, last_frame, ref_im, mov_dims, path):
    start = time.time()
    mov_hp = sup.hp_filt_data(mov_b, spacing=hp_spacing)
    rlt = sup.axon_pipeline_Y(mov_hp[:, :, first_frame:last_frame].copy(), fb_ini=np.zeros(1), ff_ini=np.zeros(1),

                              ##### Superpixel parameters
                              # thresholding level
                              th=[th_lvl],

                              # correlation threshold for finding superpixels
                              # (range around 0.8-0.99)
                              cut_off_point=[cut_off_point],

                              # minimum pixel count of a superpixel
                              # don't need to change these unless cell sizes change
                              length_cut=[int(patch_size_edge ** 2 / 5)],

                              # maximum pixel count of a superpixel
                              # don't need to change these unless cell sizes change
                              length_max=[patch_size_edge ** 2 * 2],

                              patch_size=[patch_size_edge, patch_size_edge],

                              # correlation threshold between superpixels for merging
                              # likely don't need to change this
                              residual_cut=[residual_cut],

                              pass_num=pass_num,

                              bg=bg_mask,

                              ##### Cell-finding, NMF parameters
                              # correlation threshold of pixel with superpixel trace to include pixel in cell
                              # (range 0.3-0.6)
                              corr_th_fix=corr_th_fix,

                              # correlation threshold for merging two cells
                              # (default 0.8, but likely don't need to change)
                              merge_corr_thr=merge_corr_thr,

                              ##### Other options
                              # if True, only superpixel analysis run; if False, NMF is also run to find cells
                              sup_only=sup_only,

                              # the number of superpixels to remove (starting from the dimmest)
                              remove=remove_dimmest
                              )

    print("Finding the super pixels took: " + str(time.time() - start) + " sec")

    plots.plot_super_pixels(rlt, ref_im, "super_pixels", path, show=False)
    plots.plot_nmf_traces(rlt, ref_im, mov_dims, "NMF_Traces", path, show=False)

    with open(path + '/rlt' + suffix + '.tif', 'wb') as f:
        pickle.dump(rlt, f)

    return rlt


def calculate_traces(rlt, mov_b, bg_flag, first_frame, last_frame, mov, ref_im, mov_dims):
    final_cells = nmf_cells

    n_cells = len(final_cells)

    a = rlt["fin_rlt"]["a"][:, final_cells].copy()
    c = rlt["fin_rlt"]["c"][:, final_cells].copy()
    b = rlt["fin_rlt"]["b"].copy()

    dims, movVec, normalize_factor = initialize_bg_parameters(mov_b, first_frame, last_frame)
    if bg_flag:
        fb_ini, ff_ini = load_bg_mask(mov, mov_b, output_path)
        fb = fb_ini
        ff = ff_ini[first_frame:last_frame, :]
        bg_rank = fb.shape[1]
    else:
        bg_rank = default_bg_rank
        fb, ff = initialize_bg(a, movVec, first_frame, last_frame, bg_rank)

    a, c, b, fb, ff, res, corr_img_all_r, num_list = sup.update_AC_bg_l2_Y(movVec[:, first_frame:last_frame].copy(),
                                                                           normalize_factor, a, c, b, ff, fb, dims,
                                                                           corr_th_fix=corr_th_fix,
                                                                           maxiter=update_ac_maxiter,
                                                                           tol=update_ac_tol,
                                                                           merge_corr_thr=merge_corr_thr,
                                                                           merge_overlap_thr=update_ac_merge_overlap_thr,
                                                                           keep_shape=update_ac_keep_shape
                                                                           )

    plots.plot_intermediate_traces(a, c, ref_im, mov_dims, "Intermediate_Traces", plot_path, show=False)
    plots.plot_bg_traces(fb, ff, mov_dims, "BG_Traces", plot_path, show=False)

    X, X2, Y = initialize_regression_params(mov_b, a, fb, n_cells, bg_rank)

    bg_regression(bg_rank, X, mov_b, n_cells, bg_reg_max_iterations, bg_reg_lr, X2, plot_path)

    if optopatch_stim:
        load_optopatch_stim(mov_b, data_path, plot_path)

    beta_hat2, res = find_optimal_traces(X2, Y)

    plots.plot_final_traces(beta_hat2, ref_im, mov_dims, X2, "Traces", plot_path, show=False)
    save_outputs(plot_path, suffix, X2, n_cells, beta_hat2, res, ref_im, rlt, mov_dims)


def _initialize_params():
    noise = np.squeeze(io.imread(output_path + '/Sn_image.tif'))
    mov = load_movie(output_path)
    print("Movie dimensions: " + str(mov.shape))
    mov = add_blood_mask(mov, noise)
    plots.plot_average(mov, 'Average', plot_path, show=False)
    mov_b = bin_and_trim_movie(mov, binning_flag)
    print("Binned and trimmed movie dimensions:" + str(mov_b.shape))
    plots.plot_average(mov_b, "Average_Binned_Movie", plot_path, show=False)
    first_frame, last_frame = get_valid_window(mov_b, trunc_start, window_length)
    bg_flag = os.path.isfile(output_path + '/ff.tif')
    ref_im = np.std(mov_b, axis=2).transpose(1, 0)
    mov_dims = mov_b.shape[1::-1]
    return noise, mov, mov_b, first_frame, last_frame, bg_flag, ref_im, mov_dims


def main():

    noise, mov, mov_b, first_frame, last_frame, bg_flag, ref_im, mov_dims = _initialize_params()

    if find_rois_flag:
        rlt = find_rois(mov_b, first_frame, last_frame, ref_im, mov_dims, plot_path)

    elif calculate_traces_flag:
        rlt_path = plot_path + "rlt.tif"
        if os.path.isfile(rlt_path):
            with open(rlt_path, 'rb') as f:
                rlt = pickle.load(f)
        else:
            raise FileNotFoundError(NO_RLT_FILE_MSG)

    if calculate_traces_flag:
        calculate_traces(rlt, mov_b, bg_flag, first_frame, last_frame, mov, ref_im, mov_dims)


if __name__ == "__main__":
    print("Demixing Start")
    print(str(sys.argv))
    main()
