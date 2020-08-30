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

sys.path.append(os.getcwd())

# ## Read in movie
data_dir = sys.argv[1]
cut_off_point = float(sys.argv[2])
corr_th_fix = float(sys.argv[3])
patch_size_edge = float(sys.argv[4])

print("Demixing Start")
print(data_dir)

# In[3]:
GUI = False
OPTOPATCH_STIM = False
PLOT_PATH = data_dir + '/plots/'
# input movie path
path = data_dir + '/output'


def save_plot(name):
    plt.savefig(PLOT_PATH + name + '.png')
    print("Plot saved: " + name + ".png")
    if GUI:
        plt.show()


def ask_proceed():
    proc = input('Ok to proceed? (y/n) ')
    if proc.lower() == 'n':
        raise KeyboardInterrupt('Stop requested by user.')


# read in motion corrected movie
noise = np.squeeze(io.imread(path + '/Sn_image.tif'))
[nrows, ncols] = noise.shape

if os.path.isfile(path + '/motion_corrected.tif'):
    mov = io.imread(path + '/motion_corrected.tif').transpose(1, 2, 0)
elif os.path.isfile(path + '/denoised.tif'):
    mov = io.imread(path + '/denoised.tif')
else:
    raise ValueError('No valid input file')

# read in the mask for blood
if os.path.isfile(path + '/bloodmask.tif'):
    bloodmask = np.squeeze(io.imread(path + '/bloodmask.tif'))
    mov = mov * np.repeat(np.expand_dims(noise * bloodmask, 2), mov.shape[2], axis=2)
else:
    mov = mov * np.repeat(np.expand_dims(noise, 2), mov.shape[2], axis=2)

plt.imshow(np.std(mov, axis=2))
save_plot('Average')
# display average movie


# ## Spatial 2x2 Binning

# In[4]:


movB = mov.reshape(int(mov.shape[0] / 2), 2, int(mov.shape[1] / 2), 2, mov.shape[2])
movB = np.mean(np.mean(movB, axis=1), axis=2)
movB.shape

movB = mov

plt.imshow(np.std(movB, axis=2))

save_plot('Average_Binned_Movie')
# show standard deviation image of binned movie


# ## Load Manually Initialized Background

# In[5]:

# TODO: solve bg output
bg_flag = os.path.isfile(path + '/ff.tif')

if bg_flag:
    # import manually initialized background components
    ff_ini = io.imread(path + '/ff.tif')
    fb_ini = io.imread(path + '/fb.tif')

    # bin the spatial components
    fb_ini = fb_ini.reshape(mov.shape[1], mov.shape[0], -1).transpose(1, 0, 2)
    #     fb_ini = fb_ini.reshape(int(fb_ini.shape[0]/2),2,int(fb_ini.shape[1]/2),2,fb_ini.shape[2])
    #     fb_ini = np.mean(np.mean(fb_ini,axis=1),axis=2)

    fb_ini.shape

    # plot manually initialized background components
    plt.figure(figsize=(30, 20))

    for i in range(6):
        plt.subplot(3, 4, 2 * i + 1)
        plt.plot(ff_ini[:2000, i])
        plt.subplot(3, 4, 2 * i + 2)
        plt.imshow(fb_ini[:, :, i])
        plt.show()

# In[6]:


if bg_flag:
    # select which background components to use for initialization
    bkg_components = range(3)

    fb_ini = fb_ini[:, :, bkg_components].reshape(movB.shape[0] * movB.shape[1], len(bkg_components))
    ff_ini = ff_ini[:, bkg_components]

# ## Get Cell Spatial Supports from High Pass Filtered Movie

# In[7]:


start = time.time()

# select which window to demix on
first_frame = 1
last_frame = 5000

movHP = sup.hp_filt_data(movB, spacing=10)

rlt = sup.axon_pipeline_Y(movHP[:, :, first_frame:last_frame].copy(), fb_ini=np.zeros(1), ff_ini=np.zeros(1),

                          ##### Superpixel parameters
                          # thresholding level
                          th=[4],

                          # correlation threshold for finding superpixels
                          # (range around 0.8-0.99)
                          cut_off_point=[cut_off_point],

                          # minimum pixel count of a superpixel
                          # don't need to change these unless cell sizes change
                          length_cut=[10],

                          # maximum pixel count of a superpixel
                          # don't need to change these unless cell sizes change
                          length_max=[1000],

                          patch_size=[patch_size_edge, patch_size_edge],

                          # correlation threshold between superpixels for merging
                          # likely don't need to change this
                          residual_cut=[np.sqrt(1 - (0.8) ** 2)],

                          pass_num=1, bg=False,

                          ##### Cell-finding, NMF parameters
                          # correlation threshold of pixel with superpixel trace to include pixel in cell
                          # (range 0.3-0.6)
                          corr_th_fix=corr_th_fix,

                          # correlation threshold for merging two cells
                          # (default 0.8, but likely don't need to change)
                          merge_corr_thr=0.8,

                          ##### Other options
                          # if True, only superpixel analysis run; if False, NMF is also run to find cells
                          sup_only=False,

                          # the number of superpixels to remove (starting from the dimmest)
                          remove=0
                          )

print("Demixing took: " + str(time.time() - start) + " sec")

# In[9]:


# plot pure superpixels

num_pass = len(rlt["superpixel_rlt"])

scale = np.maximum(1, (
        rlt["superpixel_rlt"][0]["connect_mat_1"].shape[1] / rlt["superpixel_rlt"][0]["connect_mat_1"].shape[0]));
plt.figure(figsize=(4 * scale * num_pass, 4))

plt.subplot(1, num_pass + 2, 1)
plt.imshow(np.std(movB, axis=2))

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
        """ax1.text((pos1)[np.array(len(pos1) / 3, dtype=int)], (pos0)[np.array(len(pos0) / 3, dtype=int)],
                 verticalalignment='bottom', horizontalalignment='right', color='black',
                 fontsize=15)"""
        ax1.text((pos1)[np.array(len(pos1) / 3, dtype=int)], (pos0)[np.array(len(pos0) / 3, dtype=int)],
                 f"{brightness_rank[ii] + 1}",
                 verticalalignment='bottom', horizontalalignment='right', color='black',
                 fontsize=15)  # , fontweight="bold")

    ax1.set(title="pass " + str(p + 1))
    ax1.title.set_fontsize(15)
    ax1.title.set_fontweight("bold")
    plt.tight_layout()
    save_plot('super_pixels')

# In[19]:


# plot all cell traces and footprints from NMF
cell_ct = rlt["fin_rlt"]["c"].shape[1]

plt.figure(figsize=(25, 3 * cell_ct))

ref_im = np.std(movB, axis=2).transpose(1, 0)

for cell_num in range(cell_ct):
    plt.subplot(cell_ct, 2, 2 * cell_num + 1)
    plt.plot(rlt["fin_rlt"]["c"][:, cell_num])
    plt.title(cell_num, size=24)

    plt.subplot(cell_ct, 2, 2 * cell_num + 2)
    lower, upper = np.percentile(ref_im.flatten(), [1, 99])
    plt.imshow(ref_im, cmap='gray', interpolation='none', clim=[lower, upper])

    cell_loc = rlt["fin_rlt"]["a"][:, cell_num].reshape(movB.shape[1], movB.shape[0])  # .transpose(1,0)
    cell_loc = np.ma.masked_where(cell_loc == 0, cell_loc)
    plt.imshow(cell_loc, cmap='jet', alpha=0.5)

save_plot('NMF_Traces')

# In[11]:

if GUI:
    ask_proceed()

# ## Get Background Components from Unfiltered Movie

# In[21]:

# TODO: Add input from user for good / bad cells

# rank of background to model, if none selected
bg_rank = 2
final_cells = [0, 1]

nCells = len(final_cells)

a = rlt["fin_rlt"]["a"][:, final_cells].copy()
c = rlt["fin_rlt"]["c"][:, final_cells].copy()
b = rlt["fin_rlt"]["b"].copy()

dims = movB.shape[:2]
T = last_frame - first_frame

movVec = movB.reshape(np.prod(dims), -1, order="F")
mov_min = movVec.min()
if mov_min < 0:
    mov_min_pw = movVec.min(axis=1, keepdims=True)
    movVec -= mov_min_pw

normalize_factor = np.std(movVec, axis=1, keepdims=True) * T

if bg_flag:
    fb = fb_ini
    ff = ff_ini[first_frame:last_frame, :]
    bg_rank = fb.shape[1]
else:
    bg_comp_pos = np.where(a.sum(axis=1) == 0)[0]
    y_temp = movVec[bg_comp_pos, first_frame:last_frame]
    fb = np.zeros([movVec.shape[0], bg_rank])
    y_temp = y_temp - y_temp.mean(axis=1, keepdims=True)
    svd = TruncatedSVD(n_components=bg_rank, n_iter=7, random_state=0)
    fb[bg_comp_pos, :] = svd.fit_transform(y_temp)
    ff = svd.components_.T
    ff = ff - ff.mean(axis=0, keepdims=True)

a, c, b, fb, ff, res, corr_img_all_r, num_list = sup.update_AC_bg_l2_Y(movVec[:, first_frame:last_frame].copy(),
                                                                       normalize_factor, a, c, b, ff, fb, dims,
                                                                       corr_th_fix=0.35,
                                                                       maxiter=35, tol=1e-8,
                                                                       merge_corr_thr=0.8,
                                                                       merge_overlap_thr=0.8, keep_shape=True
                                                                       )

# In[22]:


# plot all cell traces and footprints
cell_ct = c.shape[1]

plt.figure(figsize=(25, 3 * cell_ct))

ref_im = np.std(movB, axis=2).transpose(1, 0)

for cell_num in range(cell_ct):
    plt.subplot(cell_ct, 2, 2 * cell_num + 1)
    plt.plot(c[:, cell_num])
    plt.title(cell_num, size=16)
    plt.subplot(cell_ct, 2, 2 * cell_num + 2)
    lower, upper = np.percentile(ref_im.flatten(), [1, 99])
    plt.imshow(ref_im, cmap='gray', interpolation='none', clim=[lower, upper])
    cell_loc = a[:, cell_num].reshape(movB.shape[1], movB.shape[0])  # .transpose(1,0)
    cell_loc = np.ma.masked_where(cell_loc == 0, cell_loc)
    plt.imshow(cell_loc, cmap='jet', alpha=0.5)
    plt.colorbar()
save_plot('Intermediate_Traces')

# In[23]:


# plot all background traces and footprints
bg_rank = fb.shape[1]

plt.figure(figsize=(25, 3 * bg_rank))

for bkgd_num in range(bg_rank):
    plt.subplot(bg_rank, 2, 2 * bkgd_num + 1)
    plt.plot(ff[:, bkgd_num])

    bkgd_comp = fb[:, bkgd_num].reshape(movB.shape[1::-1])  # .transpose(1,0)
    plt.subplot(bg_rank, 2, 2 * bkgd_num + 2)
    plt.imshow(bkgd_comp)
    plt.colorbar()
save_plot('BG_Traces')

# In[24]:


if GUI:
    ask_proceed()


# ## Choose Cells and Recover Temporal Correlation Structures

# In[25]:


def tv_norm(image):
    return np.sum(np.abs(image[:, :-1] - image[:, 1:])) + np.sum(np.abs(image[:-1, :] - image[1:, :]))


Y = movB.transpose(1, 0, 2).reshape(movB.shape[0] * movB.shape[1], movB.shape[2])
X = np.hstack((a, fb))
X = X / np.ptp(X, axis=0)
X2 = np.zeros((X.shape[0], nCells + bg_rank))
X2[:, :nCells] = X[:, :nCells]

plt.figure(figsize=(25, 3 * bg_rank))
plt.title('New Background Components')

lr = 0.001
maxIters = 1000

for b in range(bg_rank):
    bg_im = X[:, -(b + 1)].reshape(movB.shape[-2::-1])

    plt.subplot(bg_rank, 2, (bg_rank - b) * 2 - 1)
    plt.imshow(bg_im)
    plt.title(str(tv_norm(bg_im)))
    plt.colorbar()

    weights = torch.zeros((nCells,), requires_grad=True, dtype=torch.double)

    image = torch.from_numpy(bg_im)

    for idx in range(maxIters):
        test_im = image - torch.reshape(torch.from_numpy(X[:, :nCells]) @ weights, movB.shape[-2::-1])
        tv = torch.sum(torch.abs(test_im[:, :-1] - test_im[:, 1:])) + torch.sum(
            torch.abs(test_im[:-1, :] - test_im[1:, :]))

        tv.backward()

        with torch.no_grad():
            weights -= lr * weights.grad

        weights.grad.zero_()

    opt_weights = weights.data.numpy()

    X2[:, -(b + 1)] = np.maximum(X[:, -(b + 1)] - np.squeeze(X[:, :nCells] @ opt_weights), 0)

    plt.subplot(bg_rank, 2, (bg_rank - b) * 2)
    plt.imshow(X2[:, -(b + 1)].reshape(movB.shape[-2::-1]), vmin=0, vmax=1)
    plt.title(str(tv_norm(X2[:, -(b + 1)].reshape(movB.shape[-2::-1]).T)))
    plt.colorbar()
save_plot('Temporal_Correlations')

# ## Reload Trend (Skip if not using Optopatch Stim)

# In[26]:

if OPTOPATCH_STIM:
    if GUI:
        proc = input('Do you want to load a trend? (y/n) ')
    else:
        proc = 'y'
    if proc.lower() == 'y':
        trend = io.imread(path + '/trend.tif')
        plt.imshow(np.mean(trend, axis=2))
        trendB = trend.reshape(int(trend.shape[0] / 2), 2, int(trend.shape[1] / 2), 2, trend.shape[2])
        trendB = np.mean(np.mean(trendB, axis=1), axis=2)
        trendB.shape

        Y = (movB + trendB).transpose(1, 0, 2).reshape(movB.shape[0] * movB.shape[1], movB.shape[2])
        print('Trend reloaded!')

# ## Get Final Traces

# In[27]:


beta_hat2 = np.linalg.lstsq(X2, Y)[0]
res = np.mean(np.square(Y - X2 @ beta_hat2), axis=0)

# ## Visualizations

# In[28]:


num_traces = beta_hat2.shape[0]
plt.figure(figsize=(25, 3 * num_traces))
ref_im = np.std(movB, axis=2).transpose(1, 0)

for idx in range(num_traces):
    plt.subplot(num_traces, 2, 2 * idx + 1)
    plt.plot(beta_hat2[idx, :])

    plt.subplot(num_traces, 2, 2 * idx + 2)
    lower, upper = np.percentile(ref_im.flatten(), [1, 99])
    plt.imshow(ref_im, cmap='gray', interpolation='none', clim=[lower, upper])

    cell_loc = X2[:, idx].reshape(movB.shape[1::-1])  # .transpose(1,0)
    cell_loc = np.ma.masked_where(abs(cell_loc) < 1e-8, cell_loc)
    plt.imshow(cell_loc, cmap='jet', alpha=0.5)
save_plot('Traces')

# ## Save Results

# In[29]:

if GUI:
    proc = input('Do you want to save out results? (y/n) ')
else:
    proc = 'y'

if proc.lower() == 'y':
    suffix = ''

    io.imsave(path + '/spatial_footprints' + suffix + '.tif', X2)
    io.imsave(path + '/cell_spatial_footprints' + suffix + '.tif', X2[:, :nCells])
    io.imsave(path + '/temporal_traces' + suffix + '.tif', beta_hat2)
    io.imsave(path + '/cell_traces' + suffix + '.tif', beta_hat2[:nCells, :])
    io.imsave(path + '/residual_var' + suffix + '.tif', res)

    cell_locations = center_of_mass(X2[:, 0].reshape(movB.shape[1::-1]).transpose(1, 0))
    for idx in range(nCells - 1):
        cell_locations = np.vstack((cell_locations,
                                    center_of_mass(X2[:, idx + 1].reshape(movB.shape[1::-1]).transpose(1, 0))))
    io.imsave(path + '/cell_locations' + suffix + '.tif', np.array(cell_locations))

    if nCells > 1:
        io.imsave(path + '/cell_demixing_matrix' + suffix + '.tif',
                  np.linalg.inv(np.array(X2[:, :nCells].T @ X2[:, :nCells])) @ X2[:, :nCells].T)
    print('Saved!')
