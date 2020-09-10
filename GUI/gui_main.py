import PySimpleGUI as sg
import os
import PIL
from PIL import Image
import io
import base64
from skimage import io as skio
import numpy as np
from matplotlib import pyplot as plt
import subprocess

IM_SIZE = (800, 600)
sg.theme('Reddit')
CHECK_BOX_SIZE = (25, 1)
INPUT_SIZE = (10, 1)
LABEL_SIZE = (25, 1)
SLIDER_SIZE = (34, 20)


def convert_to_bytes(file_or_bytes, resize=None):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def load_picture_on_canvas(values, graph, im_name):
    im_full = os.path.join(values['output_dir'], "plots", im_name)
    print(im_full)
    if os.path.exists(im_full):
        im_bin = convert_to_bytes(im_full, IM_SIZE)
        graph.delete_figure("all")
        graph.draw_image(data=im_bin, location=(0, IM_SIZE[1]))
        print(im_name + " loaded")
    else:
        print("Still running")


def open_traces_plot(values, voltage_file, footprint_file, ref_file):
    voltage_full = os.path.join(values['output_dir'], voltage_file)
    footprint_full = os.path.join(values['output_dir'], footprint_file)
    ref_full = os.path.join(values['output_dir'], ref_file)

    if os.path.exists(voltage_full) and os.path.exists(ref_full) and os.path.exists(footprint_full):
        beta_hat2 = skio.imread(voltage_full)
        X2 = skio.imread(footprint_full)
        movB = skio.imread(ref_full)

        num_traces = beta_hat2.shape[0]
        fig = plt.figure(figsize=(25, 3 * num_traces))
        ref_im = np.std(movB, axis=2).transpose(1, 0)

        for idx in range(num_traces):
            fig.subplot(num_traces, 2, 2 * idx + 1)
            fig.plot(beta_hat2[idx, :])

            fig.subplot(num_traces, 2, 2 * idx + 2)
            lower, upper = np.percentile(ref_im.flatten(), [1, 99])
            fig.imshow(ref_im, cmap='gray', interpolation='none', clim=[lower, upper])

            cell_loc = X2[:, idx].reshape(movB.shape[1::-1])
            cell_loc = np.ma.masked_where(abs(cell_loc) < 1e-8, cell_loc)
            fig.imshow(cell_loc, cmap='jet', alpha=0.5)
        """
        traces = skio.imread(voltage_full)
        elem_num, frame_num = traces.shape
        sample_frequency = float(values['sample_freq'])
        t = np.linspace(0, frame_num/sample_frequency, frame_num)#.reshape((1, frame_num))
        fig, axs = plt.subplots(elem_num, sharex=True, gridspec_kw={'hspace': 0.1}, squeeze=False)
        fig.suptitle('Temporal traces')
        for i in range(elem_num):
            multiplier = 5
            voltage = traces[i, :]
            max_v = max(np.abs(voltage)) * multiplier
            axs[i, 0].plot(t, voltage, linewidth=0.7)
            axs[i, 0].set_ylim(-max_v, max_v)
        """
        fig.show()
        print(voltage_file + " plotted")
    else:
        print("Still running")


def enforce_numbers(window, values, key):
    if values[key] != "":
        if values[key] and values[key][-1] not in ('0123456789.-e'):
            window[key].update(values[key][:-1])


def _bool_to_words(flag):
    if flag:
        return "YES"
    else:
        return "NO"


def get_args_array(values):
    args = [None]*32
    args[0] = "sbatch full_data_bash.sh"
    args[1] = values['input_file']
    args[2] = values['output_dir']
    args[3] = values['normcorre']
    args[4] = values['detrend']
    args[5] = values['moco']
    args[6] = values['demix']
    args[7] = values['cut_off_point']/100
    args[8] = values['corr_th_fix']/100
    args[9] = values['patch_size_edge']
    args[10] = values['bg_rank']
    args[11] = values['trunc_start']
    args[12] = values['trunc_length']
    args[13] = values['output_dir']
    args[14] = values['mov_in']
    args[15] = values['detr_spacing']
    args[16] = values['row_blocks']
    args[17] = values['col_blocks']
    args[18] = values['residual_cut']
    args[19] = values['th_lvl']
    args[20] = values['pass_num']
    args[21] = values['bg_mask']
    args[22] = values['merge_corr_thr']
    args[23] = values['sup_only']
    args[24] = values['remove_dimmest']
    args[25] = values['update_ac_keep_shape']
    args[26] = values['update_ac_max_iter']
    args[27] = values['update_ac_tol']
    args[28] = values['update_ac_merge_overlap_thr']
    args[29] = values['bg_reg_max_iter']
    args[30] = values['bg_reg_lr']
    args[31] = values['stim_dir']
    for i in range(len(args)):
        if type(args[i]) == bool:
            args[i] = int(args[i])
    return args


def run_command(values):
    ssh_line = "sshpass -p {} ssh -o StrictHostKeyChecking=no rotem.ovadia@bs-cluster.elsc.huji.ac.il \"{}\""
    running_line = " ".join([str(arg) for arg in get_args_array(values)])
    ssh_line = ssh_line.format(values['password'], running_line)
    subprocess.check_call(['ubuntu1804', 'run', ssh_line])
    return ssh_line


def main():
    height = 1
    main_runner = [
        [sg.Text('Movie file:', size=LABEL_SIZE), sg.InputText(key='input_file'), sg.FileBrowse()],
        [sg.Text('Output directory:', size=LABEL_SIZE), sg.InputText(key='output_dir'), sg.FolderBrowse()],
        [sg.Text('Cluster password', size=LABEL_SIZE), sg.InputText('', key='password', password_char='*')],
        [sg.Checkbox('NoRMCoRRe', size=CHECK_BOX_SIZE, default=True, key="normcorre")],
        [sg.Checkbox('Detrending', size=CHECK_BOX_SIZE, default=True, key="detrend")],
        [sg.Checkbox('Motion correction', size=CHECK_BOX_SIZE, default=True, key="moco")],
        [sg.Checkbox('Demixing', size=CHECK_BOX_SIZE, default=True, key="demix")],
        [sg.Checkbox('Quick run', size=CHECK_BOX_SIZE, default=False, key="sup_only")],
        [sg.Text('Cut off point %', size=LABEL_SIZE), sg.Slider(range=(80, 99), orientation='h', size=SLIDER_SIZE, key='cut_off_point', default_value=90)],
        [sg.Text('Correlation threshold fix %', size=LABEL_SIZE), sg.Slider(range=(30, 60), orientation='h', size=SLIDER_SIZE, key='corr_th_fix', default_value=45)],
        [sg.Text('Start frame', size=LABEL_SIZE), sg.In(default_text='1', size=INPUT_SIZE, key='trunc_start', enable_events=True), sg.Text('Number of frames', size=(25, height)), sg.In(default_text='5000', size=(5, height), key='trunc_length', enable_events=True)],
        [sg.Text('Cell diameter', size=LABEL_SIZE), sg.In(default_text='10', size=INPUT_SIZE, key='patch_size_edge', enable_events=True)],
        [sg.Text('Sample frequency[Hz]', size=LABEL_SIZE), sg.In(default_text='1000', size=INPUT_SIZE, key='sample_freq', enable_events=True)],
    ]

    advanced_params = [
        [sg.Text('Advanced Parameters')],
        [sg.Text('# bg elements', size=LABEL_SIZE), sg.In(default_text='4', size=INPUT_SIZE, key='bg_rank', enable_events=True)],
        [sg.Text('Detrend spacing', size=LABEL_SIZE), sg.In(default_text='5000', size=INPUT_SIZE, key='detr_spacing', enable_events=True)],
        [sg.Text('Row blocks', size=LABEL_SIZE), sg.In(default_text='4', size=INPUT_SIZE, key='row_blocks', enable_events=True)],
        [sg.Text('Column blocks', size=LABEL_SIZE), sg.In(default_text='2', size=INPUT_SIZE, key='col_blocks', enable_events=True)],
        [sg.Text('Threshold level', size=LABEL_SIZE), sg.In(default_text='4', size=INPUT_SIZE, key='th_lvl', enable_events=True)],
        [sg.Text('# Passes', size=LABEL_SIZE), sg.In(default_text='1', size=INPUT_SIZE, key='pass_num', enable_events=True)],
        [sg.Text('Merge correlation threshold', size=LABEL_SIZE), sg.In(default_text='0.8', size=INPUT_SIZE, key='merge_corr_thr', enable_events=True)],
        [sg.Text('Remove dimmest ', size=LABEL_SIZE), sg.In(default_text='0', size=INPUT_SIZE, key='remove_dimmest', enable_events=True)],
        [sg.Text('Residual cut', size=LABEL_SIZE), sg.In(default_text='0.6', size=INPUT_SIZE, key='residual_cut', enable_events=True)],
        [sg.Text('UAC max iterations', size=LABEL_SIZE), sg.In(default_text='35', size=INPUT_SIZE, key='update_ac_max_iter', enable_events=True)],
        [sg.Text('UAC tol', size=LABEL_SIZE), sg.In(default_text='1e-8', size=INPUT_SIZE, key='update_ac_tol', enable_events=True)],
        [sg.Text('UAC merge overlap threshold', size=LABEL_SIZE), sg.In(default_text='0.8', size=INPUT_SIZE, key='update_ac_merge_overlap_thr', enable_events=True)],
        [sg.Checkbox('UAC keep shape', size=CHECK_BOX_SIZE, default=True, key="update_ac_keep_shape")],
        [sg.Text('BGR learning rate', size=LABEL_SIZE), sg.In(default_text='0.001', size=INPUT_SIZE, key='bg_reg_lr', enable_events=True)],
        [sg.Text('BGR max iterations', size=LABEL_SIZE), sg.In(default_text='1000', size=INPUT_SIZE, key='bg_reg_max_iter', enable_events=True)],
        [sg.Text('Registered movie name', size=LABEL_SIZE), sg.InputText(key='mov_in', default_text='movReg.tif')],
        [sg.Text('Stimulation dir: ', size=LABEL_SIZE), sg.InputText(key='stim_dir'), sg.FolderBrowse()],
        [sg.Checkbox('Background mask', size=CHECK_BOX_SIZE, default=False, key="bg_mask")],
        [sg.Text('Min cell area (pix)', size=LABEL_SIZE), sg.In(default_text='10', size=INPUT_SIZE, key='min_size', enable_events=True),
         sg.Text('Max cell area (pix)', size=LABEL_SIZE), sg.In(default_text='1000', size=INPUT_SIZE, key='max_size', enable_events=True)],
    ]

    nmf_traces_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                enable_events=True, key='nmf_traces_graph')

    nmf_traces = [
        [sg.Text('Choose the ones that look like cells:')],
        [nmf_traces_graph]
    ]

    super_pixels_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                  enable_events=True, key='super_pixels_graph')

    super_pixels = [
        [sg.Text('Make sure the super pixels look like the cells or the background')],
        [super_pixels_graph]
    ]

    final_traces_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                  enable_events=True, key='final_traces_graph')

    final_traces = [
        [sg.Text('These are the final traces of the cells you chose:')],
        [final_traces_graph],
        [sg.Button('Open zoomable plot')]
    ]

    other_plots_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                 enable_events=True, key='Other_plots_graph')

    other_plots = [
        [sg.Text('Other plots')],
        [other_plots_graph],
        [sg.Button('Average'), sg.Button('Background Traces'), sg.Button('Intermediate Traces'), sg.Button('Temporal Correlations')]
    ]


    # The TabgGroup layout - it must contain only Tabs
    tab_group_layout = [[
        sg.Tab('Main Runner', main_runner, font='Courier 15', key='tab_main_runner'),
        sg.Tab('Advanced Parameters', advanced_params, key='tab_advanced_parameters'),
        sg.Tab('NMF Traces', nmf_traces, key='tab_nmf_traces'),
        sg.Tab('Super Pixels', super_pixels, key='tab_super_pixels'),
        sg.Tab('Final Traces', final_traces, key='tab_final_traces'),
        sg.Tab('Other Plots', other_plots, key='tab_outputs'),
        ]]

    # The window layout - defines the entire window
    layout = [
        [sg.TabGroup(tab_group_layout, enable_events=True, key='-TABGROUP-')],
        [sg.Button('Run'), sg.Button('Help'), sg.Button('Load outputs'), sg.Button('Quit')]
        ]

    window = sg.Window('Invivo imaging - Adam Lab - ver0.0', layout, no_titlebar=False)

    while True:
        event, values = window.read()
        print(event, values)
        if event == sg.WIN_CLOSED or event == 'Quit':
            break
        if event == 'Run':
            print(run_command(values))
        if event == 'Help':
            print("Link to github appeared")
        if event == 'Load outputs':
            print('Loading outputs')
            load_picture_on_canvas(values, nmf_traces_graph, 'NMF_Traces.png')
            load_picture_on_canvas(values, super_pixels_graph, 'super_pixels.png')
            load_picture_on_canvas(values, final_traces_graph, 'Traces.png')
            load_picture_on_canvas(values, other_plots_graph, 'Average.png')
        if event == 'Average':
            load_picture_on_canvas(values, other_plots_graph, 'Average.png')
        if event == 'Intermediate Traces':
            load_picture_on_canvas(values, other_plots_graph, 'Intermediate_Traces.png')
        if event == 'Background Traces':
            load_picture_on_canvas(values, other_plots_graph, 'BG_Traces.png')
        if event == 'Temporal Correlations':
            load_picture_on_canvas(values, other_plots_graph, 'Temporal_Correlations.png')
        if event == 'Open zoomable plot':
            open_traces_plot(values, 'temporal_traces.tif', 'spatial_footprints.tif', 'ref_im.tif')
        number_field_keys = ['patch_size_edge', 'trunc_start', 'trunc_length', 'min_size', 'max_size',
                             'sample_freq', 'bg_rank', 'detr_spacing', 'row_blocks', 'col_blocks',
                             'th_lvl', 'pass_num', 'merge_corr_th', 'remove_dimmest', 'residual_cut',
                             'update_ac_max_iter', 'update_ac_tol', 'update_ac_merge_overlap_thr',
                             'bg_reg_lr', 'bg_reg_max_iter']
        for key in number_field_keys:
            if event == key:
                enforce_numbers(window, values, key)

    window.close()


if __name__ == "__main__":
    main()
