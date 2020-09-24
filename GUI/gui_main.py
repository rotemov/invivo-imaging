import PySimpleGUI as sg
import os
import PIL
from PIL import Image
import io
import base64
from skimage import io as skio
import subprocess
from subprocess import CalledProcessError
import pickle
from pickle import UnpicklingError
import re
import webbrowser
from demix import demixing_plots


VERSION = 1.3
IM_SIZE = (800, 600)
POPUP_SIZE = (200, 400)
sg.theme('Reddit')
CHECK_BOX_SIZE = (25, 1)
INPUT_SIZE = (10, 1)
LABEL_SIZE = (25, 1)
SLIDER_SIZE = (34, 20)
NUM_PARAMS = 40
DATETIME_FORMAT = "%d%m%Y_%H%M%S"
NUM_FIELD_KEYS = ['patch_size_edge', 'trunc_start', 'trunc_length', 'min_size', 'max_size',
                  'sample_freq', 'bg_rank', 'detr_spacing', 'row_blocks', 'col_blocks',
                  'th_lvl', 'pass_num', 'merge_corr_th', 'remove_dimmest', 'residual_cut',
                  'update_ac_max_iter', 'update_ac_tol', 'update_ac_merge_overlap_thr',
                  'bg_reg_lr', 'bg_reg_max_iter', 'demix_start', 'demix_length', 'job_id',
                  'job_to_cancel', 'edge_trim']
DIR_PARAMS_IDX = [1, 13, 39]
MAX_NMF_ELEMENTS = 15
FREQ_TO_HP_SPACING = 100
LOAD_PARAMS_DONT_UPDATE = ['nmf_traces_graph', 'super_pixels_graph', 'Other_plots_graph', '-TABGROUP-',
                           '__len__', 'final_traces_graph', 'param_file_browser', 'input_file_browser',
                           'output_dir_browser', 'stim_dir_browser']
SSH_LINE = "sshpass -p {} ssh -o StrictHostKeyChecking=no rotem.ovadia@bs-cluster.elsc.huji.ac.il \"{}\""
PLOT_FAIL_POPUP = "The file {} was not found or is corrupt.\nThe job might be still running." \
                  "\nPlease check output directory and running jobs.\nError message:\n{}"
TUTORIAL_LINK = 'https://github.com/rotemov/invivo-imaging'


def convert_to_bytes(file_or_bytes, resize=None):
    """
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    """
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def load_picture_on_canvas(values, graph, im_name):
    im_full = os.path.join(values['output_dir'], "plots", im_name)
    if os.path.exists(im_full):
        try:
            im_bin = convert_to_bytes(im_full, IM_SIZE)
            graph.delete_figure("all")
            graph.draw_image(data=im_bin, location=(0, IM_SIZE[1]))
        except PIL.UnidentifiedImageError as e:
            raise FileNotFoundError(PLOT_FAIL_POPUP.format(im_name, str(e)))
    else:
        raise FileNotFoundError(PLOT_FAIL_POPUP.format(im_name, "File not found."))


def open_traces_plot(values, voltage_file, footprint_file, ref_file):
    plot_dir = os.path.join(values['output_dir'], 'plots')
    voltage_full = os.path.join(plot_dir, voltage_file)
    footprint_full = os.path.join(plot_dir, footprint_file)
    ref_full = os.path.join(plot_dir, ref_file)
    if os.path.exists(voltage_full) and os.path.exists(ref_full) and os.path.exists(footprint_full):
        beta_hat2 = skio.imread(voltage_full)
        X2 = skio.imread(footprint_full)
        with open(ref_full, 'rb') as f:
            mov_dims, ref_im = pickle.load(f)
        demixing_plots.plot_final_traces(beta_hat2, ref_im, mov_dims, X2, "Traces", plot_dir)
    else:
        sg.Popup(PLOT_FAIL_POPUP)


def plot_super_pixels(values, rlt_file, ref_file):
    plot_dir = os.path.join(values['output_dir'], 'plots')
    ref_full = os.path.join(plot_dir, ref_file)
    rlt_full = os.path.join(plot_dir, rlt_file)
    if os.path.exists(ref_full) and os.path.exists(rlt_full):
        with open(ref_full, 'rb') as f:
            mov_dims, ref_im = pickle.load(f)
        with open(rlt_full, 'rb') as f:
            rlt = pickle.load(f)
        demixing_plots.plot_super_pixels(rlt, ref_im, "super_pixels", plot_dir)
    else:
        sg.Popup(PLOT_FAIL_POPUP.format())


def plot_nmf_traces(values, rlt_file, ref_file):
    plot_dir = os.path.join(values['output_dir'], 'plots')
    ref_full = os.path.join(plot_dir, ref_file)
    rlt_full = os.path.join(plot_dir, rlt_file)
    if os.path.exists(ref_full) and os.path.exists(rlt_full):
        with open(ref_full, 'rb') as f:
            mov_dims, ref_im = pickle.load(f)
        with open(rlt_full, 'rb') as f:
            rlt = pickle.load(f)
        demixing_plots.plot_nmf_traces(rlt, ref_im, mov_dims, "NMF_Traces", plot_dir)
    else:
        sg.Popup(PLOT_FAIL_POPUP)


def enforce_numbers(window, values, key):
    if values[key] != "":
        if values[key] and values[key][-1] not in ('0123456789.-e'):
            window[key].update(values[key][:-1])


def _bool_to_words(flag):
    if flag:
        return "YES"
    else:
        return "NO"


def nw_drive_path(path):
    return re.sub('[a-zA-Z]:/', '/ems/elsc-labs/adam-y/', path)


def get_args_array(values):
    args = [None] * NUM_PARAMS
    args[0] = "/opt/slurm/bin/sbatch /ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/full_data_bash.sh"
    args[1], args[2] = os.path.split(values['input_file'])
    args[3] = values['normcorre']
    args[4] = values['detrend']
    args[5] = values['moco']
    args[6] = values['find_rois']
    args[7] = values['cut_off_point'] / 100
    args[8] = values['corr_th_fix'] / 100
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
    args[31] = values['demix_start']
    args[32] = values['demix_length']
    args[33] = values['demix_all_frame_flag']
    args[34] = int(int(values['sample_freq']) / FREQ_TO_HP_SPACING)
    args[35] = values['edge_trim']
    args[36] = values['binning_flag']
    args[37] = values['optimize_traces']
    args[38] = parse_nmf_checkboxes(values)
    args[39] = values['stim_dir']

    if values['network_drive_flag']:
        for idx in DIR_PARAMS_IDX:
            args[idx] = nw_drive_path(args[idx])

    for i in range(len(args)):
        if type(args[i]) == bool:
            args[i] = int(args[i])
    return args


def handle_called_process_error(msg):
    sg.Popup(msg + '\nPlease re-check parameters and password')


def call_command_on_cluster(bash_command, password):
    command = SSH_LINE.format(password, bash_command)
    try:
        output = subprocess.check_output(['ubuntu1804', 'run', command])
        return output.decode('utf-8')
    except CalledProcessError as e:
        raise e


def run_command(values):
    running_line = " ".join([str(arg) for arg in get_args_array(values)])
    try:
        output = call_command_on_cluster(running_line, values['password'])
        job_number = [int(s) for s in output.split() if s.isdigit()][0]
        if values['network_drive_flag'] and os.path.isdir(values['output_dir']):
            param_dir = values['output_dir']
        else:
            param_dir = '../params'
        with open(param_dir + "/params_" + str(job_number) + '.pkl', 'wb') as f:
            if 'password' in values:
                del values['password']
            values['last job'] = "Last job ran: " + str(job_number)
            pickle.dump(values, f)
    except CalledProcessError:
        job_number = 'Job couldn\'t start'
        handle_called_process_error("Job couldn't start")
    return job_number


def get_nmf_trace_checkboxes(num_elements):
    cbs = [None] * num_elements
    cbs[0] = sg.Checkbox('0', default=True, key='nmf_flag_0', disabled=False, visible=True)
    for i in range(1, num_elements):
        cbs[i] = sg.Checkbox(str(i), default=False, key='nmf_flag_' + str(i), disabled=True, visible=True)
    return cbs


def enable_nmf_checkboxes(cbs, num_elements):
    for i in range(num_elements):
        cbs[i].update(disabled=False, visible=True)
    for i in range(num_elements, MAX_NMF_ELEMENTS):
        cbs[i].update(disabled=True, value=False)


def parse_nmf_checkboxes(values):
    cells = ""
    for i in range(int(values['nmf_num_elements'])):
        if values['nmf_flag_' + str(i)]:
            cells += str(i)
    return cells


def load_params_from_file(window, values):
    try:
        with open(values['param_file'], 'rb') as f:
            prev_values = pickle.load(f)
            for key in prev_values.keys():
                if key in values.keys() and key not in LOAD_PARAMS_DONT_UPDATE:
                    window[key].update(prev_values[key])
    except (FileNotFoundError, ValueError, UnpicklingError) as e:
        sg.popup("Not a param file\n" + str(e))


def print_logs(window, values):
    running_line = "cat /ems/elsc-labs/adam-y/rotem.ovadia/Programs/invivo-imaging/logs/II_" + values[
        'logs_job_id'] + ".log"
    try:
        log_text = call_command_on_cluster(running_line, values['password'])
        window['logs_mline'].update(log_text)
    except CalledProcessError:
        handle_called_process_error("Log file didn't open.\nIt may not have started yet.")


def check_running_jobs(values):
    running_line = "/opt/slurm/bin/squeue --me --Format=JobID,State,TimeUsed"
    try:
        output = call_command_on_cluster(running_line, values['password'])
        sg.Popup("Check if your job is in this list:\n" + output)
    except CalledProcessError:
        handle_called_process_error("Couldn't check jobs.")


def cancel_job(values):
    running_line = "/opt/slurm/bin/scancel " + values['job_to_cancel']
    try:
        output = call_command_on_cluster(running_line, values['password'])
        sg.Popup("Please validate job was canceled by checking the running jobs.")
    except CalledProcessError:
        handle_called_process_error("Couldn't cancel job.")


def update_pipline():
    try:
        output = subprocess.check_output('git pull')
        sg.Popup("Pipeline was updated!\nPlease reopen for changes to take effect.\ngit logs:\n" + output.decode('utf-8'))
    except CalledProcessError:
        handle_called_process_error("Could not update please revert to using step 4b in the tutorial.")


def main():
    main_runner = [
        [sg.Text('Param file:', size=LABEL_SIZE), sg.InputText(key='param_file'),
         sg.FileBrowse(key='param_file_browser'), sg.Button('Load params')],
        [sg.Text('Movie file:', size=LABEL_SIZE), sg.InputText(key='input_file', default_text=''),
         sg.FileBrowse(key='input_file_browser')],
        [sg.Text('Output directory:', size=LABEL_SIZE), sg.InputText(key='output_dir', default_text=''),
         sg.FolderBrowse(key='output_dir_browser')],
        [sg.Checkbox('From network drive', size=CHECK_BOX_SIZE, default=False, key='network_drive_flag')],
        [sg.Text('Cluster password', size=LABEL_SIZE), sg.InputText('', key='password', password_char='*')],
        [sg.Checkbox('NoRMCoRRe', size=CHECK_BOX_SIZE, default=True, key="normcorre")],
        [sg.Checkbox('Detrending', size=CHECK_BOX_SIZE, default=True, key="detrend")],
        [sg.Checkbox('Motion correction', size=CHECK_BOX_SIZE, default=True, key="moco")],
        [sg.Checkbox('Find ROIs', size=CHECK_BOX_SIZE, default=True, key="find_rois")],
        [sg.Checkbox('Optimize Traces', size=CHECK_BOX_SIZE, default=True, key="optimize_traces")],
        [sg.Checkbox('Quick run', size=CHECK_BOX_SIZE, default=False, key="sup_only")],
        [sg.Text('Cut off point %', size=LABEL_SIZE),
         sg.Slider(range=(80, 99), orientation='h', size=SLIDER_SIZE, key='cut_off_point', default_value=90)],
        [sg.Text('Correlation threshold fix %', size=LABEL_SIZE),
         sg.Slider(range=(30, 60), orientation='h', size=SLIDER_SIZE, key='corr_th_fix', default_value=45)],
        [sg.Text('Denoise start frame', size=LABEL_SIZE),
         sg.In(default_text='1', size=INPUT_SIZE, key='trunc_start', enable_events=True),
         sg.Text('Number of frames', size=LABEL_SIZE),
         sg.In(default_text='5000', size=INPUT_SIZE, key='trunc_length', enable_events=True)],
        [sg.Text('Demix start frame', size=LABEL_SIZE),
         sg.In(default_text='1', size=INPUT_SIZE, key='demix_start', enable_events=True),
         sg.Text('Number of frames', size=LABEL_SIZE),
         sg.In(default_text='10000', size=INPUT_SIZE, key='demix_length', enable_events=True),
         sg.Checkbox('All frames', default=True, key='demix_all_frame_flag', size=CHECK_BOX_SIZE)],
        [sg.Text('Cell diameter', size=LABEL_SIZE),
         sg.In(default_text='30', size=INPUT_SIZE, key='patch_size_edge', enable_events=True)],
        [sg.Text('Sample frequency[Hz]', size=LABEL_SIZE),
         sg.In(default_text='1000', size=INPUT_SIZE, key='sample_freq', enable_events=True)],
        [sg.Text('Last job started: ', size=LABEL_SIZE, key="last_job")],
        [sg.Text('Choose a job to cancel: ', size=LABEL_SIZE),
         sg.In(size=INPUT_SIZE, key="job_to_cancel", default_text=""), sg.Button("Cancel job")]
    ]

    advanced_params = [
        [sg.Text('Warning: Some of the illegal parameter combinations have not been disabled. '
                 'Please revert to the original parameters if your runs constantly fail.')],
        [sg.Text('# bg elements', size=LABEL_SIZE),
         sg.Slider(range=(1, 8), orientation='h', size=SLIDER_SIZE, key='bg_rank', default_value=4)],
        [sg.Text('Detrend spacing', size=LABEL_SIZE),
         sg.In(default_text='5000', size=INPUT_SIZE, key='detr_spacing', enable_events=True)],
        [sg.Text('Row blocks', size=LABEL_SIZE),
         sg.In(default_text='4', size=INPUT_SIZE, key='row_blocks', enable_events=True)],
        [sg.Text('Column blocks', size=LABEL_SIZE),
         sg.In(default_text='2', size=INPUT_SIZE, key='col_blocks', enable_events=True)],
        [sg.Text('Threshold level', size=LABEL_SIZE),
         sg.In(default_text='4', size=INPUT_SIZE, key='th_lvl', enable_events=True)],
        [sg.Text('# Passes', size=LABEL_SIZE),
         sg.In(default_text='1', size=INPUT_SIZE, key='pass_num', enable_events=True)],
        [sg.Text('Merge correlation threshold', size=LABEL_SIZE),
         sg.In(default_text='0.8', size=INPUT_SIZE, key='merge_corr_thr', enable_events=True)],
        [sg.Text('Edge trim', size=LABEL_SIZE),
         sg.Slider(range=(0, 10), orientation='h', size=SLIDER_SIZE, key='edge_trim', default_value=3)],
        [sg.Checkbox('Binning', size=CHECK_BOX_SIZE, default=False, key="binning_flag")],
        [sg.Text('Remove dimmest ', size=LABEL_SIZE),
         sg.In(default_text='0', size=INPUT_SIZE, key='remove_dimmest', enable_events=True)],
        [sg.Text('Residual cut', size=LABEL_SIZE),
         sg.In(default_text='0.6', size=INPUT_SIZE, key='residual_cut', enable_events=True)],
        [sg.Text('UAC max iterations', size=LABEL_SIZE),
         sg.In(default_text='35', size=INPUT_SIZE, key='update_ac_max_iter', enable_events=True)],
        [sg.Text('UAC tol', size=LABEL_SIZE),
         sg.In(default_text='1e-8', size=INPUT_SIZE, key='update_ac_tol', enable_events=True)],
        [sg.Text('UAC merge overlap threshold', size=LABEL_SIZE),
         sg.In(default_text='0.8', size=INPUT_SIZE, key='update_ac_merge_overlap_thr', enable_events=True)],
        [sg.Checkbox('UAC keep shape', size=CHECK_BOX_SIZE, default=True, key="update_ac_keep_shape")],
        [sg.Text('BGR learning rate', size=LABEL_SIZE),
         sg.In(default_text='0.001', size=INPUT_SIZE, key='bg_reg_lr', enable_events=True)],
        [sg.Text('BGR max iterations', size=LABEL_SIZE),
         sg.In(default_text='1000', size=INPUT_SIZE, key='bg_reg_max_iter', enable_events=True)],
        [sg.Text('Registered movie name', size=LABEL_SIZE), sg.InputText(key='mov_in', default_text='movReg.tif')],
        [sg.Text('Stimulation dir: ', size=LABEL_SIZE), sg.InputText(key='stim_dir'),
         sg.FolderBrowse(key='stim_dir_browser')],
        [sg.Checkbox('Background mask', size=CHECK_BOX_SIZE, default=False, key="bg_mask")],
        [sg.Text('Min cell area (pix)', size=LABEL_SIZE),
         sg.In(default_text='10', size=INPUT_SIZE, key='min_size', enable_events=True),
         sg.Text('Max cell area (pix)', size=LABEL_SIZE),
         sg.In(default_text='1000', size=INPUT_SIZE, key='max_size', enable_events=True)],
    ]

    nmf_traces_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                enable_events=True, key='nmf_traces_graph')

    nmf_trace_checkboxes = get_nmf_trace_checkboxes(MAX_NMF_ELEMENTS)
    nmf_traces = [
        [sg.Text('Choose the ones that look like cells:')],
        [nmf_traces_graph],
        [sg.Button('Open zoomable plot', key='NMF_traces_zoom')],
        [sg.Text('# of elements', size=LABEL_SIZE),
         sg.Slider(range=(1, MAX_NMF_ELEMENTS), orientation='h', size=SLIDER_SIZE, key='nmf_num_elements',
                   default_value=1, enable_events=True)],
        nmf_trace_checkboxes
    ]

    super_pixels_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                  enable_events=True, key='super_pixels_graph')

    super_pixels = [
        [sg.Text('Make sure the super pixels look like the cells or the background:')],
        [super_pixels_graph],
        [sg.Button('Open zoomable plot', key='super_pixels_zoom')]
    ]

    final_traces_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                  enable_events=True, key='final_traces_graph')

    final_traces = [
        [sg.Text('These are the final traces:')],
        [final_traces_graph],
        [sg.Button('Open zoomable plot', key='final_traces_zoom')]
    ]

    other_plots_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE,
                                 enable_events=True, key='Other_plots_graph')

    other_plots = [
        [other_plots_graph],
        [sg.Button('Average'), sg.Button('Background Traces'), sg.Button('Intermediate Traces'),
         sg.Button('Temporal Correlations')]
    ]

    logs = [
        [sg.Multiline(size=(110, 30), font='courier 10', background_color='black', text_color='white',
                      key='logs_mline', auto_refresh=True, autoscroll=True)],
        [sg.T('Job ID:'), sg.Input(key='logs_job_id')],
        [sg.Button('Load/Refresh', key='logs_load', enable_events=True, bind_return_key=True),
         sg.Button('Clear', key='logs_clear', enable_events=True)]
    ]

    # The TabgGroup layout - it must contain only Tabs
    tab_group_layout = [[
        sg.Tab('Main Runner', main_runner, font='Courier 15', key='tab_main_runner'),
        sg.Tab('Advanced Parameters', advanced_params, key='tab_advanced_parameters'),
        sg.Tab('NMF Traces', nmf_traces, key='tab_nmf_traces'),
        sg.Tab('Super Pixels', super_pixels, key='tab_super_pixels'),
        sg.Tab('Final Traces', final_traces, key='tab_final_traces'),
        sg.Tab('Other Plots', other_plots, key='tab_outputs'),
        sg.Tab('Logs', logs, key='tab_logs')
    ]]

    # The window layout - defines the entire window
    layout = [
        [sg.TabGroup(tab_group_layout, enable_events=True, key='-TABGROUP-')],
        [sg.Button('Run'), sg.Button('Help'), sg.Button('Load outputs'), sg.Button('Check running jobs'),
         sg.Button('Update pipeline'), sg.Button('Quit')]
    ]

    window = sg.Window('Invivo imaging - Adam Lab - ver' + str(VERSION), layout, no_titlebar=False)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Quit':
            break
        if event == 'Run':
            last_job = run_command(values)
            window['last_job'].update("Last job ran: " + str(last_job))
            window['logs_job_id'].update(str(last_job))
        if event == 'Help':
            webbrowser.open_new(TUTORIAL_LINK)
        try:
            if event == 'Load outputs':

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
        except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
            sg.Popup(str(e))
        if event == 'final_traces_zoom':
            open_traces_plot(values, 'temporal_traces.tif', 'spatial_footprints.tif', 'ref.tif')
        if event == 'NMF_traces_zoom':
            plot_nmf_traces(values, 'rlt.tif', 'ref.tif')
        if event == 'super_pixels_zoom':
            plot_super_pixels(values, 'rlt.tif', 'ref.tif')
        if event == 'nmf_num_elements':
            enable_nmf_checkboxes(nmf_trace_checkboxes, int(values['nmf_num_elements']))
        if event == 'Load params':
            load_params_from_file(window, values)
            # TODO: solve nmf flag issue when loading.
        if event == 'logs_load':
            print_logs(window, values)
        if event == 'Clear':
            window['logs_mline'].update("")
        if event == 'Check running jobs':
            check_running_jobs(values)
        if event == 'Cancel job':
            cancel_job(values)
        if event == 'Update pipeline':
            update_pipline()

        for key in NUM_FIELD_KEYS:
            if event == key:
                enforce_numbers(window, values, key)

    window.close()


if __name__ == "__main__":
    main()
