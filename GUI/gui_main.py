#!/usr/bin/env python
import PySimpleGUI as sg
import os
import PIL
from PIL import Image
import io
import base64

IM_SIZE = (800, 600)


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
    im_full = os.path.join(values['output_dir'], im_name)
    print(im_full)
    if os.path.exists(im_full):
        im_bin = convert_to_bytes(im_full, IM_SIZE)
        graph.delete_figure("all")
        graph.draw_image(data=im_bin, location=(0, IM_SIZE[1]))
        print(im_name + " loaded")
    else:
        print("Still running")



sg.theme('Reddit')


main_runner = [
    [sg.Text('Movie file:', size=(15, 1)), sg.InputText(key='input_file'), sg.FileBrowse()],
    [sg.Text('Output directory:', size=(15, 1)), sg.InputText(key='output_dir'), sg.FolderBrowse()],
    [sg.Text('Cut off threshold %', size=(20, 1)),
     sg.Slider(range=(80, 95), orientation='h', size=(34, 20), key='cut_off_threshold', default_value=90)],
    [sg.Text('Correlation threshold fix %', size=(20, 1)),
     sg.Slider(range=(30, 60), orientation='h', size=(34, 20), key='corr_th_fix', default_value=45)],
    [sg.Text('Start frame', size=(15, 1)), sg.In(default_text='1', size=(5, 1)),
     sg.Text('Number of frames', size=(15, 1)), sg.In(default_text='5000', size=(5, 1))],
    [sg.Text('Min cell area (pix)', size=(15, 1)), sg.In(default_text='10', size=(5, 1)),
     sg.Text('Max cell area (pix)', size=(15, 1)), sg.In(default_text='1000', size=(5, 1))],
    [sg.Checkbox('Quick run', size=(10, 1), default=False)]
]

advanced_params = [
    [sg.Text('Advanced Parameters')]
]


nmf_traces_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE, enable_events=True, key='nmf_traces_graph')

nmf_traces = [
    [sg.Text('Choose the ones that look like cells:')],
    [nmf_traces_graph]
]

super_pixels_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE, enable_events=True, key='super_pixels_graph')

super_pixels = [
    [sg.Text('Make sure the super pixels look like the cells or the background')],
    [super_pixels_graph]
]

final_traces_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE, enable_events=True, key='final_traces_graph')

final_traces = [
    [sg.Text('These are the final traces of the cells you chose:')],
    [final_traces_graph]
]

other_plots_graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE, enable_events=True, key='Other_plots_graph')

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
    event, values = window.read()  # type: str, dict
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Quit':
        break
    if event == 'Run':
        print("Running script activated")
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

window.close()
