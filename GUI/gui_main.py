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



# Simple example of TabGroup element and the options available to it


sg.theme('Reddit')  # Please always add color to your window


# The tab 1, 2, 3 layouts - what goes inside the tab
main_runner = [
    [sg.Text('Movie file:', size=(15, 1)), sg.InputText(key='input_file'), sg.FileBrowse()],
    [sg.Text('Output directory:', size=(15, 1)), sg.InputText(key='output_dir'), sg.FolderBrowse()],
    [sg.Text('Cut off threshold %', size=(15, 1)),
     sg.Slider(range=(80, 95), orientation='h', size=(34, 20), key='cut_off_threshold', default_value=90)],
    [sg.Text('Correlation threshold fix %', size=(20, 1)),
     sg.Slider(range=(30, 60), orientation='h', size=(34, 20), key='corr_th_fix', default_value=45)],
    [sg.Text('Start frame', size=(10, 1)), sg.In(default_text='1', size=(5, 1)),
     sg.Text('Number of frames', size=(15, 1)), sg.In(default_text='5000', size=(5, 1))],
    [sg.Text('Min cell size', size=(10, 1)), sg.In(default_text='10', size=(3, 1)), sg.Text('pix', size=(3, 1)),
     sg.Text('Max cell size', size=(10, 1)), sg.In(default_text='1000', size=(5, 1)), sg.Text('pix', size=(3, 1))],
    [sg.Checkbox('Quick run', size=(10, 1), default=False)]
]

advanced_params = [
    [sg.Text('Advanced Parameters')]
]

graph = sg.Graph(canvas_size=IM_SIZE, graph_bottom_left=(0, 0), graph_top_right=IM_SIZE, enable_events=True, key='-GRAPH-')

outputs = [
    [sg.Text('Cell Photos')],
    [graph]
]


# The TabgGroup layout - it must contain only Tabs
tab_group_layout = [[sg.Tab('Main Runner', main_runner, font='Courier 15', key='tab_main_runner'),
                     sg.Tab('Advanced Parameters', advanced_params, key='tab_advanced_parameters'),
                     sg.Tab('Outputs', outputs, key='tab_outputs')]]

# The window layout - defines the entire window
layout = [
    [sg.TabGroup(tab_group_layout, enable_events=True, key='-TABGROUP-')],
    [sg.Button('Run'), sg.Button('Help'), sg.Button('Load NMF_Traces'), sg.Button('Load super pixels'), sg.Button('Quit')]
    ]

window = sg.Window('Invivo imaging - Adam Lab - ver0.0', layout, no_titlebar=False)

# tab_keys = ('-TAB1-', '-TAB2-', '-TAB3-')  # map from an input value to a key
while True:
    event, values = window.read()  # type: str, dict
    print(event, values)
    if event == sg.WIN_CLOSED or event == 'Quit':
        break
    # handle button clicks
    if event == 'Run':
        print("Running script activated")
    if event == 'Help':
        print("Link to github appeared")
    if event == 'Load NMF_Traces':
        print('Traces')
        load_picture_on_canvas(values, graph, 'NMF_Traces.png')
    if event == 'Load super pixels':
        print('Super pixels')
        load_picture_on_canvas(values, graph, 'super_pixels.png')


window.close()



