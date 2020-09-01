#!/usr/bin/env python
import PySimpleGUI as sg
import os

# Simple example of TabGroup element and the options available to it

sg.theme('Dark Blue')  # Please always add color to your window

# The tab 1, 2, 3 layouts - what goes inside the tab
main_runner = [
    [sg.Text('Movie file:', size=(15, 1)), sg.InputText(key='input_file'), sg.FileBrowse()],
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

blob_elem = sg.Image(filename='GUI/blob.png')

outputs = [
    [sg.Text('Cell Photos')],
    [blob_elem]
]


# The TabgGroup layout - it must contain only Tabs
tab_group_layout = [[sg.Tab('Main Runner', main_runner, font='Courier 15', key='tab_main_runner'),
                     sg.Tab('Advanced Parameters', advanced_params, key='tab_advanced_parameters'),
                     sg.Tab('Outputs', outputs, key='tab_outputs')]]

# The window layout - defines the entire window
layout = [
    [sg.TabGroup(tab_group_layout, enable_events=True, key='-TABGROUP-')],
    [sg.Button('Run'), sg.Button('Help'), sg.Button('Load outputs'), sg.Button('Quit')]
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
    if event == 'Load outputs':
        print('Loading traces')
        if os.path.exists('GUI/blob_2.png'):
            blob_elem.Update(filename='GUI/blob_2.png')


window.close()
