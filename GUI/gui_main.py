#!/usr/bin/env python
import PySimpleGUI as sg

# Simple example of TabGroup element and the options available to it

sg.theme('Dark Blue')  # Please always add color to your window

# The tab 1, 2, 3 layouts - what goes inside the tab
main_runner = [
    [sg.Text('Movie file:', size=(15, 1)), sg.InputText(key='input_file'), sg.FileBrowse()],
    [sg.Text('Cut off threshold %'), sg.Slider(range=(80, 95), orientation='h', size=(34, 20), key='cut_off_threshold', default_value=90)],
    [sg.Text('Correlation threshold fix %'), sg.Slider(range=(30, 60), orientation='h', size=(34, 20), key='corr_th_fix', default_value=45)],
    [sg.Text('Start frame', size=(15, 1)), sg.In(default_text='1', size=(10, 1)), sg.Text('Number of frames', size=(15, 1)), sg.In(default_text='5000', size=(10, 1))],
    [sg.Checkbox('Quick run', size=(12, 1), default=False)]
    ]

advanced_params = [
    [sg.Text('Advanced Parameters')]

]

outputs = [[sg.Text('Outputs')]]


# The TabgGroup layout - it must contain only Tabs
tab_group_layout = [[sg.Tab('Main Runner', main_runner, font='Courier 15', key='tab_main_runner'),
                     sg.Tab('Advanced Parameters', advanced_params, key='tab_advanced_parameters'),
                     sg.Tab('Outputs', outputs, key='tab_outputs')]]

# The window layout - defines the entire window
layout = [[sg.TabGroup(tab_group_layout,
                       enable_events=True,
                       key='-TABGROUP-')],
          [sg.Text('Make tab number'), sg.Input(key='-IN-', size=(3, 1)), sg.Button('Invisible'), sg.Button('Visible'),
           sg.Button('Select')]]

window = sg.Window('My window with tabs', layout, no_titlebar=False)

tab_keys = ('-TAB1-', '-TAB2-', '-TAB3-')  # map from an input value to a key
while True:
    event, values = window.read()  # type: str, dict
    print(event, values)
    if event == sg.WIN_CLOSED:
        break
    # handle button clicks
    if event == 'Invisible':
        window[tab_keys[int(values['-IN-']) - 1]].update(visible=False)
    if event == 'Visible':
        window[tab_keys[int(values['-IN-']) - 1]].update(visible=True)
    if event == 'Select':
        window[tab_keys[int(values['-IN-']) - 1]].select()

window.close()
