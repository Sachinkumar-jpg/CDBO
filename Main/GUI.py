import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from Main import RUN
sg.change_look_and_feel('LightGreen6')  # for default look and feel


# Designing layout
layout = [[sg.Text("\t")],
          [sg.Text("\t\t\t  Channel      "), sg.Combo(["AWGN", "Rician", 'Rayleigh'], size=(12, 2))],
          [sg.Text("\t\t\t  No.of users "), sg.Combo(["10", "20", "30", "40", "50"], size=(12, 2)), sg.Text("\n")],
          [sg.Text("\t\t\t\t      "), sg.Button("START", size=(11, 1))], [sg.Text("\n")],

          [sg.Text("\t\t\t\tEE Optimization\t  Joint resource allocation\t  Dcdd-MCTS\t\tBee fly pattern\t\tProposed CDBO ")],
          [sg.Text('\tEnergy Efficiency\t'), sg.In(key='11', size=(20, 20)), sg.In(key='12', size=(20, 20)),
           sg.In(key='13', size=(20, 20)), sg.In(key='14', size=(20, 20)), sg.In(key='15', size=(20, 20)), sg.Text(" ")],

          [sg.Text('\tThroughput\t'), sg.In(key='21', size=(20, 20)), sg.In(key='22', size=(20, 20)),
           sg.In(key='23', size=(20, 20)), sg.In(key='24', size=(20, 20)), sg.In(key='25', size=(20, 20)), sg.Text(" ")],

          [sg.Text('\n\t\t\t\t\t\t\t\t\t\t\t\t\t\t'), sg.Button('Run Graph'), sg.Text(" "), sg.Button('Close')]]


# to plot graph
def plot_graph(result_1, result_2):

    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result = np.transpose(result)

    # labels for bars
    labels = ['EE Optimization and Dynamic Mode Selection', 'Joint resource allocation', 'Dcdd-MCTS',
              'Bee fly pattern-based resource allocation', 'Proposed CDBO']         # x-axis labels

    tick_labels = ['Energy Efficiency', 'Throughput']                                          # metrics
    bar_width, s = 0.15, 0.025                                                      # bar width, space between bars

    for i in range(len(result)):                                                    # allocating location for bars
        if i == 0:                                                                  # initial location - 1st result
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:                                                                       # location from 2nd result
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    # plotting a bar chart
    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)
    plt.legend()                                                                    # show a legend on the plot
    plt.show()                                                                      # to show the plot


# Create the Window layout
window = sg.Window('GUI', layout)
# event loop
while True:
    event, value = window.read()  # displays the window
    if event == "START":
        channel = value[0]
        users = int(value[1])

        Energy, Throughput = RUN.call_main(channel, users)
        print("\n!... Execution Done ...!")
        window.Element('11').Update(Energy[0])
        window.Element('12').Update(Energy[1])
        window.Element('13').Update(Energy[2])
        window.Element('14').Update(Energy[3])
        window.Element('15').Update(Energy[4])

        window.Element('21').Update(Throughput[0])
        window.Element('22').Update(Throughput[1])
        window.Element('23').Update(Throughput[2])
        window.Element('24').Update(Throughput[3])
        window.Element('25').Update(Throughput[4])

    if event == 'Run Graph':
        plot_graph(Energy, Throughput)

    if event == 'Close':
        window.close()
        break
window.close()
