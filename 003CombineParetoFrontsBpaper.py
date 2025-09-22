# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:20:16 2024

@author: sebas
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np


SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


heights = [0.04974608361775346, 0.043944781620859116, 0.025830440969275215, 0.0003674434040849351, 0.00024588311869464994, 0.00024016059584664113, 0.00020475014871112297, 0.00018584075206988062]
labels = ["(x₁ * -1.1303493778509395)", "((x₁ * -8.258362456795677) - x₃)", "(((x₃ * x₂) * 0.2254199205956011) - x₁)", "(sin(x₂ + x₃) + (x₁ * x₁))", "(sin(x₂ + x₃) + exp(x₁ * -0.8640225857953432))", "((sin(x₂ + x₃) * 1.1059737200217283) + exp(x₁ * -0.9079839426346769))", "(exp(sin(x₂ + x₃) - x₁) / sin(1.3232616834633946 - (x₁ * -0.7090890438253801)))", "((exp(sin(x₂ + x₃) - x₁) / sin(1.289754999432597 - x₁)) - exp(x₁))"]
heights = [np.log(x)+10 for x in heights]

highlight_index = 3  # Index of the element to highlight (0-based)

colors = ['skyblue' if i != highlight_index else 'salmon' for i in range(len(heights))]

plt.bar(labels, heights, color=colors)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Labels')
plt.ylabel('Log(MSE)+10')
#plt.title('Pareto Front')
plt.tight_layout()

# Highlighting the label
plt.text(highlight_index, heights[highlight_index], f'{labels[highlight_index]}\n', color='red', ha='center', va='bottom')

plt.show()


# Data for the three histograms
data_sets = [
    {
        "heights": [ 0.3960504178739666, 0.21751271982220155, 0.18666646899837747, 0.17378612045518257, 0.01702036295563718, 0.0006902436271454376, 0.0006902436271454375, 0.0006659386840098879, 0.0006656293236134397, 0.0006421243607299, 0.0006420403598846995, 0.0006326547232418551],
        "labels": [ "(x₂ - x₁)", "(x₂ - sq(x₁))", "(x₂ - exp(sq(x₁)))", "(x₂ - sq(exp(sq(x₁))))", "(x₁ - (sq(x₂) + exp(x₁)))", "(0.173 / (sq(x₂) + sq(x₁)))", "(x₂ - (x₂ + exp(sq(x₂) + sq(x₁))))", "((x₁ * -0.00286) - (sq(x₂) + sq(x₁)))", "((exp(x₁) * -0.00286) - (sq(x₂) + sq(x₁)))", "(sq(sq(sq(x₂) + sq(x₁)) - 55170) - x₁)", "(sq(sq(sq(x₂) + sq(x₁)) - 55170) - exp(x₁))", "((sq(sq(sq(x₂) + sq(x₁)) - 55170) - x₁) + x₂)"],
        "highlight_index": 5,  ##harmonic oscillator
    }, 
    {
        "heights": [ 0.34862033602052606, 0.19189561811464845, 0.006820900338808343, 0.001960857973715799, 0.0014925995333457118, 0.0014131125390984987, 0.0014040196333968546, 0.0013546177852783192, 0.0013124128008750755, 0.0013111224299617152],
        "labels": [ "(x₁ * x₁)", "(sq(x₁) - x₂)", "(sq(x₂) + sq(x₁))", "(sq(x₂) + exp(sq(x₁)))", "(exp(exp(sq(x₂) + exp(sq(x₁)))) - x₁)", "(sq(exp(sq(sq(x₂) + exp(sq(x₁))))) - x₁)", "(sq(sq(sq(sq(sq(sq(x₂) + exp(sq(x₁))))))) - x₁)", "(exp(exp(sq(x₂) + exp(sq(x₁) / 1.04))) - x₁)", "(sq(exp(sq(sq(x₂) + exp(sq(x₁) / 1.04)))) - x₁)", "(sq(exp(sq(sq(x₂) + exp(sq(x₁) / 1.04)))) - sin(x₁))"],
        "highlight_index": 3,  ##anharmonic oscillator poly
    },    
    {
        "heights": [ 0.3848231648054095, 0.2174089622234048, 0.17278699300242328, 0.14186792731059827, 0.015544329151563542, 0.014580061591217634, 0.0007611324155469063, 0.000761117185942587, 0.0007254689630300463],
        "labels": [ "(x₂ - x₁)", "(x₂ - exp(x₁))", "(x₂ - sq(exp(x₁)))", "((x₂ - exp(x₂)) - x₁)", "((x₂ - exp(x₂)) - sin(x₁))", "((x₂ - exp(x₂)) - sin(sin(x₁)))", "((sin(x₁) / -0.500) - sq(x₂))", "((x₂ * (x₂ * -0.501)) - sin(x₁))", "(sin(x₁ / -1.00) - sq(x₂ * 0.707))"],
        "highlight_index": 6,  ##anharmonic oscillator sin
    },
    {
        "heights":  [ 0.38157878489960423, 0.2356374923089565, 0.10980658462539436, 0.04816496569789335, 0.014115834288556534, 0.00016402234404217314,  0.00016388434420472743, 0.00016330881034563676, 0.00016309607057771264],
        "labels": [ "(x₁ * x₁)", "(exp(x₁) * x₁)", "(sq(x₁) + sq(x₂))", "((sq(x₁) + sq(x₂)) + x₁)", "(sq(exp(x₁) + x₁) + sq(x₂))", "((sq(x₁) + sq(x₂)) + (exp(x₁) / sin(0.184)))", "(exp((sq(x₁) + sq(x₂)) + (exp(x₁) / 0.183)) + exp(x₂))", "(exp((sq(x₁) + sq(x₂)) + (exp(x₁) / 0.183)) + sq(exp(x₂)))", "(exp((sq(x₁) + sq(x₂)) + (exp(x₁) / 0.183)) + (sq(exp(x₂)) * x₂))"],
        "highlight_index": 5,  ##anharmonic oscillator exp
    },
    {
        "heights": [ 0.16661533338325388, 0.10782393275199396, 0.0997045961297643, 0.010567315601496898, 0.010567315601496896, 0.007822648362014554, 0.0075806982428924105, 0.006055695303370868, 0.004671003438330628, 0.00418657788146455, 0.003552703160321812, 0.003371103426028088, 0.0031590928846262456],
        "labels": [ "(x₄ * x₁)", "(x₄ - (x₃ * x₂))", "(sin(x₄) - (x₃ * x₂))", "((x₄ * x₁) - (x₃ * x₂))", "(exp(x₄ * x₁) / exp(x₃ * x₂))", "(((x₄ - -0.136) * x₁) - (x₃ * x₂))", "(((x₄ * x₁) - (x₃ * x₂)) + (x₁ / exp(exp(x₂))))", "(((x₄ * x₁) - (x₃ * x₂)) + (sin(x₁) / exp(exp(x₂))))", "((((x₄ * x₁) - (x₃ * x₂)) / 0.152) - (x₂ - x₁))", "((((x₄ * x₁) - (x₃ * x₂)) / 0.168) - (sin(x₂) - x₁))", "(((((x₄ * x₁) - (x₃ * x₂)) / 0.193) + sin(x₁)) - sin(x₂))", "(((((x₄ * x₁) - (x₃ * x₂)) / 0.203) + sin(x₁)) - sin(sin(x₂)))", "(((((x₄ * x₁) - (x₃ * x₂)) / 0.218) + sin(sin(x₁))) - sin(sin(x₂)))"],
        "highlight_index": 3,  ##2d central potential
    },
    {
        "heights": [ 0.18606185866620495, 0.17167021511925729, 0.11669492668229088, 0.10175323109378988, 0.05883620704045088, 0.04412302672987912, 0.029832289228893114, 0.015644722812575593, 0.0010199483471364913, 0.0010194412062623317, 0.0010189904195691895, 0.0010186030423929853, 0.001013989750947124],
        "labels": [ "(x₁ - x₄)", "(sq(x₁) - x₄)", "((x₁ - x₄) - x₃)", "(sq(x₁) - (x₃ + x₄))", "(((x₁ - x₃) - x₄) - x₂)", "((x₁ - (x₄ + sq(x₂))) - x₃)", "((sq(x₁) - (x₃ + x₄)) - sq(x₂))", "((sq(x₁) - (x₃ + sq(x₄))) - sq(x₂))", "((sq(x₁) - (sq(x₃) + sq(x₄))) - sq(x₂))", "((sq(x₁) - (sq(x₃) + (x₄ ^ 1.99))) - sq(x₂))", "((sq(x₁ * 1.00) - (sq(x₃) + sq(x₄))) - sq(x₂))", "((sq(x₁ ^ sq(1.00)) - (sq(x₃) + sq(x₄))) - sq(x₂))", "((sq(x₁ * (1.00 ^ x₃)) - (sq(x₃) + sq(x₄))) - sq(x₂))"],
        "highlight_index": 8,  ##space time interval
    }, 
    # {
    #     "heights": [ 0.11278644550552014, 3.506563603596514e-10, 3.506563602078304e-10, 3.4811983929411435e-10, 1.5846443160002347e-10, 1.3857108720685526e-10, 1.3579966988477688e-10, 1.3423807022482348e-10],
    #     "labels": [ "(x₂ * x₁)", "((x₂ * -1.4034811163969947) - x₁)", "(((x₂ / -0.41606318150915966) + x₂) - x₁)", "(exp(((x₂ * -1.4035179851184993) - x₁) - -8.154893140689133) + x₂)", "(exp(((x₂ * -1.4034839404542463) - x₁) - -7.880162355728031) + (x₂ * x₁))", "((x₂ * -2.8258583427735373) - (x₁ + ((x₁ - -0.029487107960120627) + exp(x₁ * 0.013641789717423771))))", "(((x₂ * -2.8258544203940525) - (x₁ + ((x₁ + -2.8258583427735373) + exp(x₁ * 0.013647428673134102)))) - 0.1799522806045977)", "((exp(((x₂ * -1.4035629741293985) - x₁) - -8.154893141964795) + (x₂ - x₁)) + exp(x₂ * x₁))"],
    #     "highlight_index": 1,
    # },
    # {
    #     "heights": [0.04974608361775346, 0.043944781620859116, 0.025830440969275215, 0.0003674434040849351, 0.00024588311869464994, 0.00024016059584664113, 0.00020475014871112297, 0.00018584075206988062],
    #     "labels": ["(x₁ * -1.1303493778509395)", "((x₁ * -8.258362456795677) - x₃)", "(((x₃ * x₂) * 0.2254199205956011) - x₁)", "(sin(x₂ + x₃) + (x₁ * x₁))", "(sin(x₂ + x₃) + exp(x₁ * -0.8640225857953432))", "((sin(x₂ + x₃) * 1.1059737200217283) + exp(x₁ * -0.9079839426346769))", "(exp(sin(x₂ + x₃) - x₁) / sin(1.3232616834633946 - (x₁ * -0.7090890438253801)))", "((exp(sin(x₂ + x₃) - x₁) / sin(1.289754999432597 - x₁)) - exp(x₁))"],
    #     "highlight_index": 3,
    # },
    # {
    #     "heights": [0.04974608361775346, 0.043944781620859116, 0.025830440969275215, 0.0003674434040849351, 0.00024588311869464994, 0.00024016059584664113, 0.00020475014871112297, 0.00018584075206988062],
    #     "labels": ["(x₁ * -1.1303493778509395)", "((x₁ * -8.258362456795677) - x₃)", "(((x₃ * x₂) * 0.2254199205956011) - x₁)", "(sin(x₂ + x₃) + (x₁ * x₁))", "(sin(x₂ + x₃) + exp(x₁ * -0.8640225857953432))", "((sin(x₂ + x₃) * 1.1059737200217283) + exp(x₁ * -0.9079839426346769))", "(exp(sin(x₂ + x₃) - x₁) / sin(1.3232616834633946 - (x₁ * -0.7090890438253801)))", "((exp(sin(x₂ + x₃) - x₁) / sin(1.289754999432597 - x₁)) - exp(x₁))"],
    #     "highlight_index": 3,
    # },
    # Add more datasets if needed
]

fig, axes = plt.subplots(2, 3, figsize=(16, 12), dpi=120)

for i, data_set in enumerate(data_sets):
    heights = data_set["heights"]
    labels = data_set["labels"]
    highlight_index = data_set["highlight_index"]

    # Colors: pastel blue for all, highlight in orange
    colors = ['#5bc0de' if j != highlight_index else '#e6550d' 
              for j in range(len(heights))]

    ax = axes[i // 3, i % 3]
    ax.set_axisbelow(True) 
    bars = ax.bar(labels, heights, color=colors, edgecolor="black", alpha=0.7)

    # Axis styling
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=50, ha='right')
    ax.set_yscale('log')

    # Push y-axis label downward
    ax.set_ylabel('MSE on Normalized Gradients', fontsize=MEDIUM_SIZE, rotation=90, va='bottom', y=0.0)

    ax.set_title(f'Pareto Front for Experiment {i + 7}', 
                 fontsize=BIGGER_SIZE, pad=12, weight="bold")

    # Annotate highlighted label
    x = highlight_index
    y = heights[highlight_index]
    highlighted_label = labels[highlight_index]
    ax.text(x, y * 1.15, highlighted_label,
            color='red', ha='center', va='bottom', 
            fontsize=MEDIUM_SIZE, fontweight="bold")

    # Light grid for readability
    ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()