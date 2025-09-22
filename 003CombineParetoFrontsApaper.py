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
        "heights": [ 0.24738342332910465, 0.14580369911212243, 0.09003553077166812, 0.01959683150471194, 6.007481779758699e-5, 6.0074817797586765e-5, 5.11035831189026e-5, 4.757641410140728e-5, 4.7519883251303785e-5, 4.670790162134553e-5, 4.669811856528977e-5, 4.655627027640097e-5],
        "labels": ["(x₂ - x₄)", "(x₄ - exp(x₄))", "(x₂ - (x₄ + x₁))", "(x₁ - exp(x₄ + x₁))", "(-0.677 - (x₄ + x₁))", "((x₃ / exp(x₁ + x₄)) / x₃)", "((1.00 ^ x₂) / (x₁ + x₄))", "(sq(1.00 ^ x₂) / (x₁ + x₄))", "(sin(exp(x₂ * 0.0148)) / (x₄ + x₁))", "(((x₃ / (x₁ + x₄)) / x₃) - (0.993 ^ x₂))", "(exp((x₃ / (x₁ + x₄)) / x₃) / (0.993 ^ x₂))", "(((x₃ / (x₁ + x₄)) / x₃) - sin(sq(0.993 ^ x₂)))"],
        "highlight_index": 4,  ##2x2 trace
    },
    {
        "heights": [ 0.19255454866309454, 0.16633721916891586, 0.06917380470548058, 0.06786658315548952, 0.007597601645105772, 0.003669189274324758, 0.002833896221553381, 0.002502406081877353],
        "labels": [ "(x₂ * x₃)", "((x₂ * x₃) - x₁)", "((x₂ * x₃) - (x₁ * x₄))", "(((x₂ * x₃) - (x₁ * x₄)) - x₁)", "(((x₂ * x₃) - (x₁ * x₄)) / (x₁ + x₄))", "(((x₂ * x₃) - (x₁ * x₄)) / exp(sin(sin(x₁ + x₄))))",  "((((x₂ * x₃) - (x₁ * x₄)) / exp(sin(sin(x₁ + x₄)))) - x₁)", "((((x₂ * x₃) - (x₁ * x₄)) / sin(exp(sin(sin(x₁ + x₄))))) - x₁)"],
        "highlight_index": 2,  ##2x2 determinant
    },      
    {
        "heights": [ 0.040770845126557397, 5.775503841003263e-6, 5.7631147229953e-6, 5.460062647026944e-6, 5.4480284268685355e-6, 5.266357090061305e-6, 5.135509499393118e-6, 5.121291911619246e-6],
        "labels": [ "(x₁ + x₉)", "((x₅ + x₁) + x₉)", "((x₅ + (x₁ ^ 1.00)) + x₉)", "(sq(exp(exp((x₅ + x₁) + x₉))) + x₄)", "(sq(exp(exp((x₅ + x₁) + x₉))) + sin(x₄))", "((exp((x₅ + x₁) + x₉) / 0.0183) + x₄)", "(sq(exp((x₅ + x₁) + x₉) / -0.278) + x₄)", "(sq(exp(((x₅ + x₁) + 1.23) + x₉)) + sin(x₄))"],
        "highlight_index": 1,  ##3x3 trace
    },
    {
        "heights": [ 0.11188588647994965, 0.11188588647994964, 0.07460644198740202, 0.04935570914771862, 0.04497835653665629, 0.025055747801233343, 0.002695437666880793, 0.0024225278933158504, 0.0024205325695375, 0.0023487953536724367, 0.00223331280651822],
        "labels": [ "(x₆ * x₈)", "(exp(x₈) ^ x₆)", "((x₈ * x₆) - sq(x₃))", "((x₈ * x₆) + (x₄ * x₂))", "(((x₈ * x₆) - sq(x₄)) - sq(x₃))", "((x₈ * x₆) - (sq(x₃) - (x₄ * x₂)))", "((x₈ * x₆) + ((x₄ * x₂) + (x₃ * x₇)))","(((x₈ + 0.0716) * x₆) + ((x₄ * x₂) + (x₃ * x₇)))", "((x₈ * x₆) - ((x₆ * -0.0757) - ((x₃ * x₇) + (x₄ * x₂))))", "((x₈ * x₆) - ((0.0618 / exp(x₃)) - ((x₃ * x₇) + (x₄ * x₂))))", "((x₈ * x₆) - (((x₆ + x₃) * -0.0686) - ((x₃ * x₇) + (x₄ * x₂))))"],
        "highlight_index": 6,  ##3x3 antisymmetric sum of principal minors
    },
    {
        "heights": [ 0.036585451155327894, 0.016740053055263077, 1.0731772997168571e-5, 1.073177299716857e-5, 1.0643700644313095e-5, 1.0538518309660858e-5, 1.0516086554631015e-5, 1.0434535742716409e-5, 1.0413605617314861e-5, 1.0396343385681053e-5],
        "labels": [ "(x₆ + x₁)", "((x₆ + x₁₆) + x₁)", "((x₁₁ + x₁₆) + (x₆ + x₁))", "(exp((x₁₆ + x₁₁) + x₆) * exp(x₁))", "(((x₁₁ ^ 1.00) + x₁₆) + (x₆ + x₁))", "(exp(exp((x₁₁ + x₁₆) + (x₆ + x₁))) - sq(x₁₁))", "(exp(exp((x₁₁ + x₁₆) + (x₆ + x₁))) - sq(sin(x₁₁)))", "((sq(sq(exp((x₁₁ + x₁₆) + (x₆ + x₁)))) - x₁₁) + x₁₄)", "((sq(sq(exp((x₁₁ + x₁₆) + (x₆ + x₁)))) - exp(x₁₁)) + x₁₄)", "((sq(sq(exp((x₁₁ + x₁₆) + (x₆ + x₁)))) - x₁₁) + (x₁₄ * x₄))"],
        "highlight_index": 2,  ##4x4 trace
    },
    {
        "heights": [ 0.15659304393495613, 0.12171364497305516, 0.09099637298484554, 0.06916118797123193, 0.04905994202791276, 0.04859218458065067, 0.047398898214872434, 0.046636663058110324, 0.04639192426044344, 0.0460178630536912],
        "labels": [ "(x₂ + x₄)", "((x₄ + x₂) + x₃)", "(x₄ + (x₃ + (x₂ * x₅)))", "((x₃ * x₆) + ((x₂ * x₅) + x₄))", "(((x₃ * x₆) + (x₁ * x₄)) + (x₅ * x₂))", "(exp(((x₃ * x₆) + (x₁ * x₄)) + (x₅ * x₂)) + x₃)", "(((x₃ * x₆) + (x₁ * x₄)) + ((x₅ + 0.330) * x₂))", "(sq(((x₃ * x₆) + (x₁ * x₄)) + ((x₅ + 0.351) * x₂)) + x₃)", "(((x₃ * x₆) + ((x₁ - -0.258) * x₄)) + ((x₅ + 0.318) * x₂))", "(sq(((x₃ * x₆) + (x₁ * x₄)) + ((x₅ + 0.351) * x₂)) + (x₃ + x₄))"],
        "highlight_index": 4,  ##4x4 Field Strength tensor
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

    ax.set_title(f'Pareto Front for Experiment {i + 1}', 
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