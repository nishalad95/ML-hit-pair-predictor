import matplotlib.pyplot as plt
import csv
import math
import numpy as np
import pandas as pd
import seaborn as sns

def create_binned_predictions(predictions_file):
    """
    Generate binned acceptance region from classifier predictions
    Input:
    csv file containing raw predictions from classifier
    csv file should contain the following columns:
    weta, tau, predictions
    ________
    Returns:
    df object containing the binned binary acceptance region
    """
    
    predictions = pd.read_csv(predictions_file)
    
    # LUT dimension variables
    m_c = np.ndarray((4,),float)
    m_w = 30       # width of LUT
    m_h = 45       # height of LUT
    m_c[0] = 0   # min of cluster widths
    m_c[1] = 3   # max of cluster widths
    m_c[2] = 0   # min of tau
    m_c[3] = 9   # max of tau

    bin_inc_x = m_c[1] / m_w
    bin_inc_y = m_c[3] / m_h
    
    weta_bins = np.arange(m_c[0], m_c[1], bin_inc_x)
    weta_bins = np.append(weta_bins, 3.0)
    tau_bins = np.arange(m_c[2], m_c[3], bin_inc_y)
    tau_bins = np.append(tau_bins, 9.0)
    
    lut = []
    for i in range(len(weta_bins)-1):
        # calculate min and max tau
        band = predictions.loc[(predictions.weta > weta_bins[i]) 
				                & (predictions.weta <= weta_bins[i+1])
                            	& (predictions.predictions == 1)]
    
        if len(band) != 0:

            band_min = band.tau.min()
            band_max = band.tau.max()

            # convert into bin representation
            bin_min = int(round(band_min / bin_inc_y))
            bin_max = int(round(band_max / bin_inc_y))
            # append to LUT
            lut.append([i, bin_min, bin_max])

    df = pd.DataFrame(index=np.arange(0,45,1).tolist(), columns=np.arange(0,30,1).tolist())
    for i in range(len(lut)):
        if (lut[i][1] == 0 and lut[i][2] == 0):
            continue
        df[lut[i][0]][lut[i][1] : lut[i][2] + 1] = 1
    
    df = df.fillna(0)
    df = df.iloc[::-1]

    return df


def generate_binary_lut(filename):
    """
    Converts a lut from file into a df object containing binary acceptance region
    Input:
    filepath containing lut
    Output:
    dataframe object
    """
    
    lut = []
    lut_file = open(filename, "r")
    for line in lut_file.readlines():

        elements = line.split(" ")
        bin_num = elements[0]
        bin_min = elements[1]
        bin_max = elements[2]
    
        lut.append([int(bin_num), int(bin_min), int(bin_max)])

    df = pd.DataFrame(index=np.arange(0,45,1).tolist(), columns=np.arange(0,30,1).tolist())
    for i in range(len(lut)):
        if (lut[i][1] == 0 and lut[i][2] == 0):
            continue
        df.iloc[i][lut[i][1] : lut[i][2] + 1] = 1
    
    df = df.fillna(0)
    df = df.iloc[::-1]
    
    return df



def plot_lut(df, output):
    """
    Saves a visual plot of the LUT to output destination file
    """
    flatui = ["#3d77d4", "#f0b05d"]
    fig, ax = plt.subplots(figsize=(18,10))
    p = sns.heatmap(df, linewidths=0.1, annot=False, cbar=True, 
                    ax=ax, cmap=sns.color_palette(flatui), 
                    cbar_kws={'orientation': 'vertical',
                              'label': 'class'})

    colorbar = p.collections[0].colorbar
    colorbar.set_ticks([0.25, 0.75])
    colorbar.set_ticklabels(['0', '1'])

    plt.title('2D Look-Up Table')
    plt.xlabel('binned cluster width')
    plt.ylabel('binned tau')
    plt.savefig(output)


def ensemble_lut(lut_1, lut_2):
    """
    Execute the logical OR between two LUTs
    Inputs:
    lut_1 and lut_2 are dataframes containing binary binned acceptance regions
    Returns:
    Array containing the logical OR between the two LUTs
    """
    lut_1 = lut_1.to_numpy().astype(bool)
    lut_2 = lut_2.to_numpy().astype(bool)
    ensemble_or_array = np.logical_or(lut_1, lut_2)
    return ensemble_or_array.astype(int)


def create_lut_list(df):
    """
    Provided a df containing the binned acceptance region, the acceptance region 
    is extracted and converted into a LUT list format
    Returns:
    [[tau_idx, weta_min, weta_max], [tau_idx, weta_min, weta_max], ... ]
    """
    lut_to_save = df[::-1]
    lut_list = []

    for i in range(len(lut_to_save)):
        tau_bin_num = i
        weta_bin_min = list(lut_to_save[i]).index(1)
        weta_bin_max = len(lut_to_save[i]) - list(lut_to_save[i][::-1]).index(1) - 1
        
        lut_list.append([tau_bin_num, weta_bin_min, weta_bin_max])

    return lut_list


def save_lut_list(lut, outputFile):
    """"
    Save LUT list to file
    Input lut format:
    [[tau_idx, weta_min, weta_max], [tau_idx, weta_min, weta_max], ... ]
    """"
    with open(outputFile, "w") as f:
        wr = csv.writer(f, delimiter =' ')
        wr.writerows(lut)


def combine_lut(barrel_lut_file, endcap_lut_file):
    """
    Combine 2 separate LUTs into 1 file
    
    Input LUTs should be in the following file format indicating acceptance region:
    col0 col1 col2
    col0 col1 col2
    ...
    col0: tau bin number, col1: weta min bin, col2: weta max bin
    
    Output list format:
    [[col0 col1 col2 col3 col4], [col0 col1 col2 col3 col4], ...]
    col0: tau bin number, col1 & col2: min & max barrel weta,
    col3 & col4: min & max endcap weta
    """
    
    lut = []
    barrel_lut = open(barrel_lut_file, "r")
    for line in barrel_lut.readlines():
    
        elements = line.split(" ")
        bin_num = elements[0]
        bin_min = elements[1]
        bin_max = elements[2].split("\n")[0]

        lut.append([int(bin_num), int(bin_min), int(bin_max)])
    
    endcap_lut = open(endcap_lut_file, "r")
    for line in endcap_lut.readlines():
    
        elements = line.split(" ")
        bin_num = elements[0]
        bin_min = elements[1]
        bin_max = elements[2].split("\n")[0]
        
        if len(lut[int(bin_num)]) == 5:
            lut.append([int(bin_num), 0, 0, int(bin_min), int(bin_max)])
        else:
            lut[int(bin_num)].append(int(bin_min))
            lut[int(bin_num)].append(int(bin_max))
        
    return lut