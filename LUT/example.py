from lookuptable import *
import pathlib
from skimage.util import img_as_ubyte
from skimage import io
from skimage.morphology import (dilation, erosion, selem, binary_dilation, binary_erosion)
import numpy as np

path = str(pathlib.Path().absolute())


# 1. Convert classifier predictions to binned acceptance region
clf_predictions = "../pixel-barrel/barrel-kde-predictions.csv"
binned_df = create_binned_predictions(clf_predictions)
plot_lut(binned_df, "test_binned_predictions.png")


# 2. Form an ensemble LUT
strict_lut = path + "/trigseed_ML_strict.lut"
strict_df = generate_binary_lut(strict_lut)
# plot_lut(strict_df, "test_strict_lut.png")
ensemble_array = ensemble_lut(binned_df, strict_df)
plot_lut(ensemble_array, "test_ensemble_lut.png")


# 3. Morphological Filtering: Dilation
# create structuring element
se = selem.rectangle(3,6)
se[0], se[2] = 0, 0
print("Applying dilation using structuring element: \n", se)
orig_phantom = img_as_ubyte(ensemble_array)
dilated = dilation(orig_phantom, se)
plot_lut(dilated, "test_dilated_lut.png")
print("Dilated LUT plot saved")


# 4. Morphological Filtering: Erosion
# create structuring element
se = np.zeros(shape=(8,8), dtype=int)
for i in range(len(se)):
    se[i][len(se) - 1 - i] = 1
se[-1][1], se[-1][2], se[-2][2] = 1, 1, 1
print("Applying erosion using structuring element: \n", se)
orig_phantom = img_as_ubyte(dilated.astype(int))
eroded = erosion(orig_phantom, se)
plot_lut(eroded, "test_eroded_lut.png")
print("Eroded LUT plot saved")


# 5. Save smoothed df to LUT format
outputFile = path + "/test_eroded.lut"
eroded_lut = create_lut_list(eroded)
save_lut_list(eroded_lut, outputFile)


# 6. Combine barrel and endcap LUTs into 1 file
barrel = path + "/pixel_barrel_kde.lut"
endcap = path + "/pixel_endcap_kde.lut"
outputFile = path + "/test_combined.lut"
combined_lut = combine_lut(barrel, endcap)
save_lut_list(combined_lut, outputFile)