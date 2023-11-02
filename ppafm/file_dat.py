import os
import zlib

import numpy as np
import pandas as pd


def readDat(fname, Header=None):
    ndir, nfile = os.path.split(fname)
    ndir = ndir + "/"
    try:
        with open(fname, mode="rb") as binary_file:
            data = binary_file.read()
    except OSError:
        print("oops! File " + nfile + " can not be read.")
    if b"[Parameter]" in data:
        STMAFMVersion = 1
    elif b"[Paramet32]" in data:
        STMAFMVersion = 2
    elif b"[Paramco32]" in data:
        STMAFMVersion = 3
    else:
        # If none of these is found, stop and signal file error.
        print("Createc DAT file version does not match")

    # Read out all header information until the max header byte of 16384 bytes.
    header_size = 16384
    header_binary = data[0:header_size]

    idx = header_binary.index(b"DSP-COMPDATE")
    header_binary = header_binary[:idx] + b"\r" + header_binary[idx:]
    header_binary = header_binary.splitlines()
    ind = [i for i, s in enumerate(header_binary) if b"DSP-COMPDATE" in s][0]

    d = []
    for i in range(1, ind):
        tmp = header_binary[i].split(b"=")
        if len(tmp) > 1:
            d.append((tmp[0], tmp[1]))
    SplittedLine = pd.DataFrame(d, columns=("parameter", "value"))

    if Header is None:
        Header = {}

    Header["ScanPixels_X"] = int(SplittedLine.loc[SplittedLine["parameter"] == b"Num.X / Num.X"].value.item())
    Header["ScanPixels_Y"] = int(SplittedLine.loc[SplittedLine["parameter"] == b"Num.Y / Num.Y"].value.item())
    Header["GainX"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"GainX / GainX"].value.item())
    Header["GainY"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"GainY / GainY"].value.item())
    Header["GainZ"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"GainZ / GainZ"].value.item())
    Header["GainPreamplifier"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Gainpreamp / GainPre 10^"].value.item())
    Header["ChannelCount"] = int(SplittedLine.loc[SplittedLine["parameter"] == b"Channels / Channels"].value.item())
    Header["DACToZConversionFactor"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Dacto[A]z"].value.item())
    Header["DACToXYConversionFactor"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Dacto[A]xy"].value.item())
    Header["ScanRange_X"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Length x[A]"].value.item())
    Header["ScanRange_Y"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Length y[A]"].value.item())
    Header["ScanOffset_X"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Scanrotoffx / OffsetX"].value.item())
    Header["ScanOffset_Y"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Scanrotoffy / OffsetY"].value.item())
    Header["Bias"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Biasvolt[mV]"].value.item())
    # Header['Current'] = float(SplittedLine.loc[SplittedLine['parameter'] == b'Current[A]'].value.item())
    Header["ACQ_Time"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Sec/Image:"].value.item())
    Header["ScanAngle"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Rotation / Rotation"].value.item())
    Header["ZControllerSetpoint"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"FBLogIset"].value.item())
    Header["ZControllerIntegralGain"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"FBIntegral"].value.item())
    Header["ZControllerProportionalGain"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"FBProp"].value.item())
    Header["PiezoZ"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"ZPiezoconst"].value.item())
    Header["PiezoX"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Xpiezoconst"].value.item())
    Header["CHOffset"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"CHModeZoff / CHModeZoff"].value.item())
    Header["CHMode"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"CHMode / CHMode"].value.item())
    Header["LengthX"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Length x[A]"].value.item())
    Header["LengthY"] = float(SplittedLine.loc[SplittedLine["parameter"] == b"Length y[A]"].value.item())

    # Check the STMAFM version to determine the number of bytes / pixel
    if STMAFMVersion == 1:
        pass
        # Header + 2 unused "NULL"-Bytes
        data_start = header_size + 2
    elif STMAFMVersion == 2:
        pass
        # Header + 4 unused "NULL"-Bytes
        data_start = header_size + 4
    elif STMAFMVersion == 3:
        data_start = header_size
        # No Seek of additional bytes, since they are compressed:

    # Begin image data extraction
    # Read data starting after header
    image_data = data[data_start:]

    # Decompress the data with zlib library
    x = zlib.decompress(image_data)

    # Transpose the data so it is oriented from top to bottom
    y = np.frombuffer(x, dtype=np.float32).transpose()

    # Skip the first value (zero)
    y = y[1:]

    # Removes 2048 values from the array (extra Createc space)
    y = y[: -4 * Header["ScanPixels_X"] + 1]

    # Fill remaining data with zeros in case of early scan termination
    y_round_size = int(np.ceil(float(len(y)) / float(Header["ScanPixels_X"])) * Header["ScanPixels_X"])
    y_round = np.zeros(y_round_size)
    y_round[: len(y)] = y

    # Reshape the output image
    mat_image = y_round.reshape(-1, Header["ScanPixels_X"])

    # Set the Y pixel value
    y_size = Header["ScanPixels_Y"]

    # Generate output pictures depending on the number of channels
    pic1 = mat_image[:y_size, :]
    pic2 = mat_image[y_size : 2 * y_size, :]
    pic3 = mat_image[2 * y_size : 3 * y_size, :]
    pic4 = mat_image[3 * y_size :, :]

    # Crop the image if there are rows with zeroes
    ind_list = np.where(~(pic1 == 0.0).all(axis=1))
    try:
        pic1_crop = pic1[ind_list[0], :]
    except:
        pic1_crop = pic1
    try:
        pic2_crop = pic2[ind_list[0], :]
    except:
        pic2_crop = pic2
    try:
        pic3_crop = pic3[ind_list[0], :]
    except:
        pic3_crop = pic3
    try:
        pic4_crop = pic4[ind_list[0], :]
    except:
        pic4_crop = pic4

    return (pic1_crop, pic2_crop, pic3_crop, pic4_crop)
