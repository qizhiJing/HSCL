import scipy.io as scio
import numpy as np


def data_load(path, str):

    alldata = scio.loadmat(path)
    data = np.array(alldata[str])
    mri = data[0, 0]
    pet = data[1, 0]
    csf = data[2, 0]
    gnd = data[3, 0].squeeze(1)


    return mri, pet, csf, gnd



