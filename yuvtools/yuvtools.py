import sys
import numpy as np
from scipy import misc

from . import conversion


def float2str(number, decimal_points=2):
    number = float(number)
    s = '{0:.' + str(decimal_points) + 'f}'
    return s.format(number)


def imwrite(image, filename, q=100):
    """Write RGB image to file (png or jpeg)

    :param ndarray image: 3D RGB image array
    :param string filename: Filename with extension .png or .jpg/.jpeg
    :param int q: Quality parameter for JPEG
    """
    im = misc.toimage(image, cmin=0, cmax=255)

    tmp = filename.split('.')
    if tmp[-1] == 'jpg' or tmp[-1] == 'jpeg':
        im.save(filename, quality=q)
    else:
        im.save(filename)


def RGB_image_export(RGB, filename, access_mode='wb'):
    try:
        f = open(filename, access_mode)
    except:
        print('Could not open ' + filename)

    RGB[:, :, 0].astype(np.uint8).tofile(f)
    RGB[:, :, 1].astype(np.uint8).tofile(f)
    RGB[:, :, 2].astype(np.uint8).tofile(f)

    f.close()


def RGB_image_import(filename, width, height, POC=0, bitdepth=np.uint8):
    assert (bitdepth == np.uint8 or bitdepth == np.uint16)
    try:
        f = open(filename, "rb")
    except:
        print('Could not open ' + filename)
        sys.exit()

    # go to desired position
    if bitdepth == np.uint8:
        f.seek(width * height * 3 * POC)
    if bitdepth == np.uint16:
        f.seek(width * height * 3 * POC * 2)

    count = width * height
    R = np.fromfile(f, dtype=bitdepth, count=count)
    G = np.fromfile(f, dtype=bitdepth, count=count)
    B = np.fromfile(f, dtype=bitdepth, count=count)
    R = R.reshape(height, width)
    G = G.reshape(height, width)
    B = B.reshape(height, width)

    f.close()

    RGB = np.zeros((height, width, 3), dtype=np.uint8)
    RGB[:, :, 0] = R
    RGB[:, :, 1] = G
    RGB[:, :, 2] = B

    return RGB


def yuv_image_export(Y, U, V, filename, access_mode='wb'):
    # write yuv image
    try:
        f = open(filename, access_mode)
    except:
        print('Could not open ' + filename)
        sys.exit()

    if Y.dtype == np.uint8:
        Y.astype(np.uint8).tofile(f)
        U.astype(np.uint8).tofile(f)
        V.astype(np.uint8).tofile(f)
    else:
        Y.astype(np.uint16).tofile(f)
        U.astype(np.uint16).tofile(f)
        V.astype(np.uint16).tofile(f)

    f.close()


def yuv_image_import_420(filename, width, height, POC=0):
    """Read 4:2:0 image. Just for backward compatibility. Better use yuv_image_import directly.

    :param filename:
    :param width:
    :param height:
    :param POC:
    :return: (Y, Cb, Cr)
    """
    return yuv_image_import(filename, width, height, POC)


def yuv_image_import_444(filename, width, height, POC=0):
    """Read 4:4:4 image. Just for backward compatibility. Better use yuv_image_import directly.

    :param filename:
    :param width:
    :param height:
    :param POC:
    :return: (Y, Cb, Cr)
    """
    return yuv_image_import(filename, width, height, POC, colorformat=444, as444=True)


def yuv_image_import(filename, width, height, POC=0, bitdepth=np.uint8, colorformat=420, as444=False):
    assert (bitdepth == np.uint8 or bitdepth == np.uint16)
    assert (colorformat == 420 or colorformat == 444)

    f = open(filename, "rb")

    bytes_per_sample = 1
    if bitdepth == np.uint16:
        bytes_per_sample = 2

    # go to desired position
    if colorformat == 420:
        f.seek(int(width * height * 1.5 * POC * bytes_per_sample))
    if colorformat == 444:
        f.seek(width * height * 3 * POC * bytes_per_sample)

    count = width * height
    Y = np.fromfile(f, dtype=bitdepth, count=count)
    Y = Y.reshape(height, width)

    if colorformat == 420:
        width = int(width / 2)
        height = int(height / 2)
        count = width * height
    Cb = np.fromfile(f, dtype=bitdepth, count=count)
    Cr = np.fromfile(f, dtype=bitdepth, count=count)
    Cb = Cb.reshape(height, width)
    Cr = Cr.reshape(height, width)

    f.close()

    if as444:
        if colorformat == 444:
            return np.dstack((Y, Cb, Cr))
        else:
            # convert to 444 image
            return conversion.YCbCr4202YCbCr444(Y, Cb, Cr, bitdepth=bitdepth)
    else:
        return (Y, Cb, Cr)



def yuv_import(filename, width, height, numframes, startPOC=0, bitdepth=np.uint8, colorformat=420):
    Y = []
    Cb = []
    Cr = []
    for POC in range(startPOC, startPOC + numframes, 1):
        Yi, Cbi, Cri = yuv_image_import(filename, width, height, POC,
                                                                               bitdepth=bitdepth,
                                                                               colorformat=colorformat)
        Y.append(Yi)
        Cb.append(Cbi)
        Cr.append(Cri)
    return np.dstack(Y), np.dstack(Cb), np.dstack(Cr)


def yuv_export(Y, U, V, filename):
    # write yuv sequence (8 bit data only)
    num_frames = Y.shape[2]
    am = 'wb'
    for POC in range(num_frames):
        yuv_image_export(Y[:, :, POC], U[:, :, POC], V[:, :, POC], filename, access_mode=am)
        if POC == 0:
            am = 'ab'


def get_psnr(image1, image2, channel):
    height, width, channels = image1.shape
    assert channels == 3

    return getPSNR(image1[:, :, channel], image2[:, :, channel])


def get_psnr_RGB(image1, image2):
    PSNR_R = getPSNR(image1[:, :, 0], image2[:, :, 0])
    PSNR_G = getPSNR(image1[:, :, 1], image2[:, :, 1])
    PSNR_B = getPSNR(image1[:, :, 2], image2[:, :, 2])
    mean_RGB_PSNRs = (PSNR_R + PSNR_G + PSNR_B) / 3
    return mean_RGB_PSNRs


def getPSNR(im1, im2):
    # for two image planes only (not multi channel images)
    MSE = float(np.sum(np.square(im1.astype(np.int32) - im2.astype(np.int32)))) / float(im1.size)
    return float("inf") if MSE == 0 else 10.0 * np.log10(float(255 * 255) / float(MSE))
