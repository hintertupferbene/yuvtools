import numpy as np
from PIL import Image


def YCbCr4202YCbCr444(y, cb, cr, bitdepth=np.uint8):
    """Up-sample the input 4:2:0 YCbCr image to 4:4:4 YCbCr

    :param ndarray y: Y image plane
    :param ndarray cb: Cb image plane
    :param ndarray cr: Cr image plane
    :param int bitdepth: bit depth of the data
    :return: 3D 4:4:4 YCbCr image array
    :rtype: ndarray
    """
    height, width = y.shape

    ycbcr444 = np.zeros((height, width, 3), dtype=bitdepth)

    ycbcr444[:, :, 0] = y

    # Cb
    ycbcr444[0::2, 0::2, 1] = cb  # top left
    ycbcr444[1::2, 0::2, 1] = cb  # bottom left
    ycbcr444[0::2, 1::2, 1] = cb  # top right
    ycbcr444[1::2, 1::2, 1] = cb  # bottom right

    # Cr
    ycbcr444[0::2, 0::2, 2] = cr  # top left
    ycbcr444[1::2, 0::2, 2] = cr  # bottom left
    ycbcr444[0::2, 1::2, 2] = cr  # top right
    ycbcr444[1::2, 1::2, 2] = cr  # bottom right

    return ycbcr444


def YCbCr420_to_channels(ycbcr420):
    """Split the input YCbCr 4:2:0 image (represented as 4:4:4 3D array) into its Y, Cb, and Cr planes.

    :param ndarray ycbcr420: 4:2:0 image represented as 4:4:4 3D array
    :return: Tuple (Y, Cb, Cr)
    :rtype: (ndarray, ndarray, ndarray)
    """
    height, width, channels = ycbcr420.shape
    assert channels == 3

    y = ycbcr420[:, :, 0]
    cb = ycbcr420[::2, ::2, 1]
    cr = ycbcr420[::2, ::2, 2]

    return y, cb, cr


def YCbCr444_video_to_YCbCr420_video(Y444, U444, V444):
    Y420 = Y444.copy()
    U420 = []
    V420 = []
    num_frames = U444.shape[2]
    for f in range(num_frames):
        U420.append(resize_image_plane(U444[:, :, f], 0.5))
        V420.append(resize_image_plane(V444[:, :, f], 0.5))
    U420 = np.dstack(U420)
    V420 = np.dstack(V420)
    return Y420, U420, V420


def resize_image_plane(plane, factor):
    """This is currently using the default subsampling method of resize()"""
    height, width = plane.shape
    return np.array(Image.fromarray(plane).resize((int(width * factor), int(height * factor))))


def YCbCr4442YCbCr420(ycbcr444):
    """Convert 3D 4:4:4 YCbCr image array to 4:2:0 image returned as 4:4:4 3D image array

    :param ndarray: 3D 4:4:4 YCbCr image array
    :return: 3D 4:2:0 YCbCr image as 3D 4:4:4 image array
    :rtype: ndarray
    """
    height, width, channels = ycbcr444.shape
    assert channels == 3

    cb420 = resize_image_plane(ycbcr444[1, :, :], 0.5)
    cr420 = resize_image_plane(ycbcr444[1, :, :], 0.5)

    ycbcr420 = YCbCr4202YCbCr444(ycbcr444[:, :, 0].copy(), cb420.astype(np.uint8), cr420.astype(np.uint8))

    return ycbcr420


def rgb2ycbcr(rgb, flavor=601):
    """Convert 3D RGB image array to 3D YCbCr image array

    :param ndarray rgb: 3D RGB array
    :param int flavor: flavor of conversion (601 for BT.601, 709 for BT.709)
    :return: 3D YCbCr array
    :rtype: ndarray
    """
    height, width, channels = rgb.shape
    assert channels == 3

    R, G, B = np.dsplit(rgb.astype(np.int32), 3)

    if flavor == 601:
        Y = ((66 * R + 129 * G + 25 * B + 128) >> 8) + 16
        Cb = ((-38 * R - 74 * G + 112 * B + 128) >> 8) + 128
        Cr = ((112 * R - 94 * G - 18 * B + 128) >> 8) + 128
    elif flavor == 709:
        Y = ((47 * R + 157 * G + 16 * B + 128) >> 8) + 16
        Cb = ((-26 * R - 87 * G + 112 * B + 128) >> 8) + 128
        Cr = ((112 * R - 102 * G - 10 * B + 128) >> 8) + 128
    else:
        print('Unknown conversion flavor.')
        return

    ## correct overflow to headroom
    Y[Y > 235] = 235
    Cb[Cb > 240] = 240
    Cr[Cr > 240] = 240

    ycbcr = np.dstack((Y, Cb, Cr))

    ycbcr[ycbcr < 16] = 16

    return ycbcr.astype(np.uint8)


def ycbcr2rgb_601(ycbcr):
    """Convert 3D 4:4:4 YCbCr image array to 3D RGB image array.

    This is according to BT.601.

    :param ndarray ycbcr: 3D RGB array
    :return: 3D YCbCr array
    :rtype: ndarray
    """
    height, width, channels = ycbcr.shape
    assert channels == 3

    Y, Cb, Cr = np.dsplit(ycbcr.astype(np.int32), 3)

    # Conversion according to BT.601:
    R = ((298 * Y + 409 * Cr) >> 8) - 223
    G = ((298 * Y - 100 * Cb - 208 * Cr) >> 8) + 136
    B = ((298 * Y + 516 * Cb) >> 8) - 277

    rgb = np.dstack((R, G, B))

    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    return rgb.astype(np.uint8)


def ycbcr2rgb(ycbcr, flavor=601):
    """Convert 3D YCbCr 4:4:4 image array to 3D RGB image array.
    Different conversion compared to ycbcr2rgb_citrix(...)!

    :param ndarray: 3D RGB array
    :param int flavor: flavor of conversion (601 for BT.601, 709 for BT.709)
    :return: 3D YCbCr array
    :rtype: ndarray
    """
    height, width, channels = ycbcr.shape
    assert channels == 3

    Y, Cb, Cr = np.dsplit(ycbcr.astype(np.int32), 3)
    C = Y - 16
    D = Cb - 128
    E = Cr - 128

    # R = clip2((298*C + 409*E + 128) >> 8)
    # G = clip2((298*C - 100*D - 208*E + 128) >> 8)
    # B = clip2((298*C + 516*D + 128) >> 8)
    if flavor == 601:
        R = ((298 * C + 409 * E + 128) >> 8)
        G = ((298 * C - 100 * D - 208 * E + 128) >> 8)
        B = ((298 * C + 516 * D + 128) >> 8)
    elif flavor == 709:
        R = ((298 * C + 459 * E + 128) >> 8)
        G = ((298 * C - 55 * D - 136 * E + 128) >> 8)
        B = ((298 * C + 541 * D + 128) >> 8)
    else:
        print('Unknown conversion flavor.')
        return

    rgb = np.dstack((R, G, B))

    rgb[rgb > 255] = 255
    rgb[rgb < 0] = 0

    return rgb.astype(np.uint8)
