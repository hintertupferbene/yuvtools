import numpy as np
import os

from yuvtools import yuvtools
from yuvtools import conversion


tmp_yuv_filename = 'tmp.yuv'
tmp_png_filename = 'tmp.png'


class SampleData:
    @staticmethod
    def get_yuv444_sample_sequence(width=6, height=10, num_frames=42):
        sample_data = (np.random.random_sample((width, height, num_frames)) * 255).astype(np.uint8)
        y = sample_data
        u = sample_data
        v = sample_data
        return y, u, v

    @staticmethod
    def get_yuv420_sample_sequence(width=6, height=10, num_frames=42):
        y, u, v = SampleData.get_yuv444_sample_sequence(width, height, num_frames)
        return conversion.YCbCr444_video_to_YCbCr420_video(y, u, v)


def test_import_equals_export_yuv_444():
    y, u, v, = SampleData.get_yuv444_sample_sequence()
    yuvtools.yuv_export(y, u, v, tmp_yuv_filename)
    yr, ur, vr = yuvtools.yuv_import(tmp_yuv_filename, y.shape[1], y.shape[0], y.shape[2], colorformat=444)
    os.remove(tmp_yuv_filename)
    assert np.all(np.equal(y, yr))
    assert np.all(np.equal(u, ur))
    assert np.all(np.equal(v, vr))


def test_import_equals_export_yuv_420():
    y, u, v, = SampleData.get_yuv420_sample_sequence()
    yuvtools.yuv_export(y, u, v, tmp_yuv_filename)
    yr, ur, vr = yuvtools.yuv_import(tmp_yuv_filename, y.shape[1], y.shape[0], y.shape[2], colorformat=420)
    os.remove(tmp_yuv_filename)
    assert np.all(np.equal(y, yr))
    assert np.all(np.equal(u, ur))
    assert np.all(np.equal(v, vr))


def test_imported_image_equals_exported():
    y, u, v, = SampleData.get_yuv444_sample_sequence(num_frames=1)
    rgb = conversion.ycbcr2rgb(np.dstack((y, u, v)))
    yuvtools.imwrite(rgb, tmp_png_filename)
    rgb2 = yuvtools.imread(tmp_png_filename)
    assert np.all(np.equal(rgb, rgb2))
    os.remove(tmp_png_filename)


def convert():
    y, u, v = SampleData.get_yuv444_sample_sequence(num_frames=2)
    for f in range(2):
        rgb = conversion.ycbcr2rgb(np.dstack((y[:,:,0], u[:,:,0], v[:,:,0])))
        yuvtools.imwrite(rgb, f'tmp0000{f+5}.png')
    pattern = '*.png'
    out_name = tmp_yuv_filename
    yuvtools.convert_png_sequence_to_yuv420(pattern, out_name)
    # TODO: finish this test and load yuv sequence again and compare content
