import numpy as np
import os
import sys
from matplotlib.pyplot import imread
import re

"""
general utils for surface
"""

__all__ = ['alicona_read']


def alicona_read(full_path: str):
    r"""
    Reads .al3d and associated files made by alicona measurement machines

    Will look for texture and icon images automatically, reads tags and depth data at a minimum from the al3d file.

    Parameters
    ----------
    full_path : str
        The full path including extension to an al3d file

    Returns
    -------
    data : dict
        Actual keys depend on the data found:
        - 'DepthData' : Array of depth data with nan in place of invalid values
        - 'TextureData' : Array of texture data or image of the surface
        - 'Header' : Dict of tags read from the header
        - 'Icon' : Array of icon image data

    Notes
    -----
    If the file name in full_path ends with (#) or # where # is an integer this function will look first for texture (#)
    or texture # etc. otherwise just texture will be found

    This is a port of the matlab function from the Alicona file format reader (al3D) tool box

    Copyright (c) 2016, Martin
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    \* Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

    \* Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
    following disclaimer in the documentation and/or other materials provided with the distribution
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
    GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    """

    path, file_name = os.path.split(full_path)
    name, ext = file_name.split('.')
    try:
        number_tag = re.findall(r'\(\d*\)\Z', name)[-1][1:-1]
    except IndexError:
        try:
            number_tag = re.findall(r'\s\d*\Z', name)[-1][1:]
        except IndexError:
            number_tag = ''

    data = dict()
    tags = dict()

    with open(full_path, 'rb') as file:
        # read the header

        line = file.readline()
        tags['Type'] = line[:line.find(0)].decode(sys.stdout.encoding)

        line = file.readline()
        tags['Version'] = int(bytearray([byte for byte in line[20:-1] if
                                         byte != 0]).decode(sys.stdout.encoding))

        line = file.readline()
        tags['TagCount'] = int(bytearray([byte for byte in line[20:-1] if
                                          byte != 0]).decode(sys.stdout.encoding))

        for tag_num in range(tags['TagCount']):
            line = file.readline()
            tag_name = bytearray([byte for byte in line[0:20] if byte != 0]
                                 ).decode(sys.stdout.encoding)
            tv_str = bytearray([byte for byte in line[20:-1] if byte != 0]
                               ).decode(sys.stdout.encoding)
            try:
                tag_value = int(tv_str)
            except ValueError:
                try:
                    tag_value = float(tv_str)
                except ValueError:
                    tag_value = tv_str
            tags[tag_name] = tag_value

        line = file.readline()
        tags['Comment'] = bytearray([byte for byte in line[20:-1] if byte != 0]
                                    ).decode(sys.stdout.encoding)

        data['Header'] = tags

        # read the icon data

        if tags['IconOffset'] > 0:
            file.seek(tags['IconOffset'])
            icon = np.zeros([152, 150, 3], dtype='uint8')
            for i in range(3):
                icon[:, :, i] = np.reshape(np.array(file.read(22800), dtype='uint8'), (152, 150))
            data['Icon'] = icon
        else:
            try:
                icon = imread(path + os.path.sep + "icon" + number_tag + ".bmp")
                data['Icon'] = icon
            except FileNotFoundError:
                try:
                    icon = imread(path + os.path.sep + "icon.bmp")
                    data['Icon'] = icon
                except FileNotFoundError:
                    pass

        # read the depth data
        rows = int(tags['Rows'])

        if tags['DepthImageOffset'] > 0:

            if tags['TextureImageOffset'] == 0:
                cols = (file.seek(0, 2) - tags['DepthImageOffset']) / (4 * rows)
            else:
                cols = (tags['TextureImageOffset'] - - tags['DepthImageOffset']
                        ) / (4 * rows)

            cols = int(round(cols))

            file.seek(tags['DepthImageOffset'])

            depth_data = np.array(np.frombuffer(file.read(rows * cols * 4),
                                                np.float32))
            depth_data[depth_data == tags['InvalidPixelValue']] = float('nan')
            data['DepthData'] = np.reshape(depth_data, (rows, cols))[:, :tags['Cols']]

        # read the texture data

        if tags['TextureImageOffset'] > 0:

            if 'TexturePtr' in tags:
                if tags['TexturePtr'] == '0;1;2':
                    num_planes = 4
                else:
                    num_planes = 1
            elif 'NumberOfPlanes' in tags:
                num_planes = tags['NumberOfPlanes']
            else:
                msg = ("The file format may have been updated please ensure this"
                       " version is up to date then contact the developers")
                raise NotImplementedError(msg)

            cols = int((file.seek(0, 2) - tags['TextureImageOffset']) / (num_planes * rows))

            file.seek(tags['TextureImageOffset'])

            texture_data = np.zeros([cols, rows, num_planes], dtype='uint8')

            for plane in range(num_planes):
                texture_data[:, :, plane] = np.reshape(np.array(file.read(cols * rows)), (cols, rows))

            texture_data = texture_data[:tags['Cols'], :, :]

            if num_planes == 4:
                data['TextureData'] = texture_data[:, :, 0:3]
                data['QualityMap'] = texture_data[:, :, -1]
            else:
                data['TextureData'] = texture_data[:, :, 0]

        else:
            # check if there is a texture image in the current dir
            try:
                data['TextureData'] = imread(path + os.path.sep +
                                             "texture" + number_tag + ".bmp")
            except FileNotFoundError:
                try:
                    tex = imread(path + os.path.sep + "texture.bmp")
                    data['TextureData'] = tex
                except FileNotFoundError:
                    pass
    return data


if __name__ == '__main__':
    file_name_t = "D:\\Downloads\\Alicona_data\\Surface Profile Data\\dem.al3d"
    from matplotlib.pyplot import imshow

    data_t = alicona_read(file_name_t)

    imshow(data_t['DepthData'])
