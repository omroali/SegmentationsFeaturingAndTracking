import os
import glob
from natsort import natsorted


def get_images_in_path(folder_path):
    images = sorted(filter(os.path.isfile, glob.glob(folder_path + "/*")))
    list = []
    for file_path in images:
        if "GT" not in file_path:
            list.append(file_path)

    return natsorted(list)
