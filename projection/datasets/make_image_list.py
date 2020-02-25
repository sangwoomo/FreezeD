import numpy as np
from PIL import Image
import chainer
import random
import scipy.misc


if __name__ == "__main__":
    import glob, os, sys

    root_path = sys.argv[1]

    count = 0
    n_image_list = []
    filenames = glob.glob(root_path + '/*/*.jpg')
    for filename in filenames:
        filename = filename.split('/')
        dirname = filename[-2]
        label = int(dirname)
        n_image_list.append([os.path.join(filename[-2], filename[-1]), label])
        count += 1

    print("Num of examples:{}".format(count))
    n_image_list = np.array(n_image_list, np.str)
    np.savetxt(f'image_list_{root_path}.txt', n_image_list, fmt="%s")
