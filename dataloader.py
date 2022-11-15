import cv2
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from skimage.filters.rank import entropy
from skimage.morphology import disk


def regions_filter(image, max_choose):
    if(len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (512, 512))
    img_height, img_width = image.shape
    cell_size = 32

    # 提取图片的熵
    entropy_image = entropy(image, disk(3))
    local_goodness = np.zeros(
        [int(img_height/cell_size), int(img_width/cell_size)], dtype=np.float32)

    # 利用数量筛选显著区域
    for i in range(int(img_height/cell_size)):
        for j in range(int(img_width/cell_size)):
            local_goodness[i, j] = np.sum(
                entropy_image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]) / (cell_size*cell_size)
    local_goodness_flatten = local_goodness.flatten()
    local_choose = np.argpartition(
        local_goodness_flatten, -max_choose)[-max_choose:]
    local_min = np.min(local_goodness_flatten[local_choose])
    regions = np.where(local_goodness >= local_min)

    # 设置hog
    hog = cv2.HOGDescriptor((16, 16), (16, 16), (16, 16), (8, 8), 9, 1)
    hog_labels_list = []

    epsilon = 1e-8

    # 将区域图像保存为多通道
    multichannel_images = np.zeros(
        (cell_size, cell_size, max_choose), dtype=np.uint8)
    for channel in range(max_choose):
        x = regions[1][channel]*cell_size
        y = regions[0][channel]*cell_size
        multichannel_images[:, :, channel] = image[y:y +
                                                   cell_size, x:x+cell_size]

        des = hog.compute(multichannel_images[:, :, channel])
        # des /= np.linalg.norm(des
        hog_labels_list.append(des.tolist())

    hog_label = np.array(hog_labels_list, dtype='float32')
    multichannel_images = multichannel_images.transpose(2, 0, 1)
    multichannel_images = np.asarray(
        multichannel_images, dtype='float32') / 255

    return multichannel_images, hog_label


class CoCALC(Dataset):
    """
    定义CohogNet算法的数据类
    """

    def __init__(self, data_dir):
        super(CoCALC, self).__init__()
        self.data_dir = data_dir

        # 遍历图像数据
        self.images_files = []
        for root, dirs, files in os.walk(data_dir):
            for item in files:
                if item.split('.')[-1].lower() in ["jpg", 'jpeg', 'png', 'bmp']:
                    self.images_files.append(os.path.join(root, item))

    def _load_data(self, image_dir):
        """
        加载数据，并生成训练图像和label
        """
        img = cv2.resize(cv2.imread(image_dir, 0), (32, 32))
        hog = cv2.HOGDescriptor((16, 16), (16, 16), (16, 16), (8, 8), 9, 1)
        des = hog.compute(img)
        des /= np.linalg.norm(des)

        img_new = np.asarray(img, dtype='float32') / 255
        img_new = np.expand_dims(img_new, axis=0)

        return img_new, des

    def __getitem__(self, idx):
        img, label = self._load_data(self.images_files[idx])
        return img, label

    def __len__(self):
        return len(self.images_files)
