
import warnings
from skimage.morphology import disk
from skimage.filters.rank import entropy
import torch.nn as nn
import torch
import os
import random
import re
import time

import cv2
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"


warnings.filterwarnings('ignore')

seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def check_match(im_lab_k, db_lab, num_include):

    if num_include == 1:
        if db_lab == im_lab_k:
            return True
    else:
        if (int(db_lab)-num_include/2) <= int(im_lab_k) and int(im_lab_k) <= (int(db_lab)+num_include/2):
            return True

    return False


def regions_filter(image, max_choose):
    if(len(image.shape) > 2):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (512, 512))
    img_height, img_width = image.shape
    cell_size = 32

    entropy_image = entropy(image, disk(3))

    local_goodness = np.zeros(
        [int(img_height/cell_size), int(img_width/cell_size)], dtype=np.float32)

    for i in range(int(img_height/cell_size)):
        for j in range(int(img_width/cell_size)):
            local_goodness[i, j] = np.sum(
                entropy_image[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]) / (cell_size*cell_size)
    local_goodness_flatten = local_goodness.flatten()
    local_choose = np.argpartition(
        local_goodness_flatten, -max_choose)[-max_choose:]
    local_min = np.min(local_goodness_flatten[local_choose])
    regions = np.where(local_goodness >= local_min)

    hog = cv2.HOGDescriptor((16, 16), (16, 16), (16, 16), (8, 8), 9, 1)
    hog_labels_list = []

    multichannel_images = np.zeros(
        (cell_size, cell_size, max_choose), dtype=np.uint8)
    for channel in range(max_choose):
        x = regions[1][channel]*cell_size
        y = regions[0][channel]*cell_size
        multichannel_images[:, :, channel] = image[y:y +
                                                   cell_size, x:x+cell_size]

        des = hog.compute(multichannel_images[:, :, channel])
        hog_labels_list.append(des.tolist())

    hog_label = np.array(hog_labels_list).transpose()
    multichannel_images = multichannel_images.transpose(2, 0, 1)
    multichannel_images = np.asarray(
        multichannel_images, dtype='float32') / 255

    return multichannel_images, hog_label


def predict(model, net_name, img, max_choose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image to Tensor
    start_time = time.time()
    img, _ = regions_filter(img, max_choose)

    model.eval()
    torch.cuda.synchronize()

    img_input = torch.from_numpy(img).unsqueeze(dim=0).transpose(1, 0)
    img_input = img_input.to(device)
    des = model(img_input)
    des = des.data.cpu().numpy().squeeze()
    output = np.array(des, dtype=np.float32)

    torch.cuda.synchronize()
    end_time = time.time()
    cost_time = end_time - start_time

    return output, cost_time


def matrix_binaryzation(matrix_input):
    indices = np.argmax(matrix_input, axis=1)
    indices = np.expand_dims(indices, axis=1)
    output = np.zeros_like(matrix_input)
    np.put_along_axis(output, indices, 1, axis=1)

    return output


def matrix_diagonal_one(matrix_bin, matrix_eye):
    return np.multiply(matrix_bin, matrix_eye)


def get_single_prec_recall(data_path, params_path, num_include, algorithm, max_choose, num_diagonal):
    db_path = data_path + "/database"
    query_path = data_path + "/query"
    print('>> eval algorithm:', algorithm)
    print(">> database path:", db_path)
    print(">> query path:", query_path)

    mem_files = [os.path.join(db_path, f) for f in sorted(
        os.listdir(db_path), key=lambda f: int(re.sub('\D', '', f)))]
    query_files = [os.path.join(query_path, f) for f in sorted(
        os.listdir(query_path), key=lambda f: int(re.sub('\D', '', f)))]

    all_descr = []
    all_labels = []
    all_extract_time = []
    all_query_time = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_model = torch.load(params_path)
    from network import CoCALC
    model = CoCALC().to(device)
    model.load_state_dict(loaded_model)

    print('>> load image to database')
    for fl1 in mem_files:
        all_labels.append(
            re.match('.*?([0-9]+)$', os.path.splitext(os.path.basename(fl1))[0]).group(1))

        img_db = cv2.imread(fl1)
        descr, cost_time = predict(model, algorithm, img_db, max_choose)

        all_descr.append(descr)
        all_extract_time.append(cost_time)

    correct = np.zeros(len(query_files), dtype='uint8')
    scores = np.zeros(len(query_files))

    print('>> dimensions: ', descr.shape)

    k = 0

    print(">> band = ", num_diagonal)
    num_in_band = max_choose + num_diagonal*(2*max_choose-num_diagonal-1)

    matrix_eye = np.eye(max_choose)
    for i in range(1, num_diagonal):
        matrix_eye_new1 = np.eye(max_choose, k=i)
        matrix_eye_new2 = np.eye(max_choose, k=-i)
        matrix_eye = matrix_eye + matrix_eye_new1 + matrix_eye_new2

    print('>> Start querying.')
    for fl2 in query_files:
        im_label_k = re.match(
            '.*?([0-9]+)$', os.path.splitext(os.path.basename(fl2))[0]).group(1)
        img_query = cv2.imread(fl2)

        descr, cost_time = predict(model, algorithm, img_query, max_choose)

        max_sim = -1.0
        i_max_sim = -1

        start_time = time.time()
        for i in range(len(all_descr)):
            matrix_dot = cosine_similarity(descr, all_descr[i])
            matrix_bin = matrix_binaryzation(matrix_dot)
            matrix_pool = matrix_diagonal_one(matrix_bin, matrix_eye)
            curr_sim = np.sum(matrix_pool)/num_in_band + \
                np.random.uniform() * 1e-6

            if curr_sim > max_sim:
                max_sim = curr_sim
                i_max_sim = i

        end_time = time.time()
        cost_time = end_time - start_time
        all_query_time.append(cost_time/len(all_descr))

        scores[k] = max_sim
        if check_match(im_label_k, all_labels[i_max_sim], num_include):
            correct[k] = 1
        print("Proposed match:", im_label_k, ", ",
              all_labels[i_max_sim], ", score = ", max_sim, ", Correct =", correct[k])

        k += 1

    precision, recall, _ = precision_recall_curve(correct, scores)
    print(">> Auc = ", auc(recall, precision))

    mean_extract_time = np.sum(
        np.array(all_extract_time)) / np.size(np.array(all_extract_time))
    mean_query_time = np.sum(
        np.array(all_query_time)) / np.size(np.array(all_query_time))
    print(">> Mean features extraction time = ", mean_extract_time*1000)  # ms
    print(">> Mean query time = ", mean_query_time*1000)

    print('\n')

    return precision, recall
