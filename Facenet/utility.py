# coding=utf-8
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import facenet
import align.detect_face
import os
import math
import time
import utility
from sklearn.decomposition import PCA

#VIDEO_USED = './run_test.avi'
VIDEO_USED = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

#读取file_dir目录下的全部图片(image_paths中),每张图片检测后只处理一张脸
def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction,file_dir):
	#file_dir = './test/' + name
	minsize = 20  # minimum size of face
	threshold = [0.6, 0.7, 0.7]  # three steps's threshold
	factor = 0.709  # scale factor

	#print('Creating networks and loading parameters')
	with tf.Graph().as_default():
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
		with sess.as_default():
			pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

	tmp_image_paths = image_paths.copy()
	img_list = []
	##################################
	faces_sizes = []
	##################################
	print('输入图片:', tmp_image_paths)
	for image in tmp_image_paths:
		img = cv2.imread(file_dir + '/' + os.path.expanduser(image))
		img_size = np.asarray(img.shape)[0:2]
		bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
		# print('**************',bounding_boxes,'***************')
		if len(bounding_boxes) < 1:
			image_paths.remove(image)
			print("不能检测到人脸, 移除图片 ", image)
			continue
		
		##################################
		faces_sizes.append((bounding_boxes[0][2] - bounding_boxes[0][0]) / (bounding_boxes[0][3] - bounding_boxes[0][1]))
		##################################

		#测试用begin
		img_clahe = utility.cal_clahe(img)
		img_o_clahe = np.hstack((img, img_clahe))
		tmp_i = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
		cv2.imwrite('./save/1clahe_%s.jpg' % str(tmp_i), img_o_clahe)
		#测试用end

		img = utility.cal_clahe(img)

		det = np.squeeze(bounding_boxes[0, 0:4])
		bb = np.zeros(4, dtype=np.int32)
		bb[0] = np.maximum(det[0] - margin / 2, 0)
		bb[1] = np.maximum(det[1] - margin / 2, 0)
		bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
		bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
		cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
		aligned = cv2.resize(cropped, (image_size, image_size))
		tmp_i = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
		cv2.imwrite('./save/cropped%s.jpg' % str(tmp_i),cropped)
		cv2.imwrite('./save/aligned%s.jpg' % str(tmp_i),aligned)   
		prewhitened = facenet.prewhiten(aligned)
		img_list.append(prewhitened)

	images = []
	if(len(img_list) > 0):
		images = np.stack(img_list)

	return images, faces_sizes
#######################################################################################################	
	
def find_chara_point(emb):
    # n = emb.shape[0]
    m = emb.shape[0]
    # e是每张图片求出的向量在单位球面上投影的坐标，它们的长度本身就是１
    e = emb[0]
    for i in range(m-1):
        e = e + emb[i+1]
    # avere是e的均值
    avere = e/m
    # avere单位化，然后再乘以模长的均值averl
    out = avere / np.linalg.norm(avere)
    return out
	
#######################################################################################################

def dist(a, b):
    # return np.sqrt(np.sum(np.square(a - b)))
    return 1 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
	
#######################################################################################################

def save_data_csv(face_size, cp, name):
	overwrite = ''

	new_list = []
	new_list.append(name)
	###################################
	new_list.append(face_size)
	###################################
	for i in range(cp.size):
		new_list.append(cp[i])
	new_df = pd.DataFrame(new_list).T

	if('Client_Character.csv' in os.listdir(os.getcwd())):
		data_df = pd.read_csv('./Client_Character.csv', header=None,index_col=False)

		if(data_df[data_df[0]==name].shape[0] > 0):
			print('Already Exist')
			print('Over Write? [Y/N]')
			overwrite = input()
			while (overwrite != 'Y') and (overwrite != 'N'):
				print('Wrong Input! Please type Y or N .')
				print('Over Write? [Y/N]')
				overwrite = input()
				break

			if overwrite == 'Y':
				add_i = data_df[data_df[0]==name].index[0]
				data_df.drop(data_df.index[add_i],inplace=True)
			elif overwrite == 'N':
				return

		data_df = data_df.append(new_df)
		data_df.index = [i for i in range (data_df.shape[0])]
	else:
		data_df = new_df
	
	data_df.to_csv('./Client_Character.csv',index=None,header=None)
	print('已存储')

#######################################################################################################
def get_smallist(a):
    l = len(a)
    num = 0
    m = a[num]
    for i in range(l-1):
        if a[i+1] < m and a[i+1] != 0:
            m = a[i+1]
            num = i+1
    return num
#######################################################################################################

def get_smallist_and_second_smallist(a):
    l = len(a)
    fst = 0
    snd = 0
    m = a[fst]
    n = a[snd]
    for i in range(l-1):
        if a[i+1] < m and a[i+1] != 0:
            m = a[i+1]
            fst = i+1
    
    if (fst == 0):
        n = a[1]
        snd = 1
    for i in range(l-1):
        if (a[i + 1] < n and (i+1) != fst and a[i + 1] != 0):
            n = a[i+1]
            snd = i + 1
    return fst, snd
#################################################################################################

def simplest_color_balance(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = 3 / 200.0

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height

        flat = channel.reshape(vec_size)
        flat = np.sort(flat)
        n_cols = flat.shape[0]

        low_val = flat[math.floor(n_cols * half_percent)]
        high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

        low_mask = channel < low_val
        masked = np.ma.array(channel, mask=low_mask, fill_value=low_val)
        channel = masked.filled()

        high_mask = channel > high_val
        masked = np.ma.array(channel, mask=high_mask, fill_value=high_val)
        channel = masked.filled()

        normalized = cv2.normalize(channel, channel.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)
#################################################################################################


# 根据脸长宽比调整阈值，a,b分别是预存图象和新拍摄图象的脸部宽长比，返回值惩罚系数，它将乘在阈值上
def face_size_adjust_function(a,b):
    sigma = 0.2362
    return 1/(1+np.exp(5 * np.abs(a-b)/sigma - 10))

#################################################################################################


# 降噪增强对比度:contrast local adaptive histogram equalization
def cal_clahe(img):

    #双边滤波
    img_bf = cv2.bilateralFilter(img,9,75,75)

    img_lab = cv2.cvtColor(img_bf, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(img_lab)

    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))

    lab_planes[0] = clahe.apply(lab_planes[0])

    img_lab2 = cv2.merge(lab_planes)

    img_CLAHE = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    return img_CLAHE
	
##################################################################################


def mkdir(path):

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

##################################################################################


def pre(img, bounding_boxes):
    # bounding_boxes就是faces
    margin = 44
    image_size = 160

    ''' 增加双边滤波
    img = utility.cal_clahe(img)
    '''

    img_size = np.asarray(img.shape)[0:2]
    img_list = []
    cropped_faces = []
    count = 0
    if len(bounding_boxes) < 1:
        return
    for bounding_box in bounding_boxes:
        # det = np.squeeze(bounding_box[0, 0:4])
        det = bounding_box
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = cv2.resize(cropped, (image_size, image_size))
        prewhitened = facenet.prewhiten(aligned)
        cropped_faces.append(cropped)
        count = count + 1
        img_list.append(prewhitened)
    images = np.stack(img_list)
    return images, cropped_faces

##################################################################################


def get_prediction_by_svc(emb, model, class_names):
    emb = emb[np.newaxis, : ]  # model.predict_proba的输入格式是(?,512)
    predictions = model.predict_proba(emb)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

    return class_names[best_class_indices[0]], best_class_probabilities[0]

##################################################################################

def one_by_one_search(data, distance_i, emb_i):
    name_list = data[0]
    for j in range(len(name_list)):
        distance_i[j] = dist(data.loc[j][1:].astype(float), emb_i)
        # distance_adjust[i][j] = utility.face_size_adjust_function(faces_sizes[i], data.loc[j][1])
        # distance[i][j] = distance[i][j] * distance_adjust[i][j]
    s, t = get_smallist_and_second_smallist(distance_i)

    return s, t

#####################################################################################

num_of_group = 32
num_of_dim_in_each_group = int(512/num_of_group)

def get_statistic_method_group_number(emb):
    a = np.zeros(num_of_group)
    for i in range(num_of_group):  # 32 groups
        for j in range(num_of_dim_in_each_group):  # 512/32=16 dimensoins in each group
            # print('AAAA',line[2 + j + 32*i])
            if float(emb[j + num_of_dim_in_each_group * i]) > 0.044:
                a[i] += 1
    a = np.array(a)
    index = np.argmax(a)

    return index

#####################################################################################

def grouping_by_statistic_method(data, distance_i, emb_i):
    group_num = get_statistic_method_group_number(emb_i)
    with open("./Group_data_6000.txt", "r") as group_data:
        line = group_data.readline()
        while line:
            line = line.replace(' ', '').replace('\n', '')
            line = np.array(line.split(','))
            if int(line[0])==group_num:
                for j in range(len(line)):
                    if line[j]:
                        distance_i[j] = dist(data.loc[int(line[j])][1:].astype(float), emb_i)
            line = group_data.readline()
    s, t = get_smallist_and_second_smallist(distance_i)
    return s, t

##################################################################################### K-mean
N_SAMPLES = 5760
N_DIM = 512


def kmean_clustered_data(cluster_file):
    """
    Argument:
    data_file: string, name of a data file
    cluster_file: string, name of a cluster data file, whose each line starts with the cluster number and contains indices belong to the cluster.

    Return:
    clst_data: list of numpy arrays, each array has the data belongs to the cluster.
    clst_ids: list of list of strings, each array has the ids belong to the clustsr
    """
    ids = []
    data = np.zeros((N_SAMPLES, N_DIM))
    #with open('Client_Character_6000.csv', newline='') as csvfile:
    #    reader = csv.reader(csvfile, delimiter=',')
    reader = pd.read_csv('./Client_Character_6000.csv', header=None)

    #for i, row in enumerate(reader):
    #    ids.append(row[0])
    #    data[i, :] = np.array([e for e in row[1:]]).astype(float)
    for i in range(len(reader)):
        ids.append(reader.loc[i][0])
        data[i, :]= np.array(reader.loc[i][1:]).astype(float)
    clst_data = []
    clst_ids = []
    with open(cluster_file, 'r') as f:
        for clst, lin in enumerate(f):
            lin = lin.replace('\n', '')
            indices = np.array([e for e in lin.split(', ')[:-1]][1:], dtype=int)
            clst_data.append(data[indices])
            clst_ids.append([id for i, id in enumerate(ids) if i in indices])
    return [clst_data, clst_ids]


def kmean_centroids(clst_data):
    """
    Argument:
    clst_data: list of numpy arrays, each array has the data belongs to the cluster.

    Return:
    cs: numpy array, array of centroids of clusters
    """
    cs = []
    for clst in clst_data:
        cs.append(np.mean(clst, axis=0))
    return np.array(cs)


def kmean_lookup(target, centroids):
    """
    Argument:
    target: 1x512 numpy array, the target data to be classified.
    centroids: k x 512 numpy array, centroids for each cluster

    Return:
    clst: int, the cluster target belongs to.
    """
    clst = np.argmin(np.linalg.norm(centroids - target, axis=1))
    return clst

def grouping_by_kmean(data, centriuds_data, distance_i, emb_i):
    group_num = kmean_lookup(emb_i, centriuds_data) # look up function

    with open("./Client_Character_6000_kmean_75_cluster.txt", "r") as group_data:
        line = group_data.readline()
        while line:
            line = line.replace(' ', '').replace('\n', '')
            line = np.array(line.split(','))
            if int(line[0]) == group_num:
                for j in range(len(line)):
                    if line[j]:
                        distance_i[j] = dist(data.loc[int(line[j])][1:].astype(float), emb_i)
            line = group_data.readline()
    s, t = get_smallist_and_second_smallist(distance_i)
    return s, t

#####################################################################################

# Pass in a single row array, numbers only, no names
# For example, [1,2,3,4]
# It can be numpy array or regular Python array
def pca_singleVector(singleVector):
    #if singleVector.shape[0] != 512:
    #    return

    reshaped = np.reshape(singleVector, (8, 64))
    pca = PCA(n_components = 8)
    reduced = pca.fit_transform(reshaped)
    return reduced.flatten()

def reduce_dimension_PCA(distance_i, emb_i):
    data = pd.read_csv('./PCA_client_character.csv', header=None)
    pca_emb_i = pca_singleVector(emb_i)
    s, t = one_by_one_search(data, distance_i, pca_emb_i)
    return s, t


#####################################################################################