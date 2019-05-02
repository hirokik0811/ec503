'''
Created on Apr 12, 2019

@author: Yilin Dong
'''
# coding=utf-8
import argparse
import sys
import time
import align
import tensorflow as tf
import align.detect_face
import cv2
import numpy as np
import pandas as pd
import facenet
import os
import utility
import pickle
from sklearn.decomposition import PCA

##########################################################################################################  parameters
face_detect_threshold = 0.95  # between 0 and 1, the larger the stricter
MAX_FACE_NUM = 50  # at most 50 faces per frame
distance_approve_thresholds = np.zeros((1,MAX_FACE_NUM))
distance_disapprove_threshold = 0.25
distance_parameter = 0.15  #the smallest distant need to be at most 0.35 time of the second large distant
SVM_prob_approve_threshold = 0.80
SVM_prob_disapprove_threshold = 0.50
blurry_threshold = 100  # The minimum clarity of the face accepted. The lower, the more blurred face accepted.

model = './20180402-114759'
data = pd.read_csv('./Client_Character_6000.csv', header=None)


######################################################################################################## preprocessing
def pre(img,bounding_boxes,facial_features):
    # bounding_boxes is the faces
    margin = 44
    image_size = 160
    facial_features = np.transpose(facial_features)
    #################################### Bilateral filtering
    img = utility.cal_clahe(img)
    ####################################
    img_size = np.asarray(img.shape)[0:2]
    img_list = []
    count = 0
    if len(bounding_boxes) < 1:
        return
    valid_bounding_boxes = []
    valid_facial_features = []
    for i in range(len(bounding_boxes)):
        det = bounding_boxes[i]
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        aligned = cv2.resize(cropped, (image_size, image_size))
        #################################### Image sharpness detection
        gray_pic = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        blurry_para = cv2.Laplacian(gray_pic, cv2.CV_64F).var()
        if blurry_para < blurry_threshold:
            print('Too blury！')
            continue
        ####################################
        prewhitened = facenet.prewhiten(aligned)
        #################################### out put log
        # tmp_i = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
        # cv2.imwrite('./save/cropped%s.jpg' % str(tmp_i),cropped)
        # cv2.imwrite('./save/aligned%s.jpg' % str(tmp_i),aligned)
        count = count + 1
        img_list.append(prewhitened)
        valid_bounding_boxes.append(bounding_boxes[i])
        valid_facial_features.append(facial_features[i])
    images = []
    if len(img_list) != 0:
        images = np.stack(img_list)
    valid_facial_features = np.transpose(valid_facial_features)

    return images, valid_bounding_boxes, valid_facial_features


##########################################################################################################
def set_threshold(faces):
    thresholds = np.zeros((1,MAX_FACE_NUM))
    i = 0
    for face in faces:
        face_w = (face[2]-face[0]).astype(int)
        face_h = (face[3]-face[1]).astype(int)

        if face_w/utility.FRAME_WIDTH > 0.8 or face_h/utility.FRAME_HEIGHT > 0.8:
            thresholds[0][i] = 0.25
        elif face_w/utility.FRAME_WIDTH < 0.1 or face_h/utility.FRAME_HEIGHT < 0.1:
            thresholds[0][i] = 0.25
        elif face_h/face_w < 1.4:
            thresholds[0][i] = 0.25
        else:
            thresholds[0][i] = 0.2

        print('detected face:%d x %d of scale %.4f with threshold %.2f'%(face_w,face_h,face_h/face_w,thresholds[0][i]))

        i += 1

    return thresholds


#####################################################################################################  get 512 dim vector
def get_emb(images):
    sess = tf.get_default_session()
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb


##################################################################################################### print label and blank frame
def recognize(emb,distance_approve_thresholds,faces_sizes):

    NUM_OF_FACES = np.shape(emb)[0]
    names = []
    smallist_dist = []
    sec_smallist_dist = []
    # names is the names of person in current frame
    # data: a global variable, is the all user data

    distance = np.zeros((8000, 8000))
    distance_adjust = np.zeros((8000,8000))

    print('Total number of faces: ', NUM_OF_FACES)
    i = 0
    while i < NUM_OF_FACES:
        print('{}-th face：'.format(i+1))
        ##################################### recognize with SVM first
        st = time.time()
        suspect_name, prob = utility.get_prediction_by_svc(emb[i], model_svm, class_names)
        name_index = data[data[0] == suspect_name].index
        # assert len(name_index) != 0, 'The name found by SVM is not in csv file!'
        # assert len(name_index) < 2, 'The name found by SVM has more than 1 Chara Point in csv file!'
        # name_num = name_index[0]  # name_num: int type, is the number of the line where the SVM finds the name in the CSV file.
        # print('SVM：%s，SVM confidence：%.4f'%(suspect_name, prob))
        # suspect_charapoint = data.loc[name_num][2:]
        et = time.time()
        # print('SVM time cost：%.8f s'%(et - st))

        ###################################### Find the nearest point and the second near point by comparing one by one
        # **************** we modify this part

        detect_start_time = time.clock()

        # s, t = utility.one_by_one_search(data, distance[i], emb[i])
        # s, t = utility.grouping_by_statistic_method(data, distance[i], emb[i])
        s, t = utility.grouping_by_kmean(data, centriuds_data, distance[i], emb[i])
        # s, t = utility.grouping_by_LSH(data, distance[i], emb[i])
        # s, t = utility.reduce_dimension_PCA(distance[i], emb[i]) # pca use its own pregenerated data

        detect_end_time = time.clock()

        print('Comparing time cost: %.2fs' % float(detect_end_time - detect_start_time))


        print('nearest person:%s %s, %.4f' % (s, data[0][s], distance[i][s]))
        smallist_dist.append(distance[i][s])
        if (len(data[0]) > 1):
            print('second nearest person:%s %s, %.4f' % (t, data[0][t], distance[i][t]))
            second_near = distance[i][t]
        else:
            second_near = 10
        sec_smallist_dist.append(second_near)

        #########################################
        # The comparison on dist and SVM are performed side by side.
        # Consider both the results and both have a veto power.
        DIST_approve = 0
        SVM_approve = 0

        if distance[i][s] < distance_approve_thresholds[0][i] and second_near > (1 + distance_parameter) * distance[i][s]:
            DIST_approve = 1
        elif distance[i][s] > distance_disapprove_threshold:
            DIST_approve = -1

        if prob > SVM_prob_approve_threshold:
            SVM_approve = 1
        elif prob < SVM_prob_disapprove_threshold:
            SVM_approve = -1

        sec_smallist_dist.append(second_near)

        if SVM_approve==1 or SVM_approve==0:
            names.append(suspect_name)
        #elif DIST_approve == -1 or SVM_approve == -1:
            # print('At leaset one test disapprove')
        elif DIST_approve == 1 and SVM_approve == 1:
            if suspect_name == data[0][s]:
                # print('Approved by both tests')
                names.append(suspect_name + '(*)')
            elif suspect_name != data[0][s]:
                # print('Approved by both tests with different label')
                names.append(' ')
        elif DIST_approve == 1:
            # print('Aprove by dist test, not significant for SVM')
            # names.append(data[0][s] + '(dist)')
            names.append(data[0][s])
        elif DIST_approve == -1 and SVM_approve == -1:
            # print('Significant for SVM，not approved by dist')
            # names.append(suspect_name + '(SVM)')
            names.append(' ')
        elif SVM_approve == 0 and DIST_approve == 0:
            # print('Not significant for both SVM and distance test')
            names.append(suspect_name)
        else:
            print('!!', DIST_approve, SVM_approve, suspect_name, data[0][s])
            assert False
        # cv2.imwrite('./save/%s.jpg' % str(i),images[i])
        i += 1
    return names, smallist_dist, sec_smallist_dist  # name_by_svm, dist_of_name_by_svm


#####################################################################################################  detect glasses
def glasses_detect(frame,facial_features, names, i):  # detect whether i-th face has a glasses
    # print(np.shape(facial_features))
    # facial_features is a 10 dimensional vector，it's the same as the outer variable facial_features[:][i]
    # the reason why i is a input of this function, is that this function print log to the screen, it needs to know where to put those logs.
    mg = 0.5 * ((facial_features[0] - facial_features[1]) ** 2 +
                (facial_features[5] - facial_features[6]) ** 2) ** 0.5
    img_size = np.asarray(frame.shape)[0:2]
    zuoshangx = np.maximum(facial_features[0] - mg, 0)
    zuoshangy = np.maximum(np.minimum(facial_features[6], facial_features[5]) - 0.5 * mg, 0)
    youxiax = np.minimum(facial_features[1] + mg, img_size[1])
    youxiay = np.minimum(np.maximum(facial_features[6], facial_features[5]) + 0.35 * mg, img_size[0])
    #

    # eye area blurry parameter
    cropped = frame[int(zuoshangx): int(youxiax), int(zuoshangy): int(youxiay), :]
    gray_pic = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    if gray_pic is not None:
        blurry_para = cv2.Laplacian(gray_pic, cv2.CV_64F).var()
    else:
        blurry_para = 0
    # full face blurry parameter
    # cropped_face = frame[int(face_bb[0])+20 : int(face_bb[2])-20, int(face_bb[1])+20 : int(face_bb[3])-20, :]
    # gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
    # blurry_para_face = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    #
    glasses = 'without glasses'
    if blurry_para > 150:
        glasses = ': with glasses '
        cv2.rectangle(frame, (int(zuoshangx), int(zuoshangy)), (int(youxiax), int(youxiay)), (0, 0, 255), 1)
    if (len(names[i].strip()) != 0):
        # print the blurry parameter on the top left corner
        cv2.putText(frame, names[i] + glasses + str(int(blurry_para)), (0, 100 * (i + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)


##################################################################################################### Print blank and label on each frame
def add_overlays(frame, faces, facial_features, frame_rate, names, smallist_dist):

    i = 0
    # faces.shape[0] the number of faces on current frame
    while i < np.shape(faces)[0]:  # faces.shape[0]:
        # the blank for each face
        face_bb = faces[i].astype(int)
        if ('likely' in names[i]) or len(names[i].replace(' ',''))==0:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame,
                      (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                      color, 2)
        # output landmark(eyes, mouth and nose)
        for j in range(5):
            cv2.rectangle(frame,
                          (int(facial_features[j][i])-1, int(facial_features[j+5][i])-1),
                          (int(facial_features[j][i])+1, int(facial_features[j+5][i])+1), color, 3)

        print('detect frame {} x {} of {}'.format(face_bb[2]-face_bb[0],face_bb[3]-face_bb[1],names[i]))
        ##################################################################
        glasses_detect(frame, np.transpose(facial_features)[i][:], names, i)  # Facial_features only brings in the data of the i-th face, and the names bring in all

        ##################################################################
        # print name on each frame
        if len(names[i].strip()) != 0:
            cv2.putText(frame, names[i]+' '+str(1-round(smallist_dist[i],2)), (face_bb[0], face_bb[3]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,thickness=2, lineType=2)
        i += 1

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
        thickness=2, lineType=2)


########################################################################################################## main
def save_face_data(w, h, right_eye, left_eye, smallist_dist, sec_smallist_dist, name, threshold):
    if name==' ' or 'likely' in name:
        return
    slope = (left_eye[1] - right_eye[1])/(left_eye[0] - right_eye[0])
    txt = open('./face_size_data/'+name, 'a')
    txt.write(str(w)+'\t'+str(h)+'\t'+'%.4f'%(w/h)+'\t'+ '%.2f'%slope + '\t' +
              '%.4f'%smallist_dist + '\t' + '%.4f'%sec_smallist_dist + '\t' + str(threshold))
    txt.write('\n')
    txt.close()


########################################################################################################## main
def main():
    ############################################### Frame rate related parameters to be used when displaying the face frame
    fps_display_interval = 0.5  # seconds
    frame_rate = 0
    frame_count = 0

    ############################################### The parameters to be used for the MTCNN(the first network, the algorithm that find faces)
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    ###############################################  Select camera and set image resolution
    video_capture = cv2.VideoCapture(utility.VIDEO_USED)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, utility.FRAME_WIDTH)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, utility.FRAME_HEIGHT)

    ############################################### Frame-by-frame processing
    start_time = time.time()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        ############################################### preprocessing on each frame
        frame = cv2.flip(frame, 1)  # Mirror flip
        frame = utility.simplest_color_balance(frame, 5)  # white balance

        ###############################################
        print('*************************************')
        print('\nDetecting...')
        detect_start_time = time.time()
        unsifted_faces, unsifted_facial_features = align.detect_face.detect_face(frame, minsize, pnet, rnet, onet,
                                                                             threshold, factor)
        faces = np.zeros([0, 5])
        facial_features = np.empty([10, 0])
        if len(unsifted_faces) != 0:
            for i in range(np.shape(unsifted_faces)[0]):
                if unsifted_faces[i][4] > face_detect_threshold:  # Detecting faces and deleting, leaves only those with high confidence
                    faces = np.row_stack((faces, unsifted_faces[i]))
                    facial_features = np.column_stack((facial_features, np.transpose(unsifted_facial_features)[i]))
        detect_end_time = time.time()
        print('Face recognition time cost: %.2fs'%float(detect_end_time - detect_start_time))

        if faces is not None and faces.shape[0] != 0:  # Ensure that the face that meets the requirements exists
            ############################################### preprcessing on face
            images, faces, facial_features = pre(frame, faces, facial_features)  # Cut the face from the original frame, remove those blury ones, and then resize to the size required by facenet
            distance_approve_thresholds = set_threshold(faces)  # Set different thresholds for faces of different sizes

            if len(images) != 0:
                faces_sizes = np.zeros((np.shape(images)[0]))  # Get the size of each face
                for i in range(np.shape(images)[0]):
                    faces_sizes[i] = (faces[i][2] - faces[i][0]) / (faces[i][3] - faces[i][1])
                ############################################### Get the 512 vector of each face
                emb = get_emb(images)

                ############################################### get the label from 512 vector (by different methods, eg.facenet, SVM, ...)
                names, smallist_dist, sec_smallist_dist = recognize(emb, distance_approve_thresholds, faces_sizes)

            ############################################### Adjust the frame rate
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

            ############################################### output box(box that surround faces)
            if len(images) != 0:
                add_overlays(frame, faces, facial_features, frame_rate, names, smallist_dist)

            ############################################### save log to./face_size_data/xxx.txt）
            while i < np.shape(images)[0]:
                right_eye = [facial_features[0][i], facial_features[5][i]]
                left_eye = [facial_features[1][i], facial_features[6][i]]
                face_bb = faces[i].astype(int)
                save_face_data(face_bb[2] - face_bb[0], face_bb[3] - face_bb[1],
                               right_eye, left_eye,
                               smallist_dist[i], sec_smallist_dist[i],
                               names[i], distance_approve_thresholds[0][i])
                i += 1

        frame_count += 1

        # display
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # preprocessing for kmean
    # with open('Client_Character_6000.csv', newline='') as csvfile:
    # reader = pd.read_csv('./Client_Character_6000.csv', header=None)
    [clst_data, clst_ids] = utility.kmean_clustered_data("./Client_Character_6000_kmean_75_cluster.txt")
    centriuds_data = utility.kmean_centroids(clst_data)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess1.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess1, None)

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)).as_default():
            facenet.load_model(model)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            ##################### load SVM
            classifier_filename_exp = os.path.expanduser('./classifier/clfy.pkl')
            with open(classifier_filename_exp, 'rb') as infile:
                (model_svm, class_names) = pickle.load(infile)
                main()
