# from crowd_count import CrowdCounter
# import network
import numpy as np
import cv2
import os
import sys

# Extend on our system's path and can load the other folder's file
sys.path.append('..')
import lib

# Use the utf-8 coded format
reload(sys)
sys.setdefaultencoding('utf-8')


# class CrowdPredictor():
#     def __init__(self, model_path):
#         self.net = CrowdCounter()
#         network.load_net(model_path, self.net)
#         self.net.cuda()
#         self.net.eval()

#     @staticmethod
#     def img_preprocess(img):
#         img_cp = img.copy()
#         ht = img_cp.shape[0]
#         wd = img_cp.shape[1]
#         ht_1 = (ht / 16) * 16
#         wd_1 = (wd / 16) * 16
#         if len(img.shape) == 3:
#             img_cp = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
#         img_f32 = img_cp.astype(np.float32, copy=False)
#         img_f32 = cv2.resize(img_f32, (int(wd_1), int(ht_1)))
#         img_f32 = img_f32.reshape((1, 1, int(ht_1), int(wd_1)))
#         return img_f32

#     def predict(self, img):
#         img_pp = self.img_preprocess(img)
#         density_map = self.net(img_pp)
#         density_map = density_map.data.cpu().numpy()
#         count = int(np.sum(density_map))
#         return count, density_map[0][0]


fps_count_i = 1
original_count_j = 1
fps = 30
video_fps = 10

final_models_path = os.path.abspath('../saved_models')

input_path = os.path.abspath('../input_original')
input_fps_path = os.path.abspath('../input_original_{}'.format(fps))

output_path = os.path.abspath('../output_original')
output_fps_path = os.path.abspath('../output_original_{}'.format(fps))

output_video_path = os.path.abspath('../outtest_1.avi')

total_num = lib.get_file_max_number(input_path)
total_fps_num = lib.get_file_max_number(input_fps_path)

print(total_num)
print(total_fps_num)

# cp = CrowdPredictor('{}/mcnn_20180426_994.h5'.format(final_models_path))
video_width = 640
video_height = 480
video_size = (video_width, video_height)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter(output_video_path, fourcc, video_fps, video_size)

for i in range(1, total_num + 1):
    file_path = '{}/{}.jpg'.format(input_path, i)
    img_file = cv2.imread(file_path, 1)

    img_height = img_file.shape[0]
    img_width = img_file.shape[1]

    half_of_height = img_height / 2
    half_of_width = img_width / 2
    # cut_img = img[min_y:max_y, min_x:max_x]
    img_area1 = img_file[0: half_of_height, 0:half_of_width]
    img_area2 = img_file[0: half_of_height, half_of_width:img_width]
    img_area3 = img_file[half_of_height:img_height, 0:half_of_width]
    img_area4 = img_file[half_of_height:img_height, half_of_width:img_width]
    # count, dens_map = cp.predict(img_file)
    # count_area1, dens_map1 = cp.predict(img_area1)
    # count_area2, dens_map2 = cp.predict(img_area2)
    # count_area3, dens_map3 = cp.predict(img_area3)
    # count_area4, dens_map4 = cp.predict(img_area4)
    # pts = ([[], []])

    yellow = (0, 255, 255)
    cv2.line(img_file, (half_of_width, 0),
             (half_of_width, img_height), yellow, 2)
    cv2.line(img_file, (0, half_of_height),
             (img_width, half_of_height), yellow, 2)
    # cv2.polylines(img_file, [pts], False, (0, 255, 255), 2)
    cv2.imshow('full', img_file)
    # cv2.imshow('area1', img_area1)
    # cv2.imshow('area2', img_area2)
    # cv2.imshow('area3', img_area3)
    # cv2.imshow('area4', img_area4)

    # text = 'count is {}'.format(count)
    # text1 = 'area 1 count is {}'.format(count_area1)
    # print(text1)
    # text2 = 'area 2 count is {}'.format(count_area2)
    # print(text2)
    # text3 = 'area 3 count is {}'.format(count_area3)
    # print(text3)
    # text4 = 'area 4 count is {}'.format(count_area4)
    # print(text4)

    cv2.waitKey(0)
    quit()
    # dens_map = cv2.resize(dens_map, (img.shape[1], img.shape[0]))
    # dens_map = 255 * dens_map / np.max(dens_map)
    # dens_map = dens_map.astype(np.uint8, copy=False)
    # for i in range(0, img.shape[0]):
    #     for j in range(0, img.shape[1]):
    #         if dens_map[i, j] > 0:
    #             cv2.circle(img, (j, i), 1, (0, 0, int(dens_map[i, j])))

    # new_img = cv2.putText(img_file, text, (50, 50),
    #                       cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    # new_img_path = '{}/{}.jpg'.format(output_path, i)
    # cv2.imwrite(new_img_path, new_img)

    # print('original \033[0;32m%s\033[0m was saved. And count is %s' % (
    #     new_img_path, count))
    # videoWriter.write(new_img)


# Store the all image file's path
# image_file_list = []
# print(input_path)
# lib.get_image_path_from_folder(input_path,
#                                image_file_list, False)
# print(image_file_list)

# for i in range(1, total_fps_num + 1):
#     file_fps_path = '{}/{}.jpg'.format(input_fps_path, i)
#     img_fps_file = cv2.imread(file_fps_path, 1)
#     count, dens_map = cp.predict(img_fps_file)

#     # 0 ~ 30 coount
#     text = 'count is {}'.format(count)

#     # new_fps_image = cv2.putText(img_fps_file, text, (50, 50),
#     #                             cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)

#     # new_fps_img_path = '{}/{}.jpg'.format(output_fps_path, i)
#     # cv2.imwrite(new_fps_img_path, new_fps_image)

#     # print('\033[0;32m%s\033[0m was saved.' % (new_fps_img_path))

#     for j in range(1, (total_num / total_fps_num + 1)):
#         file_path = '{}/{}.jpg'.format(input_path, original_count_j)
#         img_file = cv2.imread(file_path, 1)
#         new_img = cv2.putText(img_file, text, (50, 50),
#                               cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
#         new_img_path = '{}/{}.jpg'.format(output_path, original_count_j)
#         videoWriter.write(new_img)
#         # cv2.imwrite(new_img_path, new_img)
#         original_count_j = original_count_j + 1

#         print('original \033[0;32m%s\033[0m was saved.' % (new_img_path))

# videoWriter.release()
print 'finish'
