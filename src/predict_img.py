from crowd_count import CrowdCounter
import network
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


class CrowdPredictor():
    def __init__(self, model_path):
        self.net = CrowdCounter()
        network.load_net(model_path, self.net)
        self.net.cuda()
        self.net.eval()

    @staticmethod
    def img_preprocess(img):
        img_cp = img.copy()
        ht = img_cp.shape[0]
        wd = img_cp.shape[1]
        ht_1 = (ht / 16) * 16
        wd_1 = (wd / 16) * 16
        if len(img.shape) == 3:
            img_cp = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
        img_f32 = img_cp.astype(np.float32, copy=False)
        img_f32 = cv2.resize(img_f32, (int(wd_1), int(ht_1)))
        img_f32 = img_f32.reshape((1, 1, int(ht_1), int(wd_1)))
        return img_f32

    def predict(self, img):
        img_pp = self.img_preprocess(img)
        density_map = self.net(img_pp)
        density_map = density_map.data.cpu().numpy()
        count = int(np.sum(density_map))
        return count, density_map[0][0]


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

cp = CrowdPredictor('{}/mcnn_20180426_994.h5'.format(final_models_path))
img_size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter(output_video_path, fourcc, video_fps, img_size)


for i in range(1, total_num + 1):
    file_path = '{}/{}.jpg'.format(input_path, i)

    img_file = cv2.imread(file_path, 1)

    count, dens_map = cp.predict(img_file)

    text = 'count is {}'.format(count)
    new_img = cv2.putText(img_file, text, (50, 50),
                          cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
    new_img_path = '{}/{}.jpg'.format(output_path, i)
    cv2.imwrite(new_img_path, new_img)

    print('original \033[0;32m%s\033[0m was saved. And count is %s' % (
        new_img_path, count))
    videoWriter.write(new_img)


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

videoWriter.release()
print 'finish'
