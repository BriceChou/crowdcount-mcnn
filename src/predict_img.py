from crowd_count import CrowdCounter
import network
import numpy as np
import cv2
import os
import sys
import time

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

    def predict(self, img):
        img_pp = self.img_preprocess(img)
        density_map = self.net(img_pp)
        density_map = density_map.data.cpu().numpy()
        count = int(np.sum(density_map))
        return count, density_map[0][0]

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

    @staticmethod
    def line_img_area(img, start_location=(0, 0), size=(2, 2)):
        '''
        @parameters 
           start_location: where start to line current image
           size: line current image to how many part

        @result
            return : a new image matrix
        '''
        if size[0] < 0 or size[1] < 0:
            print('x_size or y_size can\'t allow to set zero.')
            return None

        start_x = start_location[0]
        start_y = start_location[1]
        x_area_unit = int(img.shape[1] / size[0])
        y_area_unit = int(img.shape[0] / size[1])

        # Draw X axis
        for x in range(0, size[0] - 1):
            y_axis = start_y + y_area_unit
            cv2.line(img, (start_location[0], y_axis),
                     (img.shape[1], y_axis), (0, 0, 255))
            start_y = y_axis

        # Draw Y axis
        for y in range(0, size[1] - 1):
            x_axis = start_x + x_area_unit
            cv2.line(img, (x_axis, start_location[1]),
                     (x_axis, img.shape[0]), (0, 0, 255))
            start_x = x_axis

        lined_img = img.copy()
        return lined_img

    @staticmethod
    def split_img_area(img_mat, start_location=(0, 0), size=(2, 2)):
        '''
        @parameters 
           start_location: where start to split current image
           size: split current image to how many part

        @result
            return a list for all image area
        '''
        if size[0] < 0 or size[1] < 0:
            print('x_size or y_size can\'t allow to set zero.')
            return None

        start_x = start_location[0]
        start_y = start_location[1]
        x_area_unit = int(img_mat.shape[1] / size[0])
        y_area_unit = int(img_mat.shape[0] / size[1])

        all_area = []
        for x in range(0, size[0]):
            y_sum_area_unit = start_y + y_area_unit
            for y in range(0, size[1]):
                x_sum_area_unit = start_x + x_area_unit
                all_area.append(img_mat[start_y:y_sum_area_unit,
                                        start_x:x_sum_area_unit])
                start_x = x_sum_area_unit

            start_x = start_location[0]
            start_y = y_sum_area_unit

        return all_area

        @staticmethod
        def video2img(input_video_path, output_path):
            cap = cv2.VideoCapture(input_video_path)
            count = 0
            rval = cap.isOpened()
            # fps = 1
            while rval:
                count = count + 1
                rval, frame = cap.read()
                # if(count%fps == 0):
                #     cv2.imwrite('{}/{}.jpg'.format(output_path, count), frame)
                if rval:
                    img_path = '{}/{}.jpg'.format(output_path, count)
                    cv2.imwrite(img_path, frame)
                    cv2.waitKey(1)
                else:
                    break
            cap.release()


# count for image
# fps_count_i = 1
# original_count_j = 1


# How many fps to get the iamge for original video
fps = 30

# Config the model weights path
final_models_path = os.path.abspath('../saved_models')

# Config the input image folder
input_path = os.path.abspath('../input_original')
input_fps_path = os.path.abspath('../input_original_{}'.format(fps))

# Config the output image folder
output_path = os.path.abspath('../output_original')
output_fps_path = os.path.abspath('../output_original_{}'.format(fps))

# Cofig the output video folder
video_output_path = os.path.abspath('../output_video')
video_output_fps_path = os.path.abspath('../output_video_{}'.format(fps))

# Get the folder file counts
total_num = lib.get_file_max_number(input_path)
total_fps_num = lib.get_file_max_number(input_fps_path)

# Config the output video parameter
video_fps = 10
video_size = (960, 540)
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
output_video_path = '{}/dl_crowdcount_{}.avi'.format(
    video_output_path, current_time)
videoWriter = cv2.VideoWriter(output_video_path, fourcc, video_fps, video_size)

# initializer for current class
cp = CrowdPredictor('{}/mcnn_20180426_994-4.h5'.format(final_models_path))

# When we start this predict, should confirm the path and size
# Start to predict all images one by one
for i in range(1, total_num + 1):
    file_path = '{}/{}.jpg'.format(input_path, i)
    img_file = cv2.imread(file_path, 1)
    if img_file is None:
        continue

    count, dens_map = cp.predict(img_file)
    # text = 'count is {}'.format(count)
    # Draw the thermodynamic chart
    # dens_map = cv2.resize(dens_map, (new_img.shape[1], new_img.shape[0]))
    # dens_map = 255 * dens_map / np.max(dens_map)
    # dens_map = dens_map.astype(np.uint8, copy=False)
    # for z in range(0, new_img.shape[0]):
    #     for x in range(0, new_img.shape[1]):
    #         if dens_map[z, x] > 30:
    #             cv2.circle(img_file, (x, z), 1, (0, 0, 255))

    new_img = cp.line_img_area(img_file)
    all_dens_map_area = cp.split_img_area(dens_map)
    all_img_area = cp.split_img_area(new_img)

    for index in range(len(all_dens_map_area)):
        area_count = int(round(np.sum(all_dens_map_area[index])))
        area_text = '{}'.format(area_count)
        area_img = cv2.putText(all_img_area[index], area_text, (60, 60),
                               cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

    new_img_path = '{}/{}.jpg'.format(output_path, i)
    cv2.imwrite(new_img_path, new_img)
    print('New image \033[0;32m%s\033[0m was saved. And total count is %s' % (
        new_img_path, count))

    img2video = cv2.resize(new_img, video_size)
    videoWriter.write(img2video)

videoWriter.release()
print("All task is finished.")
