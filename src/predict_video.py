from crowd_count import CrowdCounter
import network
import numpy as np
import cv2
import os
import time


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


# Config the model weights path
final_models_path = os.path.abspath('../saved_models')

# Cofig the output video folder
video_output_path = os.path.abspath('../output_video')

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

# count the frame
count_frame = 0
cap = cv2.VideoCapture('../20180406.mov')
rval = cap.isOpened()
# frame_fps = 30
while rval:
    count_frame += 1
    rval, frame = cap.read()
    # if(count_frame%frame_fps == 0):
    #     cv2.imwrite('{}/{}.jpg'.format(output_path, count_frame), frame)
    if rval:
        # save current frame to image
        # img_path = '{}/{}.jpg'.format(output_path, count_frame)
        # cv2.imwrite(img_path, frame)
        # print('output image \033[0;32m%s\033[0m was saved. ' % (img_path))

        count, dens_map = cp.predict(frame)
        new_img = cp.line_img_area(frame)
        all_dens_map_area = cp.split_img_area(dens_map)
        all_img_area = cp.split_img_area(new_img)

        for index in range(len(all_dens_map_area)):
            area_count = int(round(np.sum(all_dens_map_area[index])))
            area_text = '{}'.format(area_count)
            area_img = cv2.putText(all_img_area[index], area_text, (60, 60),
                                   cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

        # To quickly finsh the task, we could not save the new image.
        # new_img_path = '{}/{}.jpg'.format(output_path, count_frame)
        # cv2.imwrite(new_img_path, new_img)
        # print('New image \033[0;32m%s\033[0m was saved. And total count is %s' % (
        #     new_img_path, count))

        img2video = cv2.resize(new_img, video_size)
        videoWriter.write(img2video)
        print('Video Frame[\033[0;32m%s\033[0m] was saved. And total count is %s' % (
            count_frame, count))
    else:
        break

# When everything done, release the video capture object
cap.release()
videoWriter.release()
print("All task is finished.")
