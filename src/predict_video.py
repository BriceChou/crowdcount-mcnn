from crowd_count import CrowdCounter
import network
import numpy as np
import cv2

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
        ht_1 = (ht/16)*16
        wd_1 = (wd/16)*16
        if len(img.shape) == 3:
            img_cp = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)
        
        print(wd_1)
        print(ht_1)
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

    def predict2(self):
        img = cv2.imread("tmp.jpg", 1)
        return self.predict(img)

cp = CrowdPredictor('../final_models/mcnn_shtechB_110.h5')
#cap = cv2.VideoCapture("../20180426.mov")
cap = cv2.VideoCapture(-1)
out = cv2.VideoWriter('../trainingTest.mov',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))
print(cap.isOpened())
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        count, dens_map = cp.predict(frame)
        print(count)
        frame = cv2.putText(frame, "count is " + str(count), (10, 10), cv2.FONT_HERSHEY_PLAIN, 12, (0, 255, 255))
        out.write(frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
# When everything done, release the video capture object
cap.release()
out.release()