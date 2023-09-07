from ultralytics import YOLO
import cv2, random
import numpy as np

class PeopleDetector():
    def __init__(self) -> None:
        self.h, self.w = 480, 640
        self.video_id = 5
        video_path = './videos/00{}.avi'.format(self.video_id)
        self.dic = {}

        self.cap = cv2.VideoCapture (video_path)
        self.model = YOLO ('yolov8n.pt')

        self._get_matrix_pt()
        self.Video()
        pass
   
    def _get_matrix_pt(self):
        self.pt_A, self.pt_B, self.pt_C, self.pt_D = ([300,150], [25, 480], [550,480], [500,150]) if self.video_id == 2 \
            else ([150,50], [25, 480], [550,480], [200,50]) if self.video_id == 5 \
            else ([230,130],[75, 470],[600,470],[300,130]) if self.video_id == 1 \
            else ([240,270],[60, 480],[600,480],[550,270]) if self.video_id == 3 \
            else ([380,95],[5, 470],[500,480],[450,100])
    
    def _output_img(self):
        im_out = cv2.warpPerspective(self.frame, self.matrix, (self.maxWidth, self.maxHeight),flags=cv2.INTER_LINEAR)        
        im_out  = self.Frame_connect(self.frame,im_out,self.h,self.w)
        return im_out

    def _update_dic(self, obj_id, x1, x2, y2):
        lX = x1 + int((x2-x1)/2)
        if str(obj_id) not in self.dic: 
            self.dic[str(obj_id)] = [[],[]]
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            self.dic[str(obj_id)][1].append(color)
        self.dic[str(obj_id)][0].append((lX, y2))
        color =self.dic[str(obj_id)][1][0]
        return color, lX

    def _create_detection_info(self, obj_id, x1, y1, x2, y2, class_id, score, result):
        
        color, lX = self._update_dic(obj_id, x1, x2, y2)
        text = "ID: %s. %s %s" % (obj_id, result.names[class_id], round(score,2))
        (w_t, h_t), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2) 
        cv2.circle(self.frame, (lX, y2),3, color, -1)
        cv2.rectangle(self.frame, (x1, y1 - 20), (x1 + w_t, y1), color, -1)
        cv2.putText(self.frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        l = len(self.dic[str(obj_id)][0])
        for pt in range(l):
            if not pt + 1 == l: self.frame = cv2.line(self.frame, self.dic[str(obj_id)][0][pt], self.dic[str(obj_id)][0][pt+1], color, 2)
    
    def MaxWH(self):
        width_AD = np.sqrt(((self.pt_A[0] - self.pt_D[0]) ** 2) + ((self.pt_A[1] - self.pt_D[1]) ** 2))
        width_BC = np.sqrt(((self.pt_B[0] - self.pt_C[0]) ** 2) + ((self.pt_B[1] - self.pt_C[1]) ** 2))
        height_AB = np.sqrt(((self.pt_A[0] - self.pt_B[0]) ** 2) + ((self.pt_A[1] - self.pt_B[1]) ** 2))
        height_CD = np.sqrt(((self.pt_C[0] - self.pt_D[0]) ** 2) + ((self.pt_C[1] - self.pt_D[1]) ** 2))
        
        maxWidth = max(int(width_AD), int(width_BC))
        maxHeight = max(int(height_AB), int(height_CD))
        return maxWidth, maxHeight

    def create_HMatrix(self):
        pt1 = np.float32([self.pt_A, self.pt_B, self.pt_C, self.pt_D]) 
        self.maxWidth, self.maxHeight = self.MaxWH()
        pt2 = np.float32([[0,0],[0,self.maxHeight-1],[self.maxWidth-1,self.maxHeight-1],[self.maxWidth,0]])
        self.matrix = cv2.getPerspectiveTransform(pt1, pt2)

    def Frame_connect(self, frame1, frame2, h, w ):
        frame2 = cv2.resize(frame2, (int(w), int(h)), interpolation = cv2.INTER_AREA)
        BG = cv2.resize(frame1, (int(w * 2), int(h)), interpolation = cv2.INTER_AREA)
        BG[0:int(h),0:int(w)] = frame1
        BG[0:int(h),int(w):int(w*2)] = frame2
        return (BG)
           
    def detect(self, results):
        for result in results:
            detections = result.boxes.data.tolist()
            for r in detections:
                x1, y1, x2, y2, obj_id, score, class_id = r
                if int(class_id) == 0: self._create_detection_info(int(obj_id),int(x1), int(y1), int(x2), int(y2), int(class_id), score, result)

    def Video(self):
        self.create_HMatrix()
        while True:
            s, self.frame = self.cap.read()
            if not s: break
            results = self.model.track(source=self.frame, iou = 0.5, tracker="bytetrack.yaml", persist=True)
            try:
                self.detect(results)
                im_out = self._output_img()
                cv2.imshow ('frame', im_out)
            except:
                pass
            key = cv2.waitKey(1)
            if key == ord('q'): break
        self.cap.release ()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    PeopleDetector()














# import scipy.io
# h_matrix = scipy.io.loadmat('./homography/Homography.mat')['Homography'][0][4]
# print (h_matrix)
