from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2
import time
import torch




model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture("20240318_092055.mp4")
# cap = cv2.VideoCapture("Test_Video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Video writer
video_writer = cv2.VideoWriter("speed_estimation_v3.avi",
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps,
                               (w, h))

# line_pts = [(0, 360), (1280, 360)]
# line_pts = [(20, 400), (1260, 400)]
# line_pts = [(1200, 0), (1200, 1080)]

line_pts = [(0, 540), (2400, 540)]

lin1 = [(1100, 0), (1100, 1080)]
lin2 = [(1400, 0), (1400, 1080)]

moving_id_time = {}
moving_id_pos = {}
moving_id_cls = {}
detect_static_id = {}
totlen = 35 # meters 
class_names = {}
class_names[1.0] = "Bicycle"
class_names[2.0] = "Car"
class_names[3.0] = "Motorcycle"
class_names[4.0] = "Airplane"
class_names[5.0] = "Bus"
class_names[6.0] = "Train"
class_names[7.0] = "Truck"
class_names[8.0] = "Boat"


# Init speed-estimation obj
speed_obj = speed_estimation.SpeedEstimator()
speed_obj.set_args(reg_pts=line_pts,
                   names=names,
                   view_img=True)



while cap.isOpened():

    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    tracks = model.track(im0, persist=True, show=False, classes = [1, 2, 3, 4, 5, 6, 7, 8], tracker="bytetrack.yaml", conf=0.25, iou=0.5)

    print("----FRAME-------\n")
    if tracks[0].boxes.id is not None:
        print(tracks[0].boxes.xyxy.cpu())
        print(tracks[0].boxes.cls.cpu().tolist())
        print(tracks[0].boxes.id.int().cpu().tolist())
        boxes_xyxy = tracks[0].boxes.xyxy.cpu()
        box_ids = tracks[0].boxes.id.int().cpu().tolist()
        obj_cls = tracks[0].boxes.cls.cpu().tolist()
        for i, box in enumerate(boxes_xyxy):
            flag = True # moving 
            x1, y1, x2, y2 = box
            x1 = x1.item()
            y1 = y1.item()
            x2 = x2.item()
            y2 = y2.item()

            cx = (x1+x2)/2
            cy = (y1+y2)/2

            mov_dis = int(abs(cx-lin1[0][0]))
            if box_ids[i] not in detect_static_id:
                detect_static_id[box_ids[i]] = mov_dis

            else:
                static_dis = detect_static_id[box_ids[i]]
                if abs(mov_dis-static_dis)<40:
                    flag = False


            if cx>lin1[0][0] and box_ids[i] not in moving_id_time:
                moving_id_time[box_ids[i]] = time.time()
                moving_id_pos[box_ids[i]] = (cx,cy)
                moving_id_cls[box_ids[i]] = obj_cls[i]

            if cx>lin2[0][0] and box_ids[i] in moving_id_time:
                elapsed_time = time.time() - moving_id_time[box_ids[i]]
                dis1 = cx - lin1[0][0]
                dis = (35/2400)*dis1
                a_speed_ms = dis / elapsed_time
                a_speed_kh = a_speed_ms * 3.6

                class_name =  class_names[moving_id_cls[box_ids[i]]]
                if flag:
                    cv2.rectangle(im0,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)
                    cv2.putText(im0,str(box_ids[i]),(int(x1),int(y1)),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
                    cv2.putText(im0,str(int(a_speed_kh))+'Km/h',(int(x2),int(y2)),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

    cv2.line(im0, lin1[0], lin1[1], (0, 255, 0), 2)
    cv2.line(im0, lin2[0], lin2[1], (0, 255, 0), 2)
    video_writer.write(im0)

                



            

    
    # speed detection
    #  im0 = speed_obj.estimate_speed(im0, tracks)
    print("------------FRAME---------\n")

    


    

cap.release()
video_writer.release()
cv2.destroyAllWindows()