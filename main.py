# from ultralytics import YOLO

# # Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)

# # Evaluate the model's performance on the validation set
# results = model.val()

# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')



from ultralytics import YOLO



# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8x.pt')
# model.train(data='coco128.yaml', epochs=3) # --- Pre trained model ---
# results = model(source = "20240318_092055.mp4", show=True, conf=0.4, save=True, classes = [1, 2, 3, 4, 5, 6, 7, 8])
# results = model.track(source = "20240318_092055.mp4", show=True, conf=0.4, save=True, classes = [1, 2, 3, 4, 5, 6, 7, 8]) # --- tracking 

results = model.track(source = "20240318_092055.mp4", show=True, conf=0.4, save=True, classes = [1, 2, 3, 4, 5, 6, 7, 8]) # --- tracking 
