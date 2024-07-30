# Experiment-with-Yolov8

# Vehicle Detection and Speed Calculation

## Task 1: Vehicle Detection

- Used a pre-trained model `yolov8x.pt`.
- It is trained on the COCO dataset, which includes 80 classes.
- Our focus is on vehicles, so only vehicle-related classes are considered.
- For this dataset, vehicle class IDs are: 1, 2, 3, 4, 5, 6, 7, 8.

## Task 2: Moving Vehicle Detection

- In the case of a moving vehicle we have to calculate the distance from a static thing.
I have drawn a line and for each frame I have calculated how far the vehicle is from the
line.
- Our video sequence is 2400 in width and 1080 in height. So I have taken a line in the
middle.
- If for each frame the distance between the vehicle and the x-axis of the line is almost the
same we have not considered that vehicle.
- As each frame is 2400 and I assume that the total road length in the frame is 35 meters,
I have made a threshold.
- Within this threshold every vehicle is static.
- As the video sequence is a bit shaky it has detected a static vehicle a moving one while it
was a bit more shaky at the end.
- Moving vehicle detection is the building block for moving vehicle speed calculation.
- I have done both of the part in the same code.

## Task 3: Speed Calculation for moving vehicle

Here I have considered that vehicles are only moving from left to right. In the case of right
to left the concept is very similar.
- I have considered that the total length of the road is 35 meters(as there are four parking
spaces and also more space in the front and the end and each car is more or less five
meters).
- I have considered two lines. line1 (the left line is called line1 and the right line is called
line 2).
- when a line crosses the left one and then enters the right one that means the car/vehicle
is moving.
- From that point I calculate the speed using the concept of a road length of 35 meters and
a frame width of 2400.
- Before calculating the speed I make sure that the car is not static from the concept of
Task 2


