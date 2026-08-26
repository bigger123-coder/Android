import cv2

from vision.VisionPipeline import VisionPipeline
# Initialize the camera
cam = cv2.VideoCapture(0)

vision = VisionPipeline()

while True:
    ret, frame = cam.read()

    if not ret:
        break

    frame = vision.process(frame, user_name=None)

    cv2.imshow("AI vision", frame)

    if cv2.waitKey(1) == ord("q"):
        break

vision.close()
cv2.destroyAllWindows()