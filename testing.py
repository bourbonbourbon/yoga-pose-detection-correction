#
# import cv2
# # from time import sleep
# # from sys import exit

#
# def init_cam():
#     cam = cv2.VideoCapture(0)
#     print("Starting camera...")
#     cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#     cam.set(cv2.CAP_PROP_FOCUS, 360)  # fix this
#     cam.set(cv2.CAP_PROP_BRIGHTNESS, 130)
#     cam.set(cv2.CAP_PROP_SHARPNESS, 125)
#     cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#     face_cascade = cv2.CascadeClassifier(
#         "./haarcascade/haarcascade_frontalface_default.xml")
#     return cam, face_cascade


# def destory_cam(cam):
#     cam.release()
#     cv2.destroyAllWindows()


# def variance_of_laplacian(image):
#     return cv2.Laplacian(image, cv2.CV_64F).var()


#
# cam, face_cascade = init_cam()

# while True:
#     i = 360
#     result, image = cam.read()
#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     face = face_cascade.detectMultiScale(
#         gray_img,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(40, 40)
#     )
#     key = cv2.waitKey(1)
#     if key == ord("q"):
#         destory_cam(cam=cam)
#         break
#     if key == ord("h"):
#         i += 5
#         print(i)
#         cam.set(cv2.CAP_PROP_FOCUS, i)
#     if result:
#         var = variance_of_laplacian(image)
#         if var > 30 and len(face) != 0:
#             print("In focus")
#     else:
#         print("Not in focus")
#     cv2.imshow("Something", image)


# import pyttsx4

# engine = pyttsx4.init()
# engine.say("Hello how are you doing today?")
# engine.runAndWait()
# del engine

from time import time, sleep

start = time()
sleep(1)
end = time()
print(end - start)
