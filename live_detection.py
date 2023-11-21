# %%
import cv2
from time import sleep
import pickle as pk
import mediapipe as mp
import pandas as pd
import pyttsx4

# import warnings
# warnings.filterwarnings("ignore")

# import TTS

# %%
from recommendations import check_pose_angle
from landmarks import extract_landmarks
from calc_angles import rangles

# %%
# say USE TTS FFS (recommendations)
#   say this every 8 secs or something like that
#   no pose detected
#   say video not clear
# consistenly show the output but only take one pic every second or so
# use resized thingy and pass it to extract_landmarks as well
# show the pose name right above the bounding box
# check names


# %%
def init_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, 360)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cam.set(cv2.CAP_PROP_SHARPNESS, 125)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam


def destory_cam(cam):
    cv2.destroyAllWindows()
    cam.release()


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def load_model():
    return pk.load(open("./models/11_poses.model", "rb"))


def get_pose_name(index):
    names = {
        0: "Ardha Chandrasana",
        1: "Downdog",
        2: "Goddess",
        3: "Marjaryasana",
        4: "Padmasana",
        5: "Plank",
        6: "Sivasana",
        7: "Tree",
        8: "Urdhvamukha shvanasana",
        9: "Utkatasana",
        10: "Warrior"
    }
    return str(names[index])


def init_dicts():
    landmarks_points = {
        "nose": 0,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_heel": 29, "right_heel": 30,
        "left_foot_index": 31, "right_foot_index": 32,
    }

    landmarks_points_array = {
        "left_shoulder": [], "right_shoulder": [],
        "left_elbow": [], "right_elbow": [],
        "left_wrist": [], "right_wrist": [],
        "left_hip": [], "right_hip": [],
        "left_knee": [], "right_knee": [],
        "left_ankle": [], "right_ankle": [],
        "left_heel": [], "right_heel": [],
        "left_foot_index": [], "right_foot_index": [],
    }

    col_names = []
    for i in range(len(landmarks_points.keys())):
        name = list(landmarks_points.keys())[i]
        col_names.append(name + "_x")
        col_names.append(name + "_y")
        col_names.append(name + "_z")
        col_names.append(name + "_v")

    cols = col_names.copy()

    return mp.solutions.pose, cols, landmarks_points_array


cam = init_cam()
model = load_model()
mp_pose, cols, landmarks_points_array = init_dicts()
angles_df = pd.read_csv("./csv_files/11_poses_angles.csv")
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx4.init()

while True:
    result, image = cam.read()
    resized = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
    key = cv2.waitKey(1)
    if key == ord("q"):
        destory_cam(cam=cam)
        break
    if result:
        var = variance_of_laplacian(resized)
        if var > 30.0:
            err, df, xy, landmarks, mp_pose = extract_landmarks(
                resized, mp_pose, cols)
            if err == False:
                prediction = model.predict(df)
                probabilities = model.predict_proba(df)
                cv2.rectangle(image, (xy[0], xy[1]),
                              (xy[2], xy[3]), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(
                    image, landmarks, mp_pose.POSE_CONNECTIONS)
                if probabilities.max() > 0.6:
                    cv2.putText(image, get_pose_name(
                        prediction[0]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)
                    angles = rangles(df, landmarks_points_array)
                    check_pose_angle(prediction[0], angles, angles_df)
                else:
                    cv2.putText(image, "No Pose Detected",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv2.LINE_AA)
                    pass

    cv2.imshow("Something", image)
