import cv2
from time import time
import pickle as pk
import mediapipe as mp
import pandas as pd
import pyttsx4
import multiprocessing as mtp

from recommendations import check_pose_angle
from landmarks import extract_landmarks
from calc_angles import rangles


def init_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cam.set(cv2.CAP_PROP_FOCUS, 360)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 130)
    cam.set(cv2.CAP_PROP_SHARPNESS, 125)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cam


def get_pose_name(index):
    names = {
        0: "Adho Mukha Svanasana",
        1: "Phalakasana",
        2: "Utkata Konasana",
        3: "Vrikshasana",
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
    return cols, landmarks_points_array


engine = pyttsx4.init()


def tts(tts_q):
    while True:
        objects = tts_q.get()
        if objects is None:
            break
        message = objects[0]
        engine.say(message)
        engine.runAndWait()
    tts_q.task_done()


def cv2_put_text(image, message):
    cv2.putText(
        image,
        message,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 0, 0),
        5,
        cv2.LINE_AA
    )


def destory(cam, tts_proc, tts_q):
    cv2.destroyAllWindows()
    cam.release()
    tts_q.put(None)
    tts_q.close()
    tts_q.join_thread()
    tts_proc.join()


if __name__ == "__main__":
    cam = init_cam()
    model = pk.load(open("./models/4_poses.model", "rb"))
    cols, landmarks_points_array = init_dicts()
    angles_df = pd.read_csv("./csv_files/4_angles_poses_angles.csv")
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    tts_q = mtp.JoinableQueue()

    tts_proc = mtp.Process(target=tts, args=(tts_q, ))
    tts_proc.start()

    tts_last_exec = time() + 5

    while True:
        result, image = cam.read()
        flipped = cv2.flip(image, 1)
        resized_image = cv2.resize(
            flipped,
            (640, 360),
            interpolation=cv2.INTER_AREA
        )

        key = cv2.waitKey(1)
        if key == ord("q"):
            destory(cam, tts_proc, tts_q)
            break

        if result:
            err, df, landmarks = extract_landmarks(
                resized_image,
                mp_pose,
                cols
            )

            if err == False:
                prediction = model.predict(df)
                probabilities = model.predict_proba(df)

                mp_drawing.draw_landmarks(
                    flipped,
                    landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                if probabilities[0, prediction[0]] > 0.85:
                    cv2_put_text(
                        flipped,
                        get_pose_name(prediction[0])
                    )

                    angles = rangles(df, landmarks_points_array)
                    suggestions = check_pose_angle(
                        prediction[0], angles, angles_df)

                    if time() > tts_last_exec:
                        tts_q.put([
                            suggestions[0]
                        ])
                        tts_last_exec = time() + 5

                else:
                    cv2_put_text(
                        flipped,
                        "No Pose Detected"
                    )
            cv2.imshow("Frame", flipped)
