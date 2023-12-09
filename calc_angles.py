import numpy as np


def angle(p1, p2, p3):
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


def rangles(pose_df, landmarks_points):
    for index, row in pose_df.iterrows():
        landmarks_points["left_shoulder"] = [
            row["left_shoulder_x"], row["left_shoulder_y"]]

        landmarks_points["right_shoulder"] = [
            row["right_shoulder_x"], row["right_shoulder_y"]]

        landmarks_points["left_elbow"] = [
            row["left_elbow_x"], row["left_elbow_y"]]

        landmarks_points["right_elbow"] = [
            row["right_elbow_x"], row["right_elbow_y"]]

        landmarks_points["left_wrist"] = [
            row["left_wrist_x"], row["left_wrist_y"]]

        landmarks_points["right_wrist"] = [
            row["right_wrist_x"], row["right_wrist_y"]]

        landmarks_points["left_hip"] = [
            row["left_hip_x"], row["left_hip_y"]]

        landmarks_points["right_hip"] = [
            row["right_hip_x"], row["right_hip_y"]]

        landmarks_points["left_knee"] = [
            row["left_knee_x"], row["left_knee_y"]]

        landmarks_points["right_knee"] = [
            row["right_knee_x"], row["right_knee_y"]]

        landmarks_points["left_ankle"] = [
            row["left_ankle_x"], row["left_ankle_y"]]

        landmarks_points["right_ankle"] = [
            row["right_ankle_x"], row["right_ankle_y"]]

        landmarks_points["left_heel"] = [
            row["left_heel_x"], row["left_heel_y"]]

        landmarks_points["right_heel"] = [
            row["right_heel_x"], row["right_heel_y"]]

        landmarks_points["left_foot_index"] = [
            row["left_foot_index_x"], row["left_foot_index_y"]]

        landmarks_points["right_foot_index"] = [
            row["right_foot_index_x"], row["right_foot_index_y"]]

        armpit_left = angle(
            landmarks_points["left_elbow"],
            landmarks_points["left_shoulder"],
            landmarks_points["left_hip"]
        )
        armpit_right = angle(
            landmarks_points["right_elbow"],
            landmarks_points["right_shoulder"],
            landmarks_points["right_hip"]
        )

        elbow_left = angle(
            landmarks_points["left_shoulder"],
            landmarks_points["left_elbow"],
            landmarks_points["left_wrist"]
        )
        elbow_right = angle(
            landmarks_points["right_shoulder"],
            landmarks_points["right_elbow"],
            landmarks_points["right_wrist"]
        )

        hip_left = angle(
            landmarks_points["right_hip"],
            landmarks_points["left_hip"],
            landmarks_points["left_knee"]
        )
        hip_right = angle(
            landmarks_points["left_hip"],
            landmarks_points["right_hip"],
            landmarks_points["right_knee"]
        )

        knee_left = angle(
            landmarks_points["left_hip"],
            landmarks_points["left_knee"],
            landmarks_points["left_ankle"]
        )
        knee_right = angle(
            landmarks_points["right_hip"],
            landmarks_points["right_knee"],
            landmarks_points["right_ankle"]
        )

        ankle_left = angle(
            landmarks_points["left_knee"],
            landmarks_points["left_ankle"],
            landmarks_points["left_foot_index"]
        )
        ankle_right = angle(
            landmarks_points["right_knee"],
            landmarks_points["right_ankle"],
            landmarks_points["right_foot_index"]
        )

        angles = [
            armpit_left,
            armpit_right,
            elbow_left,
            elbow_right,
            hip_left,
            hip_right,
            knee_left,
            knee_right,
            ankle_left,
            ankle_right
        ]
        return angles
