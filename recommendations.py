def init():
    angles_dict = {
        "armpit_left": 0,
        "armpit_right": 1,
        "elbow_left": 2,
        "elbow_right": 3,
        "hip_left": 4,
        "hip_right": 5,
        "knee_left": 6,
        "knee_right": 7,
        "ankle_left": 8,
        "ankle_right": 9,
    }
    return angles_dict


def error_margin(control, value):
    if int(value) in range(control - 20, control + 21):
        return True
    return False


def check_joint(angles, joint_name, threshold, body_position):
    angles_dict = init()
    joint_index = angles_dict[joint_name]

    if error_margin(threshold, angles[joint_index]):
        return None

    if angles[joint_index] > threshold:
        return f"Bring {' '.join(joint_name.split('_')[::-1])} closer to {body_position}."
    elif angles[joint_index] < threshold:
        return f"Put {' '.join(joint_name.split('_')[::-1])} further away from {body_position}."

    return None


def check_pose_angle(pose_index, angles, df):
    result = []

    result.append(check_joint(
        angles,
        "armpit_right",
        int(df.loc[pose_index, "armpit_left"]),
        "body"
    ))
    result.append(check_joint(
        angles,
        "armpit_left",
        int(df.loc[pose_index, "armpit_right"]),
        "body"
    ))
    result.append(check_joint(
        angles,
        "elbow_right",
        int(df.loc[pose_index, "elbow_left"]),
        "arm"
    ))
    result.append(check_joint(
        angles,
        "elbow_left",
        int(df.loc[pose_index, "elbow_right"]),
        "arm"
    ))
    result.append(check_joint(
        angles,
        "hip_right",
        int(df.loc[pose_index, "hip_left"]),
        "pelvis"
    ))
    result.append(check_joint(
        angles,
        "hip_left",
        int(df.loc[pose_index, "hip_right"]),
        "pelvis"
    ))
    result.append(check_joint(
        angles,
        "knee_right",
        int(df.loc[pose_index, "knee_left"]),
        "calf"
    ))
    result.append(check_joint(
        angles,
        "knee_left",
        int(df.loc[pose_index, "knee_right"]),
        "calf"
    ))
    result.append(check_joint(
        angles,
        "ankle_right",
        int(df.loc[pose_index, "ankle_left"]),
        "foot"
    ))
    result.append(check_joint(
        angles,
        "ankle_left",
        int(df.loc[pose_index, "ankle_right"]),
        "foot"
    ))

    return [message for message in result if message is not None]
