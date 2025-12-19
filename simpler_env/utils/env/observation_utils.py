def _base_env(env):
    return getattr(env, "unwrapped", env)

def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    base = _base_env(env)

    if camera_name is None:
        if "google_robot" in base.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in base.robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError(f"Unknown robot_uid={getattr(base,'robot_uid',None)}")

    return obs["image"][camera_name]["rgb"]
