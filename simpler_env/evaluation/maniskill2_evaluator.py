"""
Evaluate a model on ManiSkill2 environment.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video


def _base_env(env):
    return getattr(env, "unwrapped", env)


def extract_eef_pos_from_obs(obs, robot_name=None):
    """
    Build eef_pos expected by policy wrappers:
      [x, y, z, qw, qx, qy, qz, gripper_open]  -> shape (8,)

    Current ManiSkill/SimplerEnv obs schema often provides:
      - obs["extra"]["tcp_pose"] -> (7,)
      - obs["agent"]["qpos"]     -> joint positions (robot-specific length)

    We reconstruct an 8D proprio vector by combining tcp_pose + a gripper openness proxy.
    """
    if not isinstance(obs, dict):
        return None

    # 1) TCP pose (xyz + quat[wxyz]) from nested obs
    extra = obs.get("extra", None)
    tcp_pose = None
    if isinstance(extra, dict):
        tcp_pose = extra.get("tcp_pose", None)

    if tcp_pose is None:
        return None

    tcp_pose = np.asarray(tcp_pose, dtype=np.float32)
    if tcp_pose.ndim != 1 or tcp_pose.shape[0] != 7:
        return None

    # 2) Gripper openness proxy from qpos
    gripper_open = None
    agent = obs.get("agent", {})
    if isinstance(agent, dict):
        qpos = agent.get("qpos", None)
        if qpos is not None:
            qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)

            if qpos.size >= 2:
                # Prefer last two joints (commonly finger joints on Panda/google robot)
                finger_vals = qpos[-2:]
                gripper_raw = float(np.mean(np.abs(finger_vals)))
                # Conservative normalization to [0, 1] (tune later if needed)
                gripper_open = float(np.clip(gripper_raw / 0.04, 0.0, 1.0))
            elif qpos.size == 1:
                gripper_raw = float(np.abs(qpos[-1]))
                gripper_open = float(np.clip(gripper_raw / 0.04, 0.0, 1.0))

    # Fallback for first pass
    if gripper_open is None:
        gripper_open = 1.0

    eef_pos = np.concatenate(
        [tcp_pose, np.array([gripper_open], dtype=np.float32)], axis=0
    ).astype(np.float32)

    return eef_pos


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # Put raytracing keys first for compatibility with existing result naming/metrics
        additional_env_build_kwargs = ray_tracing_dict

    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # Initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }

    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }

    obs, _ = env.reset(options=env_reset_options)
    base = _base_env(env)

    # For long-horizon environments, check whether current subtask is final
    is_final_subtask = base.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        task_description = base.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Step the environment
    task_descriptions = []
    info = {}

    while not (predicted_terminated or truncated):
        # Build eef_pos from current obs schema (tcp_pose + gripper proxy)
        eef_pos = extract_eef_pos_from_obs(obs, robot_name=robot_name)

        # Lightweight debug (remove later if noisy)
        tcp_pose_dbg = None
        qpos_dbg = None
        if isinstance(obs, dict):
            extra_dbg = obs.get("extra", {})
            agent_dbg = obs.get("agent", {})
            if isinstance(extra_dbg, dict):
                tcp_pose_dbg = extra_dbg.get("tcp_pose", None)
            if isinstance(agent_dbg, dict):
                qpos_dbg = agent_dbg.get("qpos", None)

        print(
            "DEBUG tcp_pose shape:",
            None if tcp_pose_dbg is None else np.asarray(tcp_pose_dbg).shape,
            "| qpos shape:",
            None if qpos_dbg is None else np.asarray(qpos_dbg).shape,
            "| eef_pos:",
            None if eef_pos is None else eef_pos.shape,
        )

        if eef_pos is None:
            raise RuntimeError(
                "Failed to reconstruct eef_pos from observation. "
                "Expected obs['extra']['tcp_pose'] and obs['agent']['qpos']."
            )

        # Step the model
        # Pass raw camera obs through kwargs so policy wrapper can map real camera names if supported.
        camera_obs = obs.get("image", None) if isinstance(obs, dict) else None
        raw_action, action = model.step(
            image,  # legacy arg kept for compatibility/visualization path
            task_description,
            eef_pos=eef_pos,
            camera_obs=camera_obs,
        )
        predicted_actions.append(raw_action)

        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated and not is_final_subtask:
            # Advance environment to the next subtask
            predicted_terminated = False
            base.advance_to_next_subtask()

        # Step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            ),
        )

        success = "success" if done else "failure"

        new_task_description = base.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)

        is_final_subtask = base.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=obs_camera_name
        )
        images.append(image)
        task_descriptions.append(task_description)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # Save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"

    ckpt_path_basename = ckpt_path[:-1] if ckpt_path.endswith("/") else ckpt_path
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]

    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    else:  # episode
        video_name = f"{success}_obj_episode_{obj_episode_id}"

    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"

    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"

    r, p, y = quat2euler(robot_init_quat)
    video_path = (
        f"{scene_name}/{control_mode}/{env_save_name}/"
        f"rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_"
        f"rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    )
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # Save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.join(os.path.dirname(action_path), "actions")
    os.makedirs(action_root, exist_ok=True)
    action_path = os.path.join(action_root, os.path.basename(action_path))
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # Run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )

                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                        args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id,
                                **kwargs,
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr