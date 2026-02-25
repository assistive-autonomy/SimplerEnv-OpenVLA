from typing import Optional, Sequence, List
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from collections import deque
from PIL import Image
import torch
import cv2 as cv
from simpler_env.utils.action.action_ensemble import ActionEnsembler
from .geometry import quat2mat, mat2euler
import numpy as np
import torch

def auto_model_fn(path):
    """
    Return the correct LeRobot policy class for pi0 / pi0-fast checkpoints.
    Prefer installed LeRobot package classes, fallback to local modeling files.
    """
    import os
    import json
    from pathlib import Path

    # Try to inspect config.json if path is local
    cfg = {}
    cfg_path = Path(path) / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}

    # Heuristics for pi0-fast vs pi0
    cfg_text = str(cfg).lower()
    is_pi0_fast = (
        "pi0fast" in str(path).lower()
        or "pi0_fast" in str(path).lower()
        or "pi0fast" in cfg_text
        or "pi0_fast" in cfg_text
        or "pi0fastconfig" in cfg_text
    )

    # 1) Installed LeRobot package (preferred)
    try:
        if is_pi0_fast:
            from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
            return PI0FastPolicy
        else:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
            return PI0Policy
    except Exception:
        pass

    # 2) Fallback: local checkpoint folder contains modeling file(s)
    import sys
    if os.path.isdir(path):
        sys.path.append(path)
        if is_pi0_fast:
            try:
                from modeling_pi0_fast import PI0FastPolicy  # type: ignore
                return PI0FastPolicy
            except Exception:
                pass
        from modeling_pi0 import PI0Policy  # type: ignore
        return PI0Policy

    raise ImportError(f"Could not resolve PI0/Pi0Fast policy class for checkpoint: {path}")

class LerobotPiFastInference:
    def __init__(
        self,
        saved_model_path: str = "pretrained/pi0",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        exec_horizon: int = 4,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        action_ensemble_temp: float = -0.8,
    ) -> None:
        gpu_idx = os.environ.get("GPU_IDX", 0)
        self.device = f"cuda:{gpu_idx}"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig/1.0.0" if unnorm_key is None else unnorm_key
            action_ensemble = True
            self.sticky_gripper_num_repeat = 1
            # EE pose in Bridge data was relative to a top-down pose, instead of robot base
            self.default_rot = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])  # https://github.com/rail-berkeley/bridge_data_robot/blob/b841131ecd512bafb303075bd8f8b677e0bf9f1f/widowx_envs/widowx_controller/src/widowx_controller/widowx_controller.py#L203
        elif policy_setup == "google_robot":
            unnorm_key = (
                "fractal20220817_data/0.1.0" if unnorm_key is None else unnorm_key
            )
            action_ensemble = True
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")

        # TODO: add pi0 loading ...
        PolicyCls = auto_model_fn(saved_model_path)
        self.vla = PolicyCls.from_pretrained(saved_model_path, map_location=self.device)
        self.vla.reset()

        self.image_size = image_size
        self.action_scale = action_scale
        self.obs_horizon = 1
        self.obs_interval = 1
        self.pred_action_horizon = 5
        self.image_history = deque(maxlen=self.obs_horizon)
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.task = None
        self.task_description = None

    def reset(self, task_description: str) -> None:
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_plan = deque()

    def preprocess_widowx_proprio(self, eef_pos) -> np.array:
        """convert ee rotation to the frame of top-down
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L167
        """
        # StateEncoding.POS_EULER: xyz + rpy + pad + gripper(openness)
        proprio = eef_pos
        rm_bridge = quat2mat(proprio[3:7])
        rpy_bridge_converted = mat2euler(rm_bridge @ self.default_rot.T)
        gripper_openness = proprio[7] # from simpler, 0 for close, 1 for open
        raw_proprio = np.concatenate(
            [
                proprio[:3],
                rpy_bridge_converted,
                np.zeros(1),
                [gripper_openness],
            ]
        )
        return raw_proprio

    def preprocess_google_robot_proprio(self, eef_pos) -> np.array:
        """convert wxyz quat from simpler to xyzw used in fractal
        https://github.com/allenzren/open-pi-zero/blob/c3df7fb062175c16f69d7ca4ce042958ea238fb7/src/agent/env_adapter/simpler.py#L204
        """
        # StateEncoding.POS_QUAT: xyz + q_xyzw + gripper(closeness)
        quat_xyzw = np.roll(eef_pos[3:7], -1)
        gripper_width = eef_pos[
            7
        ]  # from simpler, 0 for close, 1 for open
        # need invert as the training data comes from closeness
        gripper_closedness = (
            1 - gripper_width
        )  # TODO(allenzren): change fractal data processing in training so also use gripper openness in proprio (as in bridge) instead of closedness
        raw_proprio = np.concatenate(
            (
                eef_pos[:3],
                quat_xyzw,
                [gripper_closedness],
            )
        )
        return raw_proprio

    def _extract_rgb_from_camera_entry(self, cam_entry):
        """
        ManiSkill obs['image'][camera_name] is typically a dict containing rgb/rgbd tensors.
        Return HxWx3 uint8 RGB numpy array.
        """
        if cam_entry is None:
            return None

        if isinstance(cam_entry, dict):
            # Common keys across ManiSkill variants
            for k in ["rgb", "Color", "color"]:
                if k in cam_entry:
                    rgb = cam_entry[k]
                    break
            else:
                rgb = None
        else:
            rgb = cam_entry

        if rgb is None:
            return None

        rgb = np.asarray(rgb)

        # If batched or weird shape, squeeze singleton dims
        rgb = np.squeeze(rgb)

        # Convert float [0,1] to uint8 if needed
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 1) if np.issubdtype(rgb.dtype, np.floating) else rgb
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = (rgb * 255.0).astype(np.uint8)
            else:
                rgb = rgb.astype(np.uint8)

        # If RGBA, drop alpha
        if rgb.ndim == 3 and rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError(f"Unexpected RGB shape from camera entry: {rgb.shape}")

        return rgb


    def _checkpoint_image_feature_keys(self):
        """
        Return image feature keys expected by the loaded LeRobot policy checkpoint.
        """
        cfg = getattr(self.vla, "config", None)
        input_features = getattr(cfg, "input_features", None)

        keys = []
        if isinstance(input_features, dict):
            for k, feat in input_features.items():
                feat_type = getattr(feat, "type", None)
                # FeatureType.VISUAL enum string repr can vary; compare robustly
                if feat_type is not None and "VISUAL" in str(feat_type):
                    keys.append(k)

        return keys

    def _build_language_observation(self, task_description: str):
        """
        Build PI0/PI0Fast language tokens in the keys expected by LeRobot:
        - observation.language.tokens
        - observation.language.attention_mask
        """
        import torch
        from transformers import AutoTokenizer

        if task_description is None:
            task_description = ""
        if not isinstance(task_description, str):
            task_description = str(task_description)

        candidates = []

        def _add(label, obj):
            if obj is not None:
                candidates.append((label, obj))

        # direct attrs on wrapper
        _add("self.processor", getattr(self, "processor", None))
        _add("self.tokenizer", getattr(self, "tokenizer", None))

        vla = getattr(self, "vla", None)
        if vla is not None:
            # direct attrs on vla
            for name in ("processor", "tokenizer", "text_tokenizer"):
                _add(f"self.vla.{name}", getattr(vla, name, None))

            # nested attrs on vla modules
            for parent_name in ("model", "paligemma_with_expert", "paligemma"):
                parent = getattr(vla, parent_name, None)
                if parent is None:
                    continue
                for name in ("processor", "tokenizer", "text_tokenizer"):
                    _add(f"self.vla.{parent_name}.{name}", getattr(parent, name, None))

            # inspect action_tokenizer internals (do NOT call action_tokenizer directly on text)
            actp = getattr(vla, "action_tokenizer", None)
            if actp is not None:
                print("DEBUG found self.vla.action_tokenizer:", type(actp))
                for name in ("tokenizer", "text_tokenizer", "base_tokenizer", "llm_tokenizer", "processor"):
                    _add(f"self.vla.action_tokenizer.{name}", getattr(actp, name, None))

                inner_proc = getattr(actp, "processor", None)
                if inner_proc is not None:
                    _add("self.vla.action_tokenizer.processor.tokenizer", getattr(inner_proc, "tokenizer", None))
                    _add("self.vla.action_tokenizer.processor.text_tokenizer", getattr(inner_proc, "text_tokenizer", None))

            # fallback: try loading tokenizer from config text model name
            cfg = getattr(vla, "config", None)
            text_model_candidates = []
            if cfg is not None:
                for attr in ("text_config", "vlm_config_hf", "vlm_config"):
                    sub = getattr(cfg, attr, None)
                    if sub is None:
                        continue
                    for name_attr in ("_name_or_path", "name_or_path", "model_name_or_path"):
                        nm = getattr(sub, name_attr, None)
                        if isinstance(nm, str) and nm.strip():
                            text_model_candidates.append(nm)

                    sub_text = getattr(sub, "text_config", None)
                    if sub_text is not None:
                        for name_attr in ("_name_or_path", "name_or_path", "model_name_or_path"):
                            nm = getattr(sub_text, name_attr, None)
                            if isinstance(nm, str) and nm.strip():
                                text_model_candidates.append(nm)

            text_model_candidates = list(dict.fromkeys(text_model_candidates))
            if text_model_candidates:
                print("DEBUG tokenizer fallback model candidates:", text_model_candidates)
            for model_name in text_model_candidates:
                try:
                    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
                    _add(f"AutoTokenizer.from_pretrained({model_name})", tok)
                    break
                except Exception as e:
                    print(f"DEBUG tokenizer fallback failed for {model_name}: {repr(e)}")

        # dedup
        dedup = []
        seen = set()
        for label, obj in candidates:
            oid = id(obj)
            if oid in seen:
                continue
            seen.add(oid)
            dedup.append((label, obj))
        candidates = dedup

        if not candidates:
            raise RuntimeError(
                "Could not find a text tokenizer/processor for PI0Fast. "
                "Checked self/self.vla and self.vla.action_tokenizer internals."
            )

        def _normalize_output(enc):
            if hasattr(enc, "data") and isinstance(enc.data, dict):
                data = enc.data
            elif isinstance(enc, dict):
                data = enc
            else:
                raise TypeError(f"Unsupported tokenizer output type: {type(enc)}")

            input_ids = data.get("input_ids")
            attention_mask = data.get("attention_mask")

            if input_ids is None:
                raise KeyError(f"No input_ids in tokenizer output. Keys: {list(data.keys())}")

            if not torch.is_tensor(input_ids):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            else:
                input_ids = input_ids.to(dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)

            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            elif not torch.is_tensor(attention_mask):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            else:
                attention_mask = attention_mask.to(dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

            return {
                "observation.language.tokens": input_ids.to(self.device),
                "observation.language.attention_mask": attention_mask.to(self.device),
            }

        last_err = None
        for label, tok in candidates:
            # Skip the action processor object itself (it is not a text tokenizer)
            if label == "self.vla.action_tokenizer":
                continue

            # Try processor/tokenizer call styles
            for call_style in ("positional", "text_kw"):
                try:
                    if call_style == "positional":
                        enc = tok([task_description], return_tensors="pt", padding=True, truncation=True)
                    else:
                        enc = tok(text=[task_description], return_tensors="pt", padding=True, truncation=True)
                    out = _normalize_output(enc)
                    print(f"DEBUG language tokenizer used: {label} ({call_style})")
                    return out
                except Exception as e:
                    last_err = e

            # If tok is processor-like and has .tokenizer, try that
            try:
                inner_tok = getattr(tok, "tokenizer", None)
                if inner_tok is not None:
                    enc = inner_tok([task_description], return_tensors="pt", padding=True, truncation=True)
                    out = _normalize_output(enc)
                    print(f"DEBUG language tokenizer used: {label}.tokenizer")
                    return out
            except Exception as e:
                last_err = e

        raise RuntimeError(
            "Failed to build language tokens for PI0Fast. "
            f"Last error: {repr(last_err)}. "
            f"Tried candidates: {[label for label, _ in candidates]}"
        )

    def _build_visual_observation_from_camera_obs(self, camera_obs):
        """
        Build the visual observation dict expected by the PI0/PI0Fast checkpoint.

        This version supports a fallback/alias mode for envs that do NOT provide
        wrist cameras (e.g. only base_camera + overhead_camera). It will duplicate
        available views into the checkpoint's expected feature keys so the pipeline
        can run for smoke-testing/debugging.

        Expected checkpoint keys (example):
        - observation.images.base_0_rgb
        - observation.images.left_wrist_0_rgb
        - observation.images.right_wrist_0_rgb
        """
        import numpy as np
        import torch

        if camera_obs is None:
            raise ValueError("camera_obs is None")

        if not isinstance(camera_obs, dict):
            raise ValueError(f"camera_obs must be dict-like, got {type(camera_obs)}")

        # --- helpers -------------------------------------------------------------
        def _to_chw_float_tensor(img_np):
            """HWC uint8/float -> BCHW float32 in [0,1] on self.device."""
            arr = np.asarray(img_np)

            # Handle possible batch dimension accidentally present
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0]

            if arr.ndim != 3:
                raise ValueError(f"Expected image with 3 dims (H,W,C), got shape {arr.shape}")

            # Some envs may provide RGBA; drop alpha
            if arr.shape[-1] == 4:
                arr = arr[..., :3]

            if arr.shape[-1] != 3:
                raise ValueError(f"Expected image channel-last RGB/RGBA, got shape {arr.shape}")

            # Ensure uint8-like normalization
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)

            # Resize using existing model helper (expects HWC uint8 usually), so do a safe path:
            # convert back to uint8 for consistency with current preprocessing pipeline
            arr_u8 = (arr * 255.0).astype(np.uint8)
            arr_u8 = self._resize_image(arr_u8)

            ten = torch.from_numpy(arr_u8).permute(2, 0, 1).unsqueeze(0).to(self.device).float() / 255.0
            return ten

        def _extract_rgb(cam_entry):
            """
            Try common ManiSkill/SimplerEnv camera entry layouts.
            cam_entry is often a dict/OrderedDict with keys like rgb, Color, etc.
            """
            if cam_entry is None:
                return None

            # Already an image array?
            if isinstance(cam_entry, np.ndarray):
                return cam_entry

            if isinstance(cam_entry, dict):
                # Common key guesses in priority order
                candidate_keys = [
                    "rgb",
                    "RGB",
                    "color",
                    "Color",
                    "rgb_image",
                    "image",
                ]
                for k in candidate_keys:
                    if k in cam_entry and cam_entry[k] is not None:
                        return cam_entry[k]

                # If nested unexpectedly, try first ndarray value
                for v in cam_entry.values():
                    if isinstance(v, np.ndarray):
                        return v

            return None

        # --- inspect checkpoint-required visual feature keys ---------------------
        # PI0Fast stores feature schema in config; try robust access patterns.
        image_feature_keys = []
        try:
            # Newer LeRobot configs often expose input_features dict
            feats = getattr(self.vla.config, "input_features", None)
            if isinstance(feats, dict):
                for k, v in feats.items():
                    # v may be a dataclass-like object with .type == VISUAL
                    vtype = getattr(v, "type", None)
                    vtype_str = str(vtype)
                    if "VISUAL" in vtype_str and k.startswith("observation.images."):
                        image_feature_keys.append(k)
        except Exception:
            image_feature_keys = []

        # Fallback if schema lookup fails
        if not image_feature_keys:
            image_feature_keys = ["observation.images.image"]

        # --- collect available env camera RGBs -----------------------------------
        available = {}
        for cam_name, cam_entry in camera_obs.items():
            rgb = _extract_rgb(cam_entry)
            if rgb is not None:
                available[cam_name] = rgb

        if len(available) == 0:
            raise ValueError(
                f"No RGB images found in camera_obs. camera_obs keys={list(camera_obs.keys())}"
            )

        # Helpful debug (keep/remove as you prefer)
        print("DEBUG camera_obs keys in step:", list(camera_obs.keys()))
        print("DEBUG extracted RGB camera keys:", list(available.keys()))
        print("DEBUG checkpoint image features:", image_feature_keys)

        # Prefer these env camera names if present
        base_img = available.get("base_camera", None)
        overhead_img = available.get("overhead_camera", None)

        # Fallbacks if names differ
        if base_img is None and len(available) > 0:
            # take first camera as base fallback
            first_key = next(iter(available.keys()))
            base_img = available[first_key]

        if overhead_img is None:
            # choose a different camera than base if possible, else duplicate base
            overhead_img = None
            for k, v in available.items():
                if v is not base_img:
                    overhead_img = v
                    break
            if overhead_img is None:
                overhead_img = base_img

        # --- build observation dict keyed exactly as checkpoint expects ----------
        visual_obs = {}

        for feat_key in image_feature_keys:
            # Direct single-image schemas (older/simple wrappers)
            if feat_key == "observation.images.image":
                visual_obs[feat_key] = _to_chw_float_tensor(base_img)
                continue

            # Multi-camera schemas (PI0Fast / Google-style)
            if "base_0_rgb" in feat_key:
                visual_obs[feat_key] = _to_chw_float_tensor(base_img)
            elif "left_wrist_0_rgb" in feat_key:
                # Smoke-test alias: duplicate base if wrist cam absent
                source = available.get("left_wrist_camera", None)
                if source is None:
                    source = base_img
                visual_obs[feat_key] = _to_chw_float_tensor(source)
            elif "right_wrist_0_rgb" in feat_key:
                # Smoke-test alias: use overhead if present, else duplicate base
                source = available.get("right_wrist_camera", None)
                if source is None:
                    source = overhead_img
                visual_obs[feat_key] = _to_chw_float_tensor(source)
            else:
                # Generic fallback: if an unknown image feature is requested, feed base camera
                visual_obs[feat_key] = _to_chw_float_tensor(base_img)

        return visual_obs

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        camera_obs = kwargs.get("camera_obs", None)
        eef_pos = kwargs.get("eef_pos", None)
        print("DEBUG wrapper eef_pos in step:", eef_pos, type(eef_pos), getattr(eef_pos, "shape", None))
        if isinstance(camera_obs, dict):
            print("DEBUG camera_obs keys in step:", list(camera_obs.keys()))
	
        eef_pos = kwargs.get("eef_pos", None)
        print("DEBUG wrapper eef_pos in step:", eef_pos, type(eef_pos), getattr(eef_pos, "shape", None))
            
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        eef_pos = kwargs.get("eef_pos", None)
        if self.policy_setup == "widowx_bridge":
            state = self.preprocess_widowx_proprio(eef_pos)
            image_key = "observation.images.image_0"
        elif self.policy_setup == "google_robot":
            state = self.preprocess_google_robot_proprio(eef_pos)
            image_key = "observation.images.image"

        # if self.action_ensemble:
        #     raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        if not self.action_plan:
            # state is currently a numpy array from preprocess_*; convert once here
            state_t = torch.from_numpy(state).unsqueeze(0).to(self.device).float()

            # Build real camera features matching checkpoint schema
            visual_obs = self._build_visual_observation_from_camera_obs(camera_obs)

            # Build tokenized language inputs expected by PI0/PI0Fast
            lang_obs = self._build_language_observation(task_description)

            # Final batch for PI0Fast
            observation = {
                "observation.state": state_t,
                **visual_obs,
                **lang_obs,
            }

            # Optional debug
            print("DEBUG observation keys for PI0Fast:", list(observation.keys()))

            action_chunk = self.vla.select_action(observation)[0][:self.pred_action_horizon].cpu().numpy()
            self.action_plan.extend(action_chunk[: self.exec_horizon])

        raw_actions = self.action_plan.popleft()

        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(
                raw_actions[6:7]
            ),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(
            raw_action["rotation_delta"], dtype=np.float64
        )
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0
        
        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        # images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[np.ndarray],
        images: Sequence[np.ndarray],
        save_path: str,
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate(
                    [a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1
                )
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(
                pred_actions[:, action_dim], label="predicted action"
            )
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
