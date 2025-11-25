import os
import traceback
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import ffmpeg
import torch

from configs.config import Config
from infer.modules.uvr5.mdxnet import MDXNetDereverb
from infer.modules.uvr5.vr import AudioPre, AudioPreDeEcho

config = Config()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15, config.device)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(
                    os.getenv("weight_uvr5_root"), model_name + ".pth"
                ),
                device=config.device,
                is_half=config.is_half,
            )
        is_hp3 = "HP3" in model_name
        normalized_paths = []

        def _add_path(path_like):
            if path_like is None:
                return
            if isinstance(path_like, dict):
                path_candidate = path_like.get("name") or path_like.get("path")
            else:
                path_candidate = getattr(path_like, "name", None) or path_like
            if not path_candidate:
                return
            cleaned = str(path_candidate).strip()
            if not cleaned:
                return
            normalized_paths.append(cleaned)

        if inp_root:
            for root_line in inp_root.replace("\r", "\n").split("\n"):
                cleaned_root = root_line.strip().strip('"').strip("'")
                if not cleaned_root:
                    continue
                root_path = Path(cleaned_root).expanduser()
                if root_path.is_dir():
                    for entry in sorted(root_path.iterdir()):
                        if entry.is_file():
                            normalized_paths.append(str(entry))
                elif root_path.is_file():
                    normalized_paths.append(str(root_path))
        if paths:
            for path in paths:
                _add_path(path)
        # Deduplicate while preserving order
        seen = set()
        deduped_paths = []
        for path in normalized_paths:
            if path in seen:
                continue
            seen.add(path)
            deduped_paths.append(path)

        paths = []
        for path in deduped_paths:
            resolved = Path(path).expanduser()
            if not resolved.exists():
                logger.warning("UVR batch skipping missing input: %s", path)
                infos.append("%s->Missing input" % Path(path).name)
                continue
            if resolved.is_dir():
                logger.warning("UVR batch skipping directory input: %s", path)
                infos.append("%s->Input is a directory" % Path(path).name)
                continue
            paths.append(str(resolved))

        logger.info("UVR batch collected %d file(s)", len(paths))
        for idx, inp_path in enumerate(paths, start=1):
            logger.info("UVR processing file %d/%d: %s", idx, len(paths), inp_path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3=is_hp3
                    )
                    done = 1
            except Exception:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    'ffmpeg -i "%s" -vn -acodec pcm_s16le -ac 2 -ar 44100 "%s" -y'
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except Exception:
                try:
                    if done == 0:
                        pre_fun._path_audio_(
                            inp_path, save_root_ins, save_root_vocal, format0
                        )
                    infos.append("%s->Success" % (os.path.basename(inp_path)))
                    yield "\n".join(infos)
                except Exception:
                    infos.append(
                        "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                    )
                    yield "\n".join(infos)
    except Exception:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Executed torch.cuda.empty_cache()")
    yield "\n".join(infos)
