import json
import torch
import numpy as np
import re
import sys
import os
from cached_path import cached_path
from f5_tts.infer.utils_infer import (
    load_model as f5_load_model,
    load_vocoder as f5_load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
    cross_fade_duration,
    nfe_step as default_nfe_step,
)
from f5_tts.model import DiT

_F5TTS_model = None
_vocoder = None
_ref_audio_processed = None
_ref_text_processed = None
_current_ref_audio_path = ""
_current_ref_text_from_gui = ""

# 获取资源目录的基路径
if getattr(sys, 'frozen', False):
    # 如果是 PyInstaller 打包的，使用 _MEIPASS
    # base_path = sys._MEIPASS
    base_path = os.path.dirname(sys.executable) # .exe 所在的目录
else:
    # 否则，使用脚本所在的目录
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path =os.path.join(base_path, '..')

DEFAULT_TTS_MODEL_CFG_JSON_STR = json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4))
DEFAULT_VOCODER_LOCAL_PATH = os.path.join(base_path, 'resources', 'vocos-mel-24khz')
DEFAULT_SPEED = 1.0
DEFAULT_SEED = 1

def arabic_to_chinese_digits(text):
    num_map = {
        '0': '零', '1': '一', '2': '二', '3': '三',
        '4': '四', '5': '五', '6': '六', '7': '七',
        '8': '八', '9': '九'
    }
    return ''.join(num_map.get(c, c) for c in text)

def clean_text_for_tts(text):
    text = re.sub(r"[\s\u3000]+", " ", text)
    text = text.strip()
    return text

def _load_f5tts_model_internal(model_path_gui, vocab_path_gui, model_config_json_gui):
    try:
        ckpt_path = str(cached_path(model_path_gui))
        model_config_dict = json.loads(model_config_json_gui or DEFAULT_TTS_MODEL_CFG_JSON_STR)
        model = f5_load_model(DiT, model_config_dict, ckpt_path, vocab_file=vocab_path_gui)
        return model
    except Exception as e:
        print(f"Error loading F5TTS model: {e}")
        return None

def _load_tts_vocoder_internal(vocoder_local_path_gui):
    try:
        vocoder_instance = f5_load_vocoder(is_local=True, local_path=vocoder_local_path_gui or DEFAULT_VOCODER_LOCAL_PATH)
        return vocoder_instance
    except Exception as e:
        print(f"Error loading TTS vocoder: {e}")
        return None

def load_tts_resources(model_path_gui, vocab_path_gui, model_config_json_gui,
                       ref_audio_path_gui, ref_text_gui, vocoder_local_path_gui):
    global _F5TTS_model, _vocoder, _ref_audio_processed, _ref_text_processed
    global _current_ref_audio_path, _current_ref_text_from_gui

    _vocoder = _load_tts_vocoder_internal(vocoder_local_path_gui)
    _F5TTS_model = _load_f5tts_model_internal(model_path_gui, vocab_path_gui, model_config_json_gui)

    if not _vocoder or not _F5TTS_model:
        return False

    if (_current_ref_audio_path != ref_audio_path_gui or
            _current_ref_text_from_gui != ref_text_gui or
            _ref_audio_processed is None):
        try:
            _ref_audio_processed, _ref_text_processed = preprocess_ref_audio_text(ref_audio_path_gui, ref_text_gui)
            _current_ref_audio_path = ref_audio_path_gui
            _current_ref_text_from_gui = ref_text_gui
        except Exception as e:
            print(f"Error preprocessing reference audio/text: {e}")
            _ref_audio_processed, _ref_text_processed = None, None
            return False
    return _F5TTS_model is not None and _vocoder is not None and _ref_audio_processed is not None

def is_tts_ready():
    return _F5TTS_model is not None and _vocoder is not None and _ref_audio_processed is not None

def generate_audio_chunk(text_for_tts, tts_speed, seed=DEFAULT_SEED, nfe_steps_override=None):
    if not is_tts_ready():
        raise RuntimeError("TTS resources are not loaded. Call load_tts_resources first.")

    torch.manual_seed(seed)
    cleaned_text = clean_text_for_tts(text_for_tts)

    current_nfe_step = nfe_steps_override if nfe_steps_override is not None else default_nfe_step

    try:
        if _ref_audio_processed is None or _ref_text_processed is None:
            raise ValueError("Reference audio/text not preprocessed.")

        audio_chunk_data, sample_rate, _ = infer_process(
            _ref_audio_processed,
            _ref_text_processed,
            str(cleaned_text),
            _F5TTS_model,
            _vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=current_nfe_step,
            speed=tts_speed,
            progress=None,
        )
        if audio_chunk_data.dtype != np.float32:
            audio_chunk_data = audio_chunk_data.astype(np.float32)
        return audio_chunk_data, sample_rate
    except Exception as e:
        print(f"ERROR: TTS generation failed for chunk: '{cleaned_text}'")
        raise