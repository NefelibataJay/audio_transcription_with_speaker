from asr.runtime.python.cttPunctuator import CttPunctuator
from asr.runtime.python.fsmnVadInfer import FSMNVadOnline
from asr.runtime.python.paraformerInfer import ParaformerOffline

from asr.runtime.python.svInfer import SpeakerVerificationInfer
import shutil
import os


def milisecond_to_str(miliseconds):
    seconds, miliseconds = divmod(miliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return ("%02d:%02d:%02d.%03d" % (hours, minutes, seconds, miliseconds))


def load_model():
    # add other speech models
    speaker_identify_model = SpeakerVerificationInfer(
                model_path="asr/onnx/sv/eres2net-aug-sv.onnx",
                model_name="eres2net",
                threshold=0.85
    )

    vad_model = FSMNVadOnline("asr/onnx/vad/config.yaml")
    vad_model.vad.vad_opts.max_single_segment_time = 10000
    vad_model.vad.vad_opts.max_start_silence_time = 1000

    # asr
    asr_offline_model = ParaformerOffline("asr/onnx/asr_offline", divese_id=1)

    # punctuation
    punctuate_model = CttPunctuator("asr/onnx/punc", online=True)

    return asr_offline_model, punctuate_model, vad_model, speaker_identify_model

def set_result_list(data_dict, data_id, result_list):
    data_dict[data_id]["result_list"] = result_list
    data_dict[data_id]["status"] = "finished"

def clear_files():
    """clear audio files in ./audio"""
    folder = './audio'
    
