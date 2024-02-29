import threading
import time
from typing import List
from asr.runtime.python.svInfer import SpeakerVerificationInfer
from audio_class.audio_reader import AudioReader
from queue import Queue
import numpy as np

from common import set_result_list


class SpeakerIdentificationThread(threading.Thread):
    def __init__(self, speaker_identifier, data_queue, result_queue):
        threading.Thread.__init__(self)
        self.speaker_identifier = speaker_identifier
        self.__running = threading.Event()
        self.__running.set()
        self.data_queue = data_queue
        self.result_queue = result_queue

    def stop(self):
        self.__running.clear()
        self.data_queue.put((None, None, None, None))

    def run(self):
        while self.__running.is_set():
            wav_data, start_ms, end_ms, do_cluster = self.data_queue.get()
            if wav_data is not None:
                self.speaker_identifier.store_segments_emb(
                    wav_data, start_ms, end_ms)
                if do_cluster:
                    seg_list = self.speaker_identifier.cluster()

                    for _, start_seg, end_seg, label, alias in seg_list:
                        if start_ms >= start_seg and end_ms <= end_seg:
                            self.result_queue.put(label)
                            break


class SpeakerDecoder:
    def __init__(self) -> None:
        self.model = speaker_identify_model = SpeakerVerificationInfer(
            model_path="asr/onnx/sv/eres2net-aug-sv.onnx",
            model_name="eres2net",
            threshold=0.85
        )
        self.si_data_queue = Queue()
        self.si_result_queue = Queue()

    def run(self, time_list, wav_file) -> List[List[int]]:
        """
        :param time_list: [(start, end), (start, end), ...]
        :param wav_file: file path

        :return: [[start, end, speaker], [start, end, speaker], ...]
        """
        try:
            self.result_list = []
            wav, sample_rate = AudioReader.read_wav_file(wav_file)

            wav_length = len(wav)  # wav_length / 16 = total ms

            for start, end in time_list:
                if start < 0 or end > wav_length:
                    break

                start_ms = start
                end_frame = end * 16
                end_ms = end
                start_frame = start_ms * 16

                if end_ms - start_ms > 800:
                    data = np.array(wav[start_frame: end_frame])
                    self.model.store_segments_emb(data, start_ms, end_ms)

            segments_list = self.model.cluster()

            for start, end in time_list:
                for _, start_seg, end_seg, label in segments_list:
                    if start >= start_seg and end <= end_seg:
                        self.result_list.append((f"说话人{label+1}", start, end))
                        break

            self.model.clear_speaker()

            print(f"speaker result: {self.result_list}")

            return self.result_list

        except Exception as e:
            print(f"run error: {e}")
            return [[]]
