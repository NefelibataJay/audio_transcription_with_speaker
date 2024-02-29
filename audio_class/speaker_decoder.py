import threading
import time
from typing import List
from audio_class.audio_reader import AudioReader
from queue import Queue
import numpy as np

from common import set_result_list


class SpeakerDecoder:
    def __init__(self, speaker_model, vad_model) -> None:
        self.model = speaker_model
        self.vad = vad_model

    def run(self, wav_file):
        self.result_list = []
        wav, sample_rate = AudioReader.read_wav_file(wav_file)

        wav_length = len(wav)  # wav_length / 16 = total ms
        interval = 12800
        segments_list = []
        segments_text_list = []

        for offset in range(0, wav_length, min(interval, wav_length)):
            # check if it is the last chunk
            last = False if offset + interval < len(wav) else True
            chunk_data = wav[offset: min(offset + interval, wav_length)]

            # do vad
            segments_result = self.vad.segments_online(
                chunk_data, is_final=last)

            for start, end in segments_result:
                # print(start, end)
                if start != -1:
                    start_ms = start

                # a valid sentence
                if end != -1:
                    end_frame = end * 16
                    end_ms = end
                    start_frame = start_ms * 16

                    if end_ms - start_ms > 800:
                        data = np.array(wav[start_frame: end_frame])
                        self.model.store_segments_emb(data, start_ms, end_ms)
                        segments_text_list.append((start_ms, end_ms))

        if len(segments_text_list):
            segments_list = self.model.cluster()
            self.vad.vad.all_reset_detection()
            # print(segments_list)
            # print(segments_text_list)
            for start, end in segments_text_list:
                for _, start_seg, end_seg, label in segments_list:
                    if start >= start_seg and end <= end_seg:
                        self.result_list.append(
                            (f"说话人{label+1}", start, end))
                        break

        self.model.clear_speaker()

        return self.result_list
