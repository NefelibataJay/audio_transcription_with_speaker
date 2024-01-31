import threading
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
                self.speaker_identifier.store_segments_emb(wav_data, start_ms, end_ms)
                if do_cluster:
                    seg_list = self.speaker_identifier.cluster()
                
                    for _, start_seg, end_seg, label, alias in seg_list:
                        if start_ms >= start_seg and end_ms <= end_seg:
                            self.result_queue.put(label)
                            break

class Decoder:
    def __init__(self, second_pass_decoder, punctuator, vad, speaker_identifier):
        super().__init__()
        self.second_pass_decoder = second_pass_decoder
        self.punctuator = punctuator
        self.vad = vad
        self.speaker_identifier = speaker_identifier
        self.si_data_queue = Queue()
        self.si_result_queue = Queue()
        self.keywords = []

    def set_keywords(self, keywords):
        self.keywords = keywords
    
    def run(self, wav_file):
        self.result_list = []
        wav, sample_rate = AudioReader.read_wav_file(wav_file)

        wav_length = len(wav)  #wav_length / 16 = total ms
        interval = 9600
        segments_list = []
        segments_text_list = []

        speaker_identification_thread = SpeakerIdentificationThread(self.speaker_identifier, self.si_data_queue, self.si_result_queue)
        speaker_identification_thread.start()

        for offset in range(0, wav_length, min(interval, wav_length)):
            # check if it is the last chunk
            last = False if offset + interval < len(wav) else True
            chunk_data = wav[offset: min(offset + interval, wav_length)]

            # do vad
            segments_result = self.vad.segments_online(chunk_data, is_final=last)

            for start, end in segments_result:
                #print(start, end)
                if start != -1:
                    start_ms = start

                # a valid sentence
                if end != -1:
                    end_frame = end * 16
                    end_ms = end
                    start_frame = start_ms * 16

                    if end_ms - start_ms > 600:
                        data = np.array(wav[start_frame : end_frame])
                        self.si_data_queue.put((data, start_ms, end_ms, False))
                        
                        # asr_offline_final = self.second_pass_decoder.infer_offline(data, hot_words=' '.join(personal_setting.keywords))
                        asr_offline_final = self.second_pass_decoder.infer_offline(data, hot_words=' '.join(self.keywords))
                        _final = self.punctuator.punctuate(asr_offline_final)[0]
                        segments_text_list.append((_final, start_ms, end_ms))

        speaker_identification_thread.stop()
        speaker_identification_thread.join()

        if len(segments_text_list):
            segments_list = self.speaker_identifier.cluster()
            self.vad.vad.all_reset_detection()
            # print(segments_list)
            # print(segments_text_list)
            for text, start, end in segments_text_list:
                for _, start_seg, end_seg, label in segments_list:
                    if start >= start_seg and end <= end_seg:
                        self.result_list.append((text, f"说话人{label+1}", start, end))
                        break

        self.speaker_identifier.clear_speaker()

        return self.result_list
