import os
from pathlib import Path
from typing import Union

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions, get_available_providers, get_device)

from asr.runtime.python.utils.audioHelper import AudioReader
from asr.runtime.python.utils.singleton import singleton
from asr.runtime.python.model.sv.cluster import CommonClustering, make_rttms

@singleton
class Campplus:
    def __init__(self, onnx_path=None, threshold=0.5):
        """
        :param onnx_path: onnx model file path
        :param threshold: threshold of speaker embedding similarity
        """
        self.onnx = onnx_path or os.path.join(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
            ),
            "onnx/sv/campplus.onnx",
        )
        cpu_ep = "CPUExecutionProvider"
        cpu_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
        }

        # device_id = str(0)
        # sess_opt = SessionOptions()
        # sess_opt.intra_op_num_threads = 4
        # sess_opt.log_severity_level = 4
        # sess_opt.enable_cpu_mem_arena = False
        # sess_opt.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # cuda_ep = "CUDAExecutionProvider"
        # cuda_provider_options = {
        #     "device_id": device_id,
        #     "arena_extend_strategy": "kNextPowerOfTwo",
        #     "cudnn_conv_algo_search": "EXHAUSTIVE",
        #     "do_copy_in_default_stream": "true",
        # }

        # cpu_ep = "CPUExecutionProvider"
        # cpu_provider_options = {
        #     "arena_extend_strategy": "kSameAsRequested",
        # }

        # EP_list = []
        # if (
        #     device_id != "-1"
        #     and get_device() == "GPU"
        #     and cuda_ep in get_available_providers()
        # ):
        #     EP_list = [(cuda_ep, cuda_provider_options)]
        # EP_list.append((cpu_ep, cpu_provider_options))

    
        # self.sess = InferenceSession(
        #     self.onnx, sess_options=sess_opt, providers=EP_list
        # )

        self.sess = onnxruntime.InferenceSession(
            self.onnx,
            providers=[
                (cpu_ep, cpu_provider_options),
            ],
        )
        self.output_name = [nd.name for nd in self.sess.get_outputs()]
        self.threshhold = threshold
        self.memory: np.ndarray = None

        self.segments_embs: np.ndarray = None
        self.segments_list = []
        self.cluster_fn = CommonClustering(cluster_type='spectral',
                                           mer_cos=0.8, pval=0.032)

    def compute_cos_similarity(self, emb):
        assert len(emb.shape) == 2, "emb must be length * 80"
        cos_sim = emb.dot(self.memory.T) / (
            np.linalg.norm(emb) * np.linalg.norm(self.memory, axis=1)
        )
        cos_sim[np.isneginf(cos_sim)] = 0

        return 0.5 + 0.5 * cos_sim

    def register_speaker(self, emb: np.ndarray):
        """
        register speaker with embedding and name
        :param emb:
        :param name: speaker name
        :return:
        """
        assert len(emb.shape) == 2, "emb must be length * 80"
        self.memory = np.concatenate(
            (
                self.memory,
                emb,
            )
        )

    def clear_speaker(self):
        self.memory = None
        self.segments_embs = None
        self.segments_list = []

    def extract_feature(self, audio: Union[str, Path, bytes], sample_rate=16000):
        if isinstance(audio, str) or isinstance(audio, Path):
            waveform, sample_rate = AudioReader.read_wav_file(audio)
        elif isinstance(audio, np.ndarray):
            waveform = audio
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = float(sample_rate)
        opts.frame_opts.dither = 0.0
        opts.energy_floor = 1.0
        opts.mel_opts.num_bins = 80
        fbank_fn = knf.OnlineFbank(opts)
        fbank_fn.accept_waveform(sample_rate, waveform.tolist())
        frames = fbank_fn.num_frames_ready
        mat = np.empty([frames, opts.mel_opts.num_bins])
        for i in range(frames):
            mat[i, :] = fbank_fn.get_frame(i)
        feature = mat.astype(np.float32)

        feature = feature - feature.mean(0, keepdims=True)
        feature = feature[None, ...]
        return feature

    def embedding(self, feature: np.ndarray):
        feed_dict = {"fbank": feature}
        output = self.sess.run(self.output_name, input_feed=feed_dict)
        return output

    def recognize(self, waveform: Union[str, Path, bytes], threshold=0.65):
        """
        auto register speaker with input waveform。
        input waveform, output speaker id , id in range 0,1,2,....,n
        :param waveform:
        :return index: if max similarity less than threshold, it will add current emb into memory
        """
        feature = self.extract_feature(waveform)
        emb = self.embedding(feature)[0]

        if self.memory is None:
            self.memory = emb / np.linalg.norm(emb)
            return 1
        
        sim = self.compute_cos_similarity(emb)[0]
        print(threshold, sim)
        
        max_sim_index = np.argmax(sim)

        if sim[max_sim_index] <= threshold:
            if sim[max_sim_index] < 0.75:
                # 新说话人
                self.register_speaker(emb)
                return self.memory.shape[0]
            else:
                # 置信度不足
                return max_sim_index + 1
        else:
            # 置信度较高，对memory中的embeding进行动量更新
            if sim[max_sim_index] >= 0.9:
                memory_emb = self.memory[max_sim_index]
                new_memory_emb =  memory_emb * 0.8 + emb * 0.2           
                self.memory[max_sim_index] = new_memory_emb
        
            return max_sim_index + 1

    def store_segments_emb(self, waveform: Union[str, Path, bytes], start, end):
        feature = self.extract_feature(waveform)
        emb = self.embedding(feature)[0]

        if self.segments_embs is None:
            self.segments_embs = emb
        else:
            self.segments_embs = np.concatenate(
                                (
                                    self.segments_embs,
                                    emb,
                                )
            )
        self.segments_list.append((start, end))

    def cluster(self):
        # cluster
        labels = self.cluster_fn(self.segments_embs)
        
        # output rttm
        # 保持标签顺序，并获取唯一值
        unique_labels, indices = np.unique(labels, return_index=True)

        # 按原始顺序排列唯一值
        unique_labels_sorted_by_indices = unique_labels[np.argsort(indices)]

        new_labels = np.zeros(len(labels),dtype=int)

        for i in range(len(unique_labels_sorted_by_indices)):
            new_labels[labels==unique_labels_sorted_by_indices[i]] = i

        # print(labels, new_labels, uniq) 
        #print(f'new label== {labels}, {new_labels}')
        seg_list = [(i,j) for i,j in zip(self.segments_list, new_labels)]
        #rec_id = embs_file.name.rsplit('.', 1)[0]
        #out_rttm = os.path.join(args.rttm_dir, rec_id+'.rttm')
        return make_rttms(seg_list)
