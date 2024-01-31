from asyncio import subprocess
import os
import struct
import wave
#import wave
import numpy as np

class AudioHelper:
    def __init__(self, 
                chunk_id=b'RIFF',
                chunk_size=36,
                format=b'WAVE',
                sub_chunk_1_id=b'fmt ',
                sub_chunk_1_size=16,
                audio_format=1,
                num_channels=1, 
                sample_rate=16000,
                sub_chunk_2_id=b'data',
                sample_width=2,
                data=b'') -> None:
        """
        Args:  
            chunk_id: 内容为`RIFF`
            chunk_size: 存储文件的字节数（不包括ChunkID和ChunkSize这8个字节）
            format: 内容为`WAVE`
            sub_chunk_1_id: 内容为`fmt `
            sub_chunk_1_size: 存储该子块的字节数,一般为16
            audio_format: 存储音频的编码格式 PCM为1
            num_channels: 通道数，单通道为1，双通道为2
            sample_rate: 采样率
            bits_rate: 比特率，每秒存储的bit数 =sample_rate*num_channels*bit_per_samples/8
            block_length: 块对齐大小 =num_channels*bit_per_samples/8
            bits_per_sample: 采样位宽，每个采样点的bit数
            sub_chunk_2_id: 内容为`data`
            pcm_length: 正式数据部分的字节数
            data: 音频二进制数据
        """

        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.format = format
        self.sub_chunk_1_id = sub_chunk_1_id
        self.sub_chunk_1_size = sub_chunk_1_size
        self.audio_format = audio_format
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.block_length = int(num_channels*sample_width)
        self.bits_per_sample = sample_width * 8
        self.sub_chunk_2_id = sub_chunk_2_id
        self.pcm_length = 0
        self.sample_width = sample_width
        self.bits_rate = int(sample_rate*num_channels*sample_width)
        self.pcm_length = len(data)
        self.data = data
        self._file = None

    def read(self, file_path: str) -> None:
        """
        Args:
            file_path(str): absolute path of wav file

        """
        assert file_path.endswith('wav'), 'do not support other file format of audio except wav'

        with open(file_path, 'rb') as f:
            chunk_id, chunk_size, format, sub_chunk_1_id, sub_chunk_1_size, \
            audio_format, num_channels, sample_rate, bits_rate, block_length,\
            bits_per_sample, sub_chunk_2_id, pcm_length = struct.unpack_from('<4sL4s4sLHHLLHH4sL', f.read(44))

            data = f.read(pcm_length)

        assert chunk_id == b'RIFF', 'do not support other encode format of audio except PCM'

        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.format = format
        self.sub_chunk_1_id = sub_chunk_1_id
        self.sub_chunk_1_size = sub_chunk_1_size
        self.audio_format = audio_format
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.block_length = block_length
        self.bits_per_sample = bits_per_sample
        self.sub_chunk_2_id = sub_chunk_2_id
        self.pcm_length = pcm_length
        self.sample_width = bits_per_sample / 8
        self.bits_rate = bits_rate
        self.data = data
       
        
        
    def append(self, buffer):
        """
        向音频文件追加语音数据
        Args:
            buffer: 二进制数据
        """
        self.data += buffer
        self.pcm_length += len(buffer)
        return self

    def append_sync(self, buffer):
        """
        不增加内存的情况下追加wav文件内容
        """

        self._file.seek(4, 0)
        data =  self._file.read(4)
        sub_chunk_1_size = struct.unpack('<I', data)[0]
        sub_chunk_1_size += len(buffer)
        self._file.seek(4, 0)
        self._file.write(struct.pack('<I', sub_chunk_1_size))

        self._file.seek(40, 0)
        data = self._file.read(4)
        sub_chunk_2_size = struct.unpack('<I', data)[0]
        sub_chunk_2_size += len(buffer)
        self._file.seek(40, 0)
        self._file.write(struct.pack('<I', sub_chunk_2_size))
        

        self._file.seek(0, 2)
        self._file.write(buffer)
        return self

    def open(self, file):
        is_exist = os.path.exists(file)
        #print(is_exist)
        if is_exist:
            f = open(file, 'rb+')
            self.read(file)
        else:     
            f = open(file, 'wb+')
            header = struct.pack('<4sL4s4sLHHLLHH4sL', self.chunk_id,
                        36, self.format, self.sub_chunk_1_id, self.sub_chunk_1_size,
                        self.audio_format, self.num_channels, self.sample_rate,
                        self.bits_rate,
                        self.block_length,
                        self.bits_per_sample,
                        self.sub_chunk_2_id,
                        0)
            f.write(header)

        self._file = f
        return self

    def read_partation(self, file_path, offset, chunk_size):
        """读取音频部分数据"""
        with open(file_path, 'rb') as f:
            f.seek(offset+44)
            data = f.read(chunk_size)
        return data

    def close(self):
        if self._file:
            self._file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def export(self, output_path):    
        header = struct.pack('<4sL4s4sLHHLLHH4sL', self.chunk_id,
                            36 + self.pcm_length, self.format, self.sub_chunk_1_id, self.sub_chunk_1_size,
                            self.audio_format, self.num_channels, self.sample_rate,
                            self.bits_rate,
                            self.block_length,
                            self.bits_per_sample,
                            self.sub_chunk_2_id,
                            self.pcm_length)
        
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(self.data)

        return self
    
    @staticmethod
    def downsample(input: np.ndarray, input_sample_rate, output_sample_rate) -> np.ndarray:
        """
        语音下采样， 输入采样率必须大于输出采样率
        Args:
            input: 语音输入，numpy对象
            input_sample_rate: 输入采样率
            output_sample_rate: 输出采样率
        """
        assert input_sample_rate > output_sample_rate, f"input sample rate:{input_sample_rate} < output sample rate:{output_sample_rate}"
        dtype = input.dtype
        audio_len = len(input)
        audio_time_max = 1.0*(audio_len-1) / input_sample_rate
        src_time = 1.0 * np.linspace(0,audio_len,audio_len) / input_sample_rate
        tar_time = 1.0 * np.linspace(0,np.int32(audio_time_max*output_sample_rate),np.int32(audio_time_max*output_sample_rate)) / output_sample_rate
        output = np.interp(tar_time,src_time, input).astype(dtype)
        return output


    @staticmethod
    def resample_to_PCM16le(input_path, out_path, target_sample_rate=16000):
        """
        wav文件下采样并保存
        Args:
            input_path: 输入音频文件路径
            out_path: 输出音频文件路径
            target_sample_rate: 目标采样率，默认为16000
        """
        with wave.open(input_path, 'rb') as audio:
            input_sample_rate = audio.getframerate()
            audio_data = audio.readframes(audio.getnframes())
            #print(input_sample_rate)

        if input_sample_rate > target_sample_rate:
            output = AudioHelper.downsample(np.frombuffer(audio_data, np.short), input_sample_rate, target_sample_rate)

            with wave.open(out_path, 'wb') as audio:
                audio.setframerate(target_sample_rate)
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio_data = output.tobytes()
                audio.writeframes(audio_data)
        elif input_sample_rate == target_sample_rate:
             with open(out_path, 'wb') as audio_output:
                with open(input_path, 'rb') as audio_input:
                    audio_output.write(audio_input.read())
        else:
            raise ValueError('can not handle up-sample right now') #up-sample

    @staticmethod
    def get_wav_duration(wav_path):
        with wave.open(wav_path, 'r') as f:
            try:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
            except Exception:
                duration = 0
                
        return duration

if __name__ == '__main__':
    # test 
    audio1_path = "/home/zlf/DTM1_050_ahead_000_G0100_S2092_3-convert.wav"
    output_path = "/home/zlf/appended.wav"

    with open(audio1_path, 'rb') as f:
        _ = f.read(44)
        audio1_data = f.read()
    
    new_audio = AudioHelper()
    with new_audio.open(output_path):
        for i in range(2):
            # audio2_data: 二进制数据
            new_audio.append_sync(audio1_data)
