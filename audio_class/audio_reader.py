import array
import os.path
import struct
import numpy as np

class AudioReader:
    def __init__(self):
        pass

    @staticmethod
    def get_info(path: str):
        with open(path, 'rb') as f:
            name, data_lengths, _, _, _, _, channels, sample_rate, bit_rate, block_length, sample_bit, _, pcm_length = struct.unpack_from(
                '<4sL4s4sLHHLLHH4sL', f.read(44))
            assert sample_rate == 16000, "sample rate must be 16000"
            nframes = pcm_length // (channels * 2)
        return nframes

    @staticmethod
    def read_wav_bytes(data: bytes):
        """
        convert bytes into array of pcm_s16le data
        :param data: PCM format bytes
        :return:
        """

        # header of wav file
        info = data[:44]
        frames = data[44:]
        name, data_lengths, _, _, _, _, channels, sample_rate, bit_rate, block_length, sample_bit, _, pcm_length = struct.unpack_from(
            '<4sL4s4sLHHLLHH4sL', info)
        # shortArray each element is 16bit
        data = AudioReader.read_pcm_byte(frames)
        return data, sample_rate

    @staticmethod
    def read_pcm_byte(data: bytes):
        short_array = array.array('h')
        short_array.frombytes(data)
        data = np.array(short_array)
        return data

    @staticmethod
    def read_wav_file(file: str):
        if not os.path.exists(file):
            raise FileExistsError(f"audio {file} is not exist.")

        with open(file, 'rb') as file:
            data = file.read()

        return AudioReader.read_wav_bytes(data)
