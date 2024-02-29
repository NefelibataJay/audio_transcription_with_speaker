# coding: utf-8
import json
import requests
import time

# url = "http://speech.xiaoyuzhineng.com:19876/external/decode_audio"
# url = "http://localhost:18765/decode_audio"
url = "http://localhost:18765/speaker_recognition"

with open("E:/Desktop/resources/1_test.wav", 'rb') as file:
    files = {'wav_file': ('1.wav', file)}

    response = requests.post(url, files=files, data={
                             "keywords": ["中间协会", "其他关键字"]})
    response = json.loads(response.text)
    data_id = response.get("data")
    print(response)


while data_id != None:
    url = f"http://localhost:18765/get_result/{data_id}"
    # url = f"http://speech.xiaoyuzhineng.com:19876/external/get_decode_result/{data_id}"
    response = requests.get(url)
    json_data = json.loads(response.text)
    if json_data.get("code") == 200:
        print(json_data)
        break
    else:
        print(json_data)
        time.sleep(3)
