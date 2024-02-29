# coding: utf-8
import threading
from audio_class.speaker_decoder import SpeakerDecoder
from common import load_model, set_result_list
from flask import Flask, request, jsonify
import os
import time
from waitress import serve
from queue import Queue
from audio_class.audio_decoder import Decoder
import uuid

app = Flask(__name__)

# load model

# create task queue
task_queue = Queue()
# 存储所有任务的状态
tasks_dict = {
    # "task_id": {"file_path": "", "status": "running", "result_list": [], type: "speaker"}
}

# load model
second_pass_decoder, punctuator, vad, speaker_identifier = load_model()

global_keywords = []


def load_keywords():
    with open("./hotwords", "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and len(line) <= 10:
                global_keywords.append(line)

    print(f'keywords num: {len(global_keywords)}')


def execute_task():
    load_keywords()
    while True:
        # wait for task
        if task_queue.empty():
            time.sleep(10)
            continue

        # get task
        task_id = task_queue.queue[0]
        print(f"开始执行任务--{task_id}")

        tasks_dict[task_id]['status'] = 'running'
        wav_file = tasks_dict[task_id]["file_path"]
        # keywords = tasks_dict[task_id]["keywords"]

       # print(type(keywords), keywords)
       # keywords.extend(global_keywords)
       # print(f'keywords updates: {len(keywords)}')

        # start decoder thread
        if tasks_dict[task_id]["type"] == "speaker":
            audio_decoder = SpeakerDecoder(speaker_identifier, vad)
        else:
            # type = "audio"
            audio_decoder = Decoder(second_pass_decoder,
                                    punctuator, vad, speaker_identifier)

            audio_decoder.set_keywords(global_keywords)

        tasks_dict[task_id]['result_list'] = audio_decoder.run(wav_file)
        tasks_dict[task_id]['status'] = 'Completed'
        file_path = tasks_dict[task_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        # remove task
        task_queue.get()
        print(f"任务--{task_id}--执行完毕")


@app.route('/get_result/<task_id>', methods=['GET'])
def get_result(task_id):
    print(f"获取任务--{task_id}--的结果")
    if task_id not in tasks_dict:
        return_data = {
            'code': 400,
            'message': "还未创建任务, 或任务数据已被获取导致任务过期",
            "data": None
        }
        return jsonify(return_data)
    elif tasks_dict[task_id]["status"] == "running":
        return_data = {
            'code': 400,
            'message': "任务还在处理中",
            "data": None
        }
        return jsonify(return_data)
    else:
        result_list = tasks_dict[task_id]["result_list"]
        tasks_dict.pop(task_id)
        return_data = {
            'code': 200,
            'message': "请求成功",
            "data": result_list
        }
        return jsonify(return_data)


@app.route('/decode_audio', methods=['POST'])
def decode_audio_api():
    if "wav_file" not in request.files:
        return_data = {
            'code': 400,
            'message': "文件上传失败",
            "data": None
        }
        return jsonify(return_data)

    file = request.files['wav_file']
    # check file type

    # create task_id
    task_id = str(uuid.uuid4())

    # save file
    save_path = os.path.join(f"{os.getcwd()}/audio", f"{task_id}.wav")
    file.save(save_path)

    # 判断文件是否是wav格式
    if not save_path.endswith(".wav"):
        os.remove(save_path)
        return_data = {
            'code': 400,
            'message': "文件格式错误",
            "data": None
        }
        return jsonify(return_data)

    tasks_dict[task_id] = {"file_path": save_path, "status": "running",
                           "result_list": [], "keywords": [], "type": "audio"}

    # keywords = ["热词1", "热词2"]
    if request.form.get("keywords") is not None:
        keywords = request.form.get("keywords")
        tasks_dict[task_id]["keywords"] = keywords
    else:
        keywords = []

    try:
        task_queue.put(task_id)

    except Exception as e:
        print(e)
        return_data = {
            'code': 400,
            'message': "模型加载失败",
            "data": None
        }
        return jsonify(return_data)

    data = {
        'code': 200,
        'message': f"文件上传成功,前面还有{task_queue.qsize() - 1}个任务在执行中",
        "data": task_id
    }
    return jsonify(data)


@app.route('/speaker_recognition', methods=['POST'])
def speaker_recognition():
    """
    说话人识别
    """
    if "wav_file" not in request.files:
        return_data = {
            'code': 400,
            'message': "文件上传失败",
            "data": None
        }
        return jsonify(return_data)
    file = request.files['wav_file']
    # check file type

    # create task_id
    task_id = str(uuid.uuid4())

    # save file
    save_path = os.path.join(f"{os.getcwd()}/audio", f"{task_id}.wav")
    file.save(save_path)

    # 判断文件是否是wav格式
    if not save_path.endswith(".wav"):
        os.remove(save_path)
        return_data = {
            'code': 400,
            'message': "文件格式错误",
            "data": None
        }
        return jsonify(return_data)

    tasks_dict[task_id] = {"file_path": save_path, "status": "running",
                           "result_list": [], "keywords": [], "type": "speaker"}

    try:
        task_queue.put(task_id)

    except Exception as e:
        print(e)
        return_data = {
            'code': 400,
            'message': "模型加载失败",
            "data": None
        }
        return jsonify(return_data)

    data = {
        'code': 200,
        'message': f"文件上传成功,前面还有{task_queue.qsize() - 1}个任务在执行中",
        "data": task_id
    }
    return jsonify(data)


if __name__ == '__main__':
    app.json.ensure_ascii = False
    threading.Thread(target=execute_task, args=(), daemon=True).start()
    serve(app, host="0.0.0.0", port=18765)
