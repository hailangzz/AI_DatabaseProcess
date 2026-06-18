import base64
import json
import os
import uuid

import paho.mqtt.client as mqtt

# ===================== Config =====================
BROKER = "172.16.50.91"
PORT = 1883

REQUEST_TOPIC = "rknn/infer/request"
RESPONSE_TOPIC_PREFIX = "rknn/infer/response"

MODEL_FILE = (
    "/home/chenkejing/PycharmProjects/ultralytics/rknn_models/carpet_f_seg_0512.rknn"
)
# MODEL_FILE = "/home/chenkejing/Downloads/best.rknn"
IMAGE_FILE = "/home/chenkejing/PycharmProjects/ultralytics/images_mode_test/carpet_images_test/ca216e1a94cbbc27d8c010e21a85ddf7.jpg"

# 新增：输出图片保存路径
SAVE_OUTPUT_IMAGE_PATH = "./out.png"

TASK_ID = str(uuid.uuid4())


# ===================== Utils =====================
def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_base64_image(data: str, save_path: str):
    if not data:
        print("无输出图片数据")
        return

    img_bytes = base64.b64decode(data)

    with open(save_path, "wb") as f:
        f.write(img_bytes)

    print(f"\n输出图片已保存: {save_path}")
    print("图片大小:", len(img_bytes), "bytes")


def build_response_topic(task_id: str) -> str:
    return f"{RESPONSE_TOPIC_PREFIX}/{task_id}"


def print_json(result: dict):
    print("\n========== 收到返回结果 ==========")
    print("task_id:", result["task_id"])
    print("status :", result["status"])

    if "result" in result:
        print("\n===== stdout =====\n")
        print(result["result"]["stdout"])

        print("\n===== stderr =====\n")
        print(result["result"]["stderr"])

        print("\nreturncode:", result["result"]["returncode"])

    # ===================== 新增：处理输出图片 =====================
    if "output_image" in result and result["output_image"]:
        print("\n===== output2 image received =====")
        save_base64_image(result["output_image"]["data"], SAVE_OUTPUT_IMAGE_PATH)


# ===================== MQTT Logic =====================
def send_task(client):
    # print("读取RKNN模型...")
    model_bytes = file_to_base64(MODEL_FILE)
    # print("Base64模型长度:", len(model_bytes))

    # print("读取测试图片...")
    image_bytes = file_to_base64(IMAGE_FILE)
    # print("Base64图片长度:", len(image_bytes))

    payload = {
        "task_id": TASK_ID,
        "model_name": os.path.basename(MODEL_FILE),
        "model_data": model_bytes,
        "image_name": os.path.basename(IMAGE_FILE),
        "image_data": image_bytes,
    }

    print("os.path.basename(MODEL_FILE):", os.path.basename(MODEL_FILE))

    msg = json.dumps(payload)

    # print("MQTT消息大小:", len(msg.encode()), "bytes")os.path.basename(MODEL_FILE)

    ret = client.publish(REQUEST_TOPIC, msg, qos=1)

    if ret.rc == mqtt.MQTT_ERR_SUCCESS:
        print("发送任务成功")
        print("task_id =", TASK_ID)
    else:
        print("发送失败")


def on_connect(client, userdata, flags, reason_code, properties):
    print("MQTT connected")

    topic = build_response_topic(TASK_ID)
    client.subscribe(topic)

    print("subscribe ->", topic)

    send_task(client)


def on_message(client, userdata, msg):
    result = json.loads(msg.payload.decode())

    print_json(result)

    client.disconnect()


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    print("MQTT disconnected")


# ===================== Main =====================
def main():
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    print("connecting...")

    client.connect(BROKER, PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
