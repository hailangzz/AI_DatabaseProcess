import json
import base64
import subprocess
import os
import paho.mqtt.client as mqtt


# ===================== Config =====================
BROKER = "0.0.0.0"
PORT = 1883

TASK_TOPIC = "rknn/infer/request"
RESPONSE_TOPIC_PREFIX = "rknn/infer/response"

RKNN_WORK_DIR = "/home/robot/zhangzhuo/rknn_yolov8_seg_demo"
RKNN_MODEL_DIR = os.path.join(RKNN_WORK_DIR, "model")
RKNN_IMAGE_DIR = os.path.join(RKNN_WORK_DIR, "image")

# 关键：输出图片路径
OUTPUT_IMAGE_PATH = os.path.join(RKNN_WORK_DIR, "out.png")

os.makedirs(RKNN_MODEL_DIR, exist_ok=True)
os.makedirs(RKNN_IMAGE_DIR, exist_ok=True)


# ===================== Utils =====================
def save_base64_file(path: str, data: str) -> int:
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))
    return os.path.getsize(path)


def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_rknn(model_path: str, image_path: str) -> dict:
    cmd = ["./rknn_yolov8_seg_demo", model_path, image_path]

    print("Executing:", cmd)

    result = subprocess.run(
        cmd,
        cwd=RKNN_WORK_DIR,
        capture_output=True,
        text=True
    )

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def build_response_topic(task_id: str) -> str:
    return f"{RESPONSE_TOPIC_PREFIX}/{task_id}"


# ===================== MQTT Logic =====================
def on_connect(client, userdata, flags, rc):
    print("MQTT Connected")
    client.subscribe(TASK_TOPIC)
    print("subscribe:", TASK_TOPIC)


def on_message(client, userdata, msg):
    payload = {}

    try:
        print("\n收到任务")

        payload = json.loads(msg.payload.decode())

        task_id = payload["task_id"]

        model_name = payload["model_name"]
        model_data = payload["model_data"]

        image_name = payload["image_name"]
        image_data = payload["image_data"]

        # ---------------- Model ----------------
        model_path = os.path.join(RKNN_MODEL_DIR, model_name)
        model_size = save_base64_file(model_path, model_data)

        print("保存模型:", model_path)
        print("模型大小:", model_size)

        # ---------------- Image ----------------
        image_path = os.path.join(RKNN_IMAGE_DIR, image_name)
        image_size = save_base64_file(image_path, image_data)

        print("保存图片:", image_path)
        print("图片大小:", image_size)

        # ---------------- Inference ----------------
        infer_result = run_rknn(model_path, image_path)

        # ---------------- 读取输出图片 out.png ----------------
        output_image_data = None

        if os.path.exists(OUTPUT_IMAGE_PATH):
            output_image_data = file_to_base64(OUTPUT_IMAGE_PATH)
            print("输出图片读取成功:", len(output_image_data))
        else:
            print("警告: 未找到 out.png")

        # ---------------- 返回结果 ----------------
        result = {
            "task_id": task_id,
            "status": "success",
            "result": infer_result,
            "output_image": {
                "name": "out.png",
                "data": output_image_data
            } if output_image_data else None
        }

        topic = build_response_topic(task_id)

        client.publish(topic, json.dumps(result))

        print("结果已返回:", topic)

    except Exception as e:
        print("异常:", str(e))

        task_id = payload.get("task_id", "unknown")

        result = {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "output_image": None
        }

        topic = build_response_topic(task_id)

        client.publish(topic, json.dumps(result))


# ===================== Main =====================
def main():
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT)
    client.loop_forever()


if __name__ == "__main__":
    main()