import os
import json
import base64
import subprocess
import re
import threading
import queue
import paho.mqtt.client as mqtt


# ===================== Config =====================
BROKER = "127.0.0.1"
PORT = 1883
TASK_TOPIC = "rknn/model_quant_compare/request"
RESPONSE_TOPIC_PREFIX = "rknn/model_quant_compare/response"

RKNN_WORK_DIR = "/home/robot/zhangzhuo/rknn_NPU_rknn_model_quant_compare_demo"
RKNN_MODEL_DIR = os.path.join(RKNN_WORK_DIR, "model")
RKNN_IMAGE_DIR = os.path.join(RKNN_WORK_DIR, "image")

os.makedirs(RKNN_MODEL_DIR, exist_ok=True)
os.makedirs(RKNN_IMAGE_DIR, exist_ok=True)


# ===================== Task Queue =====================
task_queue = queue.Queue()


# ===================== Utils =====================
def extract_compare_block(log_text: str) -> str:
    match = re.search(r"===== COMPARE =====", log_text)
    if not match:
        return ""

    tail = log_text[match.end():].strip()
    end = re.search(r"\n===== ", tail)
    return tail[:end.start()].strip() if end else tail


def save_base64_file(path: str, data: str) -> int:
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))
    return os.path.getsize(path)


def run_rknn(model1_path, model2_path, image_path):
    cmd = ["./rknn_npu_quant_compare_demo", model1_path, model2_path, image_path]

    print("\nExecuting:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=RKNN_WORK_DIR,
        capture_output=True,
        text=True
    )

    return {
        "stdout": extract_compare_block(result.stdout),
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def publish_result(client, task_id, status, result=None, error=None):
    topic = f"{RESPONSE_TOPIC_PREFIX}/{task_id}"

    payload = {
        "task_id": task_id,
        "status": status,
    }

    if status == "success":
        payload["result"] = result
    else:
        payload["error"] = error

    client.publish(topic, json.dumps(payload))


# ===================== Worker =====================
def worker():
    while True:
        task = task_queue.get()

        payload = {}
        try:
            print("\n========== Worker Got Task ==========")

            client = task["client"]
            payload = task["payload"]

            task_id = payload["task_id"]
            models = payload["models"]

            if len(models) != 2:
                raise ValueError(f"Expected 2 models, got {len(models)}")

            # -------- save models --------
            print("Saving models...")
            saved_models = []

            for i, m in enumerate(models, 1):
                path = os.path.join(RKNN_MODEL_DIR, m["model_name"])
                size = save_base64_file(path, m["model_data"])
                saved_models.append(path)

                print(f"Model {i}: {path} ({size} bytes)")

            # -------- save image --------
            image_path = os.path.join(RKNN_IMAGE_DIR, payload["image_name"])
            img_size = save_base64_file(image_path, payload["image_data"])

            print("Image saved:", image_path, img_size, "bytes")

            # -------- run inference --------
            print("Running inference...")
            infer_result = run_rknn(
                saved_models[0],
                saved_models[1],
                image_path
            )

            # -------- publish --------
            publish_result(
                client,
                task_id,
                "success",
                result=infer_result
            )

            print("Task success:", task_id)

        except Exception as e:
            print("Worker error:", str(e))

            task_id = payload.get("task_id", "unknown")

            publish_result(
                client,
                task_id,
                "failed",
                error=str(e)
            )

        finally:
            task_queue.task_done()


# ===================== MQTT callbacks =====================
def on_connect(client, userdata, flags, rc):
    print("MQTT Connected")
    client.subscribe(TASK_TOPIC)
    print("Subscribed:", TASK_TOPIC)


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())

        task_id = payload.get("task_id", "unknown")

        print("\nTask received:", task_id)

        # 只入队，不做任何重计算
        task_queue.put({
            "client": client,
            "payload": payload
        })

        print("Task queued:", task_id)

    except Exception as e:
        print("MQTT parse error:", e)


# ===================== Main =====================
def main():
    # start worker thread
    print("Starting worker thread...")
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # MQTT
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting MQTT...")
    client.connect(BROKER, PORT, 60)

    client.loop_forever()


if __name__ == "__main__":
    main()