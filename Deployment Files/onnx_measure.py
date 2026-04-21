import onnxruntime as ort
import numpy as np
import time
import psutil, os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# =========================
# ⚙️ CONFIG
# =========================
MODEL_PATH = "mobilemodel_int8.onnx"  
# CHANGE to "model_int8.onnx" for second run

DATASET_PATH = "../test"
NUM_LATENCY_SAMPLES = 100

# =========================
# 📊 RAM FUNCTION
# =========================
def get_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB


# =========================
# 🧠 LOAD MODEL
# =========================
def load_model(path):
    print(f"\n===== Measuring {path} =====")

    baseline_ram = get_ram()
    print(f"Baseline RAM: {baseline_ram:.2f} MB")

    session = ort.InferenceSession(
        path,
        providers=['CPUExecutionProvider']
    )

    input_name = session.get_inputs()[0].name

    after_load_ram = get_ram()
    print(f"After Load RAM: {after_load_ram:.2f} MB")

    print(f"Model Load Increase: {after_load_ram - baseline_ram:.2f} MB")

    return session, input_name, baseline_ram, after_load_ram


# =========================
# 🖼️ LOAD IMAGES
# =========================
def load_images(folder):
    transform = transforms.Compose([
        transforms.Resize((110, 100)),
        transforms.ToTensor()
    ])

    images = []
    labels = []

    class_names = sorted(os.listdir(folder))

    for label, cls in enumerate(class_names):
        cls_path = os.path.join(folder, cls)

        for file in os.listdir(cls_path):
            img_path = os.path.join(cls_path, file)

            img = Image.open(img_path).convert("RGB")
            img = transform(img).numpy()

            images.append(img)
            labels.append(label)

    return images, labels


# =========================
# 🎯 ACCURACY
# =========================
def evaluate(session, input_name, images, labels):
    print("\nRunning accuracy...")

    correct = 0

    for i in tqdm(range(len(images))):
        input_data = np.expand_dims(images[i], axis=0).astype(np.float32)

        outputs = session.run(None, {input_name: input_data})
        pred = np.argmax(outputs[0], axis=1)[0]

        if pred == labels[i]:
            correct += 1

    acc = 100 * correct / len(images)
    print(f"Accuracy: {acc:.2f}%")

    return acc


# =========================
# ⚡ LATENCY
# =========================
def measure_latency(session, input_name, images):
    print("\nMeasuring latency...")

    images = images[:NUM_LATENCY_SAMPLES]

    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: np.expand_dims(images[0], axis=0).astype(np.float32)})

    total_time = 0

    for img in images:
        input_data = np.expand_dims(img, axis=0).astype(np.float32)

        start = time.time()
        _ = session.run(None, {input_name: input_data})
        total_time += (time.time() - start)

    latency = (total_time / len(images)) * 1000
    print(f"Latency: {latency:.2f} ms")

    return latency


# =========================
# 💾 STEADY-STATE RAM
# =========================
def measure_steady_ram(session, input_name):
    print("\nMeasuring steady-state RAM...")

    dummy = np.random.randn(1, 3, 110, 100).astype(np.float32)

    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy})

    ram_before = get_ram()

    for _ in range(100):
        _ = session.run(None, {input_name: dummy})

    ram_after = get_ram()

    print(f"Steady-State RAM: {ram_after:.2f} MB")

    return ram_after


# =========================
# 🚀 MAIN
# =========================
def main():
    session, input_name, base_ram, load_ram = load_model(MODEL_PATH)
    print("\n MobileNetV2...")
    print("\nLoading dataset...")
    images, labels = load_images(DATASET_PATH)

    acc = evaluate(session, input_name, images, labels)
    lat = measure_latency(session, input_name, images)
    steady_ram = measure_steady_ram(session, input_name)

    print("\n===== FINAL RESULTS =====")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Latency: {lat:.2f} ms")
    print(f"Baseline RAM: {base_ram:.2f} MB")
    print(f"After Load RAM: {load_ram:.2f} MB")
    print(f"Steady-State RAM: {steady_ram:.2f} MB")


if __name__ == "__main__":
    main()