import os
import random
import shutil

SOURCE = r"E:\image datasets\RBC,WBC PLT\set"
DEST = r"E:\image datasets\RBC,WBC PLT\eval_subset"
SAMPLES_PER_CLASS = 100

os.makedirs(DEST, exist_ok=True)

for class_name in os.listdir(SOURCE):
    src_class = os.path.join(SOURCE, class_name)
    dst_class = os.path.join(DEST, class_name)

    if not os.path.isdir(src_class):
        continue

    os.makedirs(dst_class, exist_ok=True)

    images = [
        f for f in os.listdir(src_class)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(images) < SAMPLES_PER_CLASS:
        print(f"[WARN] {class_name}: only {len(images)} images available")
        chosen = images
    else:
        chosen = random.sample(images, SAMPLES_PER_CLASS)

    for img in chosen:
        shutil.copy(
            os.path.join(src_class, img),
            os.path.join(dst_class, img)
        )

    print(f"{class_name}: {len(chosen)} images copied")

print("✅ Evaluation subset created.")
