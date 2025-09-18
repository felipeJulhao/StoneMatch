import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def process_image(path):
    # --- pipeline principal ---
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    features = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / h

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, -1)

        mean_color = cv2.mean(img, mask=mask)[:3]
        mean_depth = 0

        features.append({
            "contour": c,
            "vector": [area, circularity, aspect_ratio,
                    mean_color[0], mean_color[1], mean_color[2], mean_depth],
            "image": img,
            "path": path
        })
    
    return features

#Carregar rede MiDaS
#net = cv2.dnn.readNet("modelos/model-small.onnx")


def processar():
    # --- Carregar várias imagens ---
    image_dir = "C:/Users/igorb/Desktop/CodeCon/pedras/pedras_separadas"  # coloque o nome da pasta aqui
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

    all_features = []
    for path in image_paths:
        feats = process_image(path)
        all_features.extend(feats)

    # --- Comparar todas as pedras ---
    best_pair = None
    min_dist = float("inf")

    for i in range(len(all_features)):
        for j in range(i+1, len(all_features)):
            v1 = np.array(all_features[i]["vector"])
            v2 = np.array(all_features[j]["vector"])
            dist = np.linalg.norm(v1 - v2)
            if dist < min_dist:
                min_dist = dist
                best_pair = (all_features[i], all_features[j])

    # --- Mostrar resultado ---
    if best_pair:
        img1 = best_pair[0]["image"].copy()
        img2 = best_pair[1]["image"].copy()

        cv2.drawContours(img1, [best_pair[0]["contour"]], -1, (0,255,0), 3)
        cv2.drawContours(img2, [best_pair[1]["contour"]], -1, (0,255,0), 3)

        fig, axes = plt.subplots(1, 2, figsize=(10,5))
        axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Pedra 1 ({os.path.basename(best_pair[0]['path'])})")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Pedra 2 ({os.path.basename(best_pair[1]['path'])})")
        axes[1].axis("off")

        plt.suptitle(f"Par mais semelhante (distância = {min_dist:.2f})")
        plt.show()