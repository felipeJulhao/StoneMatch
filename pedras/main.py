import cv2
import numpy as np
import os

from paridade import processar

# --- Configurações ---
input_image = "pedras.jpg"
output_folder = "pedras_separadas"
os.makedirs(output_folder, exist_ok=True)

# Carregar imagem e converter para cinza
image = cv2.imread("pedras.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

# Threshold global
_, thresh_global = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV)

# --- Remover pequenos ruídos ---
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh_global, cv2.MORPH_OPEN, kernel, iterations=2)

# --- Background seguro ---
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# --- Foreground seguro usando Distance Transform ---
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# --- Região desconhecida ---
unknown = cv2.subtract(sure_bg, sure_fg)

# --- Marcadores para Watershed ---
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# --- Aplicar Watershed ---
markers = cv2.watershed(image, markers)

# --- Separar cada pedra ---
for label in range(2, markers.max()+1):  # 1 é background
    mask = np.uint8(markers == label) * 255
    
    # Criar imagem da pedra
    pedra = cv2.bitwise_and(image, image, mask=mask)
    
    # Salvar cada pedra
    cv2.imwrite(os.path.join(output_folder, f"pedra_{label-1}.jpg"), pedra)

processar()

#res = cv2.bitwise_and(image, image, mask=thresh_global)

# Mostrar a imagem
#cv2.imwrite("saida.jpg", res)
#cv2.imshow("Threshold Global", res)
#cv2.waitKey(0)  # espera pressionar alguma tecla
#cv2.destroyAllWindows()

