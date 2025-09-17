import cv2
import numpy as np
import os

## uso -> segmentar_imagem("caminho_completo_arquivo")
## A função segmenta a imagem separando os objetos do fundo e realiza
def segmentar_imagem(image_path):
    image = cv2.imread(image_path)
    output_folder = "app/funcoes_utilitarias/segmentacao"
    os.makedirs(output_folder, exist_ok=True)

    # Carregar imagem e converter para cinza
    image = cv2.imread("pedras.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Filtro passa baixa, para suavizar imagem
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Threshold global
    _, thresh_global = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV)

    # Mais filtros para remover pequenos ruídos 
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh_global, cv2.MORPH_OPEN, kernel, iterations=2)

    # Background seguro 
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Foreground seguro usando Distance Transform 
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Região desconhecida 
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marcadores para Watershed 
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Aplicar Watershed 
    markers = cv2.watershed(image, markers)

    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255   # regiões dos objetos

    # Aplicar máscara na imagem original
    result = cv2.bitwise_and(image, image, mask=mask)

    # Salvar resultado
    output_path = os.path.join(output_folder, f"segmentado.png")
    cv2.imwrite(output_path, result)

    segmentar_imagem(output_path)
    os.remove(output_path) # Remove a Imagem segmentada
    return True



