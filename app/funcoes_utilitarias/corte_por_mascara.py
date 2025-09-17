import cv2
import os

# uso -> separar_objetos("caminho_completo_arquivo")
def separar_objetos(imagem_path):
    # Verifica se o arquivo existe
    if not os.path.exists(imagem_path):
        print(f"Erro: O arquivo {imagem_path} não foi encontrado!")
        return
    
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Não foi possível carregar a imagem {imagem_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_folder = "app/funcoes_utilitarias/resultados"
    
    # Cria a pasta de resultados se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:  # Ignora objetos muito pequenos
            continue

        x, y, w, h = cv2.boundingRect(contour)
        objeto = img[y:y+h, x:x+w]
        
        arquivo_out = os.path.join(output_folder, f"objeto_{i+1}.png")
        cv2.imwrite(arquivo_out, objeto)
        print(f"Salvo: {arquivo_out}")