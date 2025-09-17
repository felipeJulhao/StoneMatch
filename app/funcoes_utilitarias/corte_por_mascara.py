import cv2
import os
import glob

# uso -> separar_objetos("caminho_completo_arquivo")
def separar_objetos(imagem_path):
    if not os.path.exists(imagem_path):
        print(f"Erro: O arquivo {imagem_path} não foi encontrado!")
        return
    
    output_folder = "./gallery/"
    
    # Cria a pasta de resultados se não existir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    arquivos_gallery = os.listdir(output_folder)    
    
    # Pega o maior número de objeto existente, para não sobrescrever imagens que já existem na pasta
    maior_numero = 0
    for arquivo in arquivos_gallery:
        if "_" in arquivo:
            try:
                numero = int(arquivo.split("_")[1].split(".")[0])
                maior_numero = max(maior_numero, numero)
            except:
                continue
    
    ultimo_arquivo = max(arquivos_gallery, default="Objeto_0.png")
    ultimo_nome_img = os.path.basename(ultimo_arquivo).split("_")[0]

    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Não foi possível carregar a imagem {imagem_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:  # Ignora objetos muito pequenos
            continue

        x, y, w, h = cv2.boundingRect(contour)
        objeto = img[y:y+h, x:x+w]
        
        arquivo_out = os.path.join(output_folder, f"{ultimo_nome_img}_{maior_numero + i + 1}.png")
        cv2.imwrite(arquivo_out, objeto)
        print(f"Salvo: {arquivo_out}")

separar_objetos('C:/Users/igorb/Desktop/CodeCon/app/funcoes_utilitarias/aaa.jpg')