
import cv2
import numpy as np
import os

def separar_objetos(imagem_path):

    img = cv2.imread(imagem_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 100:  # Ignora objetos muito pequenos
            continue
            
        # Pega a região do objeto
        x, y, w, h = cv2.boundingRect(contour)
        objeto = img[y:y+h, x:x+w]
        
        cv2.imwrite(f"objeto_{i+1}.png", objeto)
        print(f"Salvou objeto_{i+1}.png")

# uso -> separar_objetos("aaa.jpg")