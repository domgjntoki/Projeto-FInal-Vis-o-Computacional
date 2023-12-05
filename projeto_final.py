import cv2 as cv
import numpy as np
from PIL import Image
import os

def smooth_path(path, window_size=15, order=8):
    # Remove outliers based on Euclidean distance
    for i in range(len(path) - 1, 0, -1):
        if np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) > 50:
            del path[i]


    smoothed_path = []
    for i in range(len(path)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(path), i + window_size // 2 + 1)
        window_points = path[window_start:window_end]
        avg_x = sum(point[0] for point in window_points) / len(window_points)
        avg_y = sum(point[1] for point in window_points) / len(window_points)
        smoothed_path.append((avg_x, avg_y))

    
    # Change the smoothed path coordinates to integers, using int and round
    smoothed_path = [(int(round(point[0])), int(round(point[1]))) for point in smoothed_path]

    return smoothed_path

# Função para processar cada frame do vídeo
def process_frame(frame, background_remover, path):
    result = frame.copy()
    image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Definir o intervalo de cores para detecção do vermelho
    lower_red = np.array([165, 30, 20])
    upper_red = np.array([189, 255, 255])

    # Criar máscara para a área vermelha
    mask = cv.inRange(image, lower_red, upper_red)
    red_area = cv.bitwise_and(result, result, mask=mask)

    # Aplicar desfoque para reduzir ruídos
    frame = cv.GaussianBlur(frame, (5, 5), 0)

    # Remover o fundo e aplicar operações morfológicas para obter contornos
    mask = background_remover.apply(red_area)
    _, mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Encontrar contornos
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Encontrar o maior contorno válido e desenhar uma elipse sobre ele
    max_cnt = [0, 0]
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 800 and len(cnt) >= 5:
            if area > max_cnt[0]:
                max_cnt = [area, cnt]

    if max_cnt[0] != 0:
        elipse = cv.fitEllipse(max_cnt[1])
        path.append([int(elipse[0][0]), int(elipse[0][1])])
        cv.ellipse(frame, elipse, (0, 255, 0), 2)

    return frame, path

# Função principal para seguir o robô Husky no vídeo
def follow_husky(input_video, name, outpath):
    cap = cv.VideoCapture(input_video)
    background_remover = cv.createBackgroundSubtractorKNN(detectShadows=True)

    frames = []
    path = []

    # Processamento de cada frame do vídeo
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Processar o frame atual
        frame, path = process_frame(frame, background_remover, path)
        frames.append(frame)

    cap.release()

    # Salvar o vídeo com a identificação da caixa vermelha
    save_video(frames, outpath, name)

    # Gerar a imagem da trajetória do marcador
    generate_path_image(path, outpath, name)

# Função para salvar o vídeo identificado com a caixa vermelha
def save_video(frames, outpath, name):
    if frames:
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(outpath + name + '.avi', fourcc, 20.0, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()

# Função para gerar a imagem da trajetória do marcador
def generate_path_image(path, outpath, name):
    if path:
        # Suavizar a trajetória
        path = smooth_path(path)

        # Criar uma imagem em branco para desenhar a trajetória
        img = np.ones((580, 940, 3), dtype=np.uint8) * 255  # Ajuste o tamanho conforme necessário

        #img = cv.cvtColor(frame_img, cv.COLOR_BGR2RGB)
        
        step = 3
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        c_inc = 0

        # Desenhar a trajetória com setas na imagem em branco
        for i in range(10, len(path) - 10, step):
            u = path[i][0] - path[i - step][0]
            v = path[i][1] - path[i - step][1]
            orientation = np.arctan2(v, u)

            # Verificar se o tamanho do vetor é aceitável
            if np.sqrt(u ** 2 + v ** 2) > 50:
                continue
            
            # The color: The closer to the end of the path, the more red it is, The closer to the beginning, the more blue it is
            color = (int(255 * (i / len(path))), 0, int(255 * (1 - i / len(path))))
            cv.arrowedLine(img, (path[i - step][0], path[i - step][1]), (path[i][0], path[i][1]),
                        color, 1, tipLength=0.8)

            c_inc += 1

        # Salvar a imagem da trajetória
        img_path = outpath + name + '.png'
        if Image.fromarray(img).save(img_path):
            print(f"Could not save image in path {img_path}")

# Chamadas para seguir o robô Husky nos vídeos fornecidos
file_path = os.path.dirname(os.path.abspath(__file__))
content = rf'{file_path}/content/'
follow_husky(f'{content}/Video1_husky.mp4', 'husky1', content)
follow_husky(f'{content}/video2_husky.mp4', 'husky2', content)
