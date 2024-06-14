from pathlib import Path
import cv2
import mediapipe as mp
import json

# Função para padronizar as imagens
def padronizar_imagem(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_LANCZOS4)  
    return img

# Carregar o modelo LBPH treinado
modelo_path = Path("catch_images/training/modelos/modelo_lbph.yml")
modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
modelo_lbph.read(str(modelo_path))

# Carregar o mapeamento de usuarios
user_map_path = Path("catch_images/training/modelos/user_map.json")
with user_map_path.open() as f:
    user_map = json.load(f)
reverse_user_map = {v: k for k, v in user_map.items()}  # Mapear IDs de usuarios para nomes

cap = cv2.VideoCapture(0)

# Configuração do MediaPipe
drawing_utils = mp.solutions.drawing_utils
face_mesh_module = mp.solutions.face_mesh
face_mesh = face_mesh_module.FaceMesh(max_num_faces=2)  # Detectar no máximo 2 rostos
drawing_specifications = drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

if not cap.isOpened():
    print("Erro ao abrir a webcam")
    exit()

while True:
    ret, frame = cap.read()  # Capturar um frame da webcam
    if not ret:
        print("Erro ao capturar frame")
        break

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converter frame para RGB
    results = face_mesh.process(imgRGB)  # Processar frame para detectar rostos

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # Determinar a região do rosto e fazer a predição
            ih, iw, ic = frame.shape
            face_coords = [(int(lm.x * iw), int(lm.y * ih)) for lm in faceLms.landmark]
            x_coords = [coord[0] for coord in face_coords]
            y_coords = [coord[1] for coord in face_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
           
            rosto = frame[y_min:y_max, x_min:x_max]  # Extrair região do rosto
            rosto_padronizado = padronizar_imagem(rosto)  # Padronizar a imagem do rosto para o modelo

            # Fazer a predição do usuario
            predicao, confianca = modelo_lbph.predict(rosto_padronizado)
            nome_user = reverse_user_map.get(predicao, "Desconhecido")  # Obter nome do usuario pela predição
           
            # Mostrar o nome do usuario no frame
            cv2.putText(frame, nome_user, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Desenhar retângulo ao redor do rosto

    cv2.imshow('Webcam', frame)  

    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()  
cv2.destroyAllWindows()  
