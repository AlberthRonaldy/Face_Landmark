import cv2
import mediapipe as mp
from pathlib import Path
import time

# Função para configurar o diretório de captura, criando-o se não existir
def setup_capture_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

# Função para inicializar a malha facial do MediaPipe e retornar os objetos necessários
def initialize_face_mesh():
    drawing_utils = mp.solutions.drawing_utils
    face_mesh_module = mp.solutions.face_mesh
    face_mesh = face_mesh_module.FaceMesh(max_num_faces=2)  # Detectar no máximo 2 rostos
    drawing_specifications = drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
    return drawing_utils, face_mesh_module, face_mesh, drawing_specifications

# Função para capturar imagens de rostos
def capture_faces(user, capture_dir, capture_interval=1, num_images=10):
    cap = cv2.VideoCapture(0)  # Iniciar captura de vídeo da webcam
    if not cap.isOpened():
        print("Erro ao abrir a webcam")
        return

    drawing_utils, face_mesh_module, face_mesh, drawing_specifications = initialize_face_mesh()
    counter = 0  # Contador de imagens capturadas
    start_time = time.time()  # Tempo inicial para controle do intervalo de captura
    landmarks_updated = False  # Flag para atualização dos landmarks

    while counter < num_images:
        ret, frame = cap.read()  # Capturar frame da webcam
        if not ret:
            print("Erro ao capturar frame")
            break

        if not landmarks_updated:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Converter imagem para RGB
            results = face_mesh.process(img_rgb)  # Processar imagem para detectar landmarks faciais
            landmarks_updated = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Desenhar landmarks faciais no frame
                drawing_utils.draw_landmarks(frame, face_landmarks, face_mesh_module.FACEMESH_CONTOURS, drawing_specifications, drawing_specifications)

                # Obter coordenadas do rosto
                ih, iw, _ = frame.shape
                face_coords = [(int(lm.x * iw), int(lm.y * ih)) for lm in face_landmarks.landmark]
                x_coords = [coord[0] for coord in face_coords]
                y_coords = [coord[1] for coord in face_coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                # Verificar se o intervalo de captura foi atingido
                if time.time() - start_time >= capture_interval:
                    face_img = frame[y_min:y_max, x_min:x_max]  # Extrair região do rosto
                    rosto_path = Path(capture_dir) / f"{user}_face_{counter + 1}.jpg"
                    cv2.imwrite(str(rosto_path), face_img)  # Salvar imagem do rosto
                    print(f"Imagem {counter + 1} salva em {rosto_path}")
                    counter += 1
                    start_time = time.time()  # Reiniciar temporizador

        cv2.imshow('Webcam', frame)  # Mostrar frame com landmarks desenhados

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Parar se a tecla 'q' for pressionada
            break

    cap.release()  # Liberar captura de vídeo
    cv2.destroyAllWindows()  # Fechar todas as janelas abertas do OpenCV

def main():
    capture_dir = "training/images/captured_faces/"
    setup_capture_directory(capture_dir)

    while True:
        user = input("Nome do usuário (ou '0' para cancelar): ")
        if user.lower() == '0':
            break
        capture_faces(user, capture_dir)  # Capturar imagens do rosto do usuário

if __name__ == "__main__":
    main()
