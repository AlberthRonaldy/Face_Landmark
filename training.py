from pathlib import Path  
import cv2 
import numpy as np  
import shutil  
import json 
from sklearn.metrics import accuracy_score 
from concurrent.futures import ThreadPoolExecutor

# Função para padronizar as imagens
def padronizar_imagem(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (200, 200), interpolation=cv2.INTER_LANCZOS4) 

# Função para organizar imagens em treino e teste
# src_dir = diretório de origem
def organizar_imagens(src_dir, train_dir, test_dir, split_number=5):
    src_dir, train_dir, test_dir = Path(src_dir), Path(train_dir), Path(test_dir)  # Converter para objetos Path
    train_dir.mkdir(parents=True, exist_ok=True) 
    test_dir.mkdir(parents=True, exist_ok=True) 

    list_faces_captured = [f for f in src_dir.iterdir() if f.is_file()] 

    for arq in list_faces_captured:  
        partes = arq.stem.split('_')  
        if len(partes) >= 3 and partes[2].isdigit(): 
            user, numero = partes[0], int(partes[2])  
            dest_dir = train_dir if numero <= split_number else test_dir  # Determinar o diretório de destino com base no número
            shutil.copyfile(arq, dest_dir / arq.name)  # Copiar o arquivo para o diretório de destino

# Função para carregar imagens e preparar dados
def preparar_dados(img_dir):
    img_dir = Path(img_dir)
    list_faces = [f for f in img_dir.iterdir() if f.is_file()] 

    dados, users = [], []

    # Utilizando ThreadPoolExecutor para paralelizar o processamento das imagens
    with ThreadPoolExecutor() as executor:
        # Mapear a função padronizar_imagem para todas as imagens listadas
        imgs = list(executor.map(padronizar_imagem, list_faces))

    for img, arq in zip(imgs, list_faces):  
        partes = arq.stem.split('_')  
        if len(partes) >= 2:  
            dados.append(img)  # Adicionar imagem aos dados
            users.append(partes[0])  # Adicionar nome do usuário à lista de usuários

    return np.array(dados), users 

# Diretórios de imagens
faces_path_captured = "catch_images/training/images/captured_faces/"
faces_path_train = "catch_images/training/images/train/"
faces_path_test = "catch_images/training/images/test/"

# Organizar imagens capturadas em treino e teste
organizar_imagens(faces_path_captured, faces_path_train, faces_path_test)

# Preparar dados de treinamento
dados_treinamento, users_train = preparar_dados(faces_path_train)

# Mapear nomes dos usuarios para inteiros
unique_users = list(set(users_train))  
user_map = {name: idx + 1 for idx, name in enumerate(unique_users)}

# Criar diretório de modelos se não existir
modelo_dir = Path("catch_images/training/modelos")
modelo_dir.mkdir(parents=True, exist_ok=True)

# Salvar o mapeamento em um arquivo JSON
with (modelo_dir / "user_map.json").open('w') as f:
    json.dump(user_map, f)

# Converter os nomes dos usuarios para inteiros
users_train = np.array([user_map[name] for name in users_train], dtype=np.int32)

# Treinar modelo LBPH
modelo_lbph = cv2.face.LBPHFaceRecognizer_create()
modelo_lbph.train(dados_treinamento, users_train)

# Salvar o modelo treinado
modelo_lbph.save(str(modelo_dir / "modelo_lbph.yml"))

# Preparar dados de teste
dados_teste, users_test = preparar_dados(faces_path_test)
users_test = np.array([user_map[name] for name in users_test], dtype=np.int32)

# Avaliar modelo LBPH
y_pred_lbph = [modelo_lbph.predict(item)[0] for item in dados_teste]
acuracia_lbph = accuracy_score(users_test, y_pred_lbph)

print("Acurácia do modelo LBPH:", acuracia_lbph)  
