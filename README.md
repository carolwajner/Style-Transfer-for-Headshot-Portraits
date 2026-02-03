# Style Transfer for Headshot Portraits

Este projeto implementa uma pipeline avançada de transferência de estilo e morfologia facial para retratos (headshots). O sistema combina técnicas clássicas de processamento de imagens e visão computacional para transformar uma imagem de entrada, permitindo que ela herde a geometria e a iluminação de uma imagem de referência.

## 🚀 Funcionalidades

O projeto utiliza uma pipeline multi-estágio para garantir resultados naturais:

* **Detecção de Landmarks**: Utiliza o `dlib` para identificar 68 pontos faciais.
* **Aprimoramento de Geometria**: Adição sintética de pontos para a testa e detecção automática da linha do cabelo via análise de cor HSV.
* **Warping Facial (Beier-Neely)**: Implementação do algoritmo de morphing baseado em campos de vetores e linhas de influência para alinhar as características faciais.
* **Transferência de Contraste Local**: Decomposição em *Laplacian Stacks* para transferir texturas e iluminação através de mapas de ganho robustos.
* **Segmentação Inteligente**: Remoção e extração de background utilizando U2-Net (`rembg`) e técnicas de inpainting para limpeza de cena.

## 🛠️ Tecnologias e Dependências

* **Python 3.x**
* **OpenCV**: Processamento de imagem e fluxo óptico (Farneback).
* **Dlib**: Localização de landmarks faciais.
* **NumPy**: Operações matriciais e cálculos de energia.
* **Rembg**: Segmentação de background baseada em redes neurais.

## 📋 Como usar

1.  **Instale as dependências**:
    ```bash
    pip3 install opencv-python dlib numpy rembg
    ```

2.  **Preparação**:
    Certifique-se de ter o arquivo `shape_predictor_68_face_landmarks.dat` no diretório raiz do projeto.

3.  **Execução**:
    ```bash
    python3 main.py <caminho_da_imagem_entrada> <caminho_da_imagem_exemplo>
    ```

## 🔬 Detalhes do Algoritmo

### Morfologia Baseada em Linhas
O sistema utiliza o algoritmo de **Beier-Neely**, que define a deformação através de pares de linhas correspondentes em vez de apenas pontos isolados. Isso permite um controle mais suave sobre a transição de características como o contorno da mandíbula e o formato dos olhos.



### Pirâmides de Frequência
A transferência de estilo não é apenas uma sobreposição de cores. A imagem é decomposta em várias bandas de frequência. O ganho é calculado localmente para cada nível da pilha laplaciana, garantindo que detalhes de alta frequência (como poros e fios de cabelo) sejam preservados ou transferidos conforme a necessidade.

---

## 📚 Referências e Créditos
* *Feature-Based Image Metamorphosis* (Beier & Neely, 1992).
* [Facial Landmarks with dlib and OpenCV](https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/).
* Projeto desenvolvido como parte dos estudos em Ciência da Computação na **UFRGS**.
