# Questão 20

- Abrir uma imagem colorida, transformar para tom de cinza e aplique a técnica Crescimento de Regiões (Region Growing).
Para isto, inicialmente faça uma imagem com dimensões 320×240 no paint, onde o fundo da imagem seja branco e exista um
círculo preto no centro. Utilize algum ponto dentro do circulo preto como semente, onde você deve determinar este ponto
analisando imagem previamente. A regra de adesão do método deve ser: “Sempre que um vizinho da região possuir tom de
cinza menor que 127, deve-se agregar este vizinho à região”. Aplique o Crescimento de Regiões de forma iterativa, em
que o algoritmo irá estabilizar apenas quando a região parar de crescer.