Relatório descritivo A3 
Reconhecimento de Placas de Trânsito 
 

Integrantes:

João Pedro Mendes Silva 						  RA: 12127001 
Lucas Cabanillas Dávila de Medeiros 	RA: 12128382       
Pedro Henrique Ferreira						    RA: 12128647
Gabriel Carvalho de Almeida						RA:121127658 
Jonas Machado Silva 							    RA: 12126667 


Relatório do Trabalho Avaliativo, referente a construção do modelo de reconhecimento de placas de trânsito de acordo com a regulamentação do CONTRAN, solicitada pela montadora de veículos.



1. Coleta de Dados
1.1 Recuperação de Imagens 
Nesta fase nosso objetivo foi reunir um conjunto de dados robusto, representativo e variável com o foco em realizar o treinamento do modelo de reconhecimento de placas de trânsito. Para alcançar esse objetivo, usamos diversos sites da internet para recuperar imagens de placas de trânsito de fontes confiáveis na internet. 

 
1.1.1 Escolha de Fontes 
Selecionamos cuidadosamente fontes confiáveis que nos forneceram imagens de alta qualidade e diversidade, em diversos ângulos, iluminações e fundos diferentes. Utilizamos essa abordagem para garantir que nosso modelo seja treinado com um conjunto de dados abrangente, capaz de lidar com diferentes tipos de situações do mundo real. 
 
 
1.1.2 Gerenciamento de Dados 
Para evitar duplicatas e garantir a qualidade do conjunto de dados, revemos foto por foto da nossa base de dados e removemos imagens duplicadas e com qualidade ruim. Esse processo é crucial para garantir a diversidade e qualidade do conjunto de dados. 
 
 
1.2 Conclusões da Fase de Coleta de Dados e Observações
A coleta de dados é uma etapa fundamental para o sucesso do projeto, por isso essa etapa e a próxima foram fases que mais demandaram atenção e tempo do nosso grupo. A seleção cuidadosa de fontes é extremamente necessária para garantir que o conjunto de dados seja uma base sólida para o treinamento do modelo de reconhecimento de placas de trânsito. 

Separamos uma parte das nossas imagens (0.2%) para testarmos o nosso modelo em etapas futuras e também separamos uma parte (0.2%) para validação


2. Anotação das imagens
2.1 Anotação e Categorização das Imagens
As imagens foram anotadas usando a ferramenta Roboflow. Cada imagem foi devidamente anotada, criando notações para identificação das diferentes placas de trânsito contidas na regulamentação do CONTRAN.

3. Pré-processamento do Dataset
No arquivo treinamentoCNN.py, o pré-processamento e a equalização foi conduzida para preparar o conjunto de dados para treinamento.
3.1 Normalização e Limpeza
As imagens foram normalizadas para o formato padrão de 32x32 pixels. O pré-processamento incluiu a regularização dos arrays e um aumento de imagens usando o ImageData Generator.
Aproveitamos também para padronizar todas as imagens realizando variações como zoom, alteração de ângulo e rotação.
Usamos 32x32 pixels pois percebemos ao longo do trabalho que o tamanho das imagens influenciava diretamente no tempo gasto para o processamento das mesmas 
3.2 Equalização
Equalizamos as imagens alterando todas as elas para escala “grey”, dessa maneira estamos padronizando a luminosidade de todas, isso é de extrema importância pois em vez de termos uma escala de 0 a 255, agora temos uma escala de 0 e 1 e usaremos isso para construção da rede neural


4. Treinamento do modelo
4.1 Escolha do modelo e nossas observações
Para o início do trabalho tivemos que compreender os modelos de redes neurais e decidimos escolher o padrão LeNet-5 que consiste em uma rede neural convolucional utilizada principalmente para reconhecimento de imagens, sua arquitetura é definida em alto nível por 2 partes com um decodificador convolucional e um bloco de camadas conectadas.

Utilizamos o Tensor Flow em conjunto com a arquitetura LeNet-5 pois ele possui algumas funções que auxiliam no teste dos modelos de acordo com a nossa necessidade, no caso o reconhecimento das placas. Importamos as bibliotecas NumPy e Matplotlib que serão utilizadas para análise das imagens e definição das mesmas.


Criação do Projeto

Foi criado um ambiente em Python 3.8 no PyCharm, inserimos as pastas das imagens em um diretório para que pudéssemos acessá-las, criado 2 arquivos Python, um para testar o modelo e o outro de treinamento da CNN, no arquivo de treinamento da CNN inserimos os parâmetros, definimos todas as variáveis e que o modelo seria testado por 25 épocas, 607 interações e 12 imagens por interação para que possamos acompanhar todas as imagens do nosso dataset, separamos todas as imagens, fizemos pré processamento, regularizamos os arrays, fizemos um aumento de imagens com o ImageDataGenerator do Keras criando assim uma quantidade maior de imagens, alteramos as dimensões de todas as imagens, regularizamos as fotos dando zoom, rotacionando e alterando ângulo das mesmas. 

Foi feito também uma separação das imagens em duas partes, treinamento e teste, além de selecionar uma parte das imagens para a validação do treinamento. 






Após toda essa preparação criamos o modelo na arquitetura LeNet, com duas camadas, 5x5 e 2x2 e adicionamos um Dropout, utilizado para evitar o overfitting, caso comum onde o modelo treinado se ajusta muito bem ao treinamento mas em novos testes com novas imagens se mostra irregular. Criamos uma função para o histórico do treinamento que irá mostrar a perda e a acurácia em forma de gráfico.

Assim dando início ao treinamento, pelo tamanho do nosso dataSet o processo do treinamento levou mais tempo que o esperado, nos levando a cogitar a possibilidade de diminuir as épocas para que fosse possível entregar algum resultado, tivemos bastante dificuldade no início para ajustar quantas interações e a quantidade de imagens por interação para obtermos um resultado coerente de acordo com o tempo de treinamento.

5. Avaliação do modelo
5.1 Observações
Uma observação a ser feita é que o nosso dataSet não estava bem balanceado, ou seja para algumas placas tínhamos menos imagens do que para outras então, em alguns casos de teste o modelo identificava erroneamente a placa que tinha um mesmo padrão um exemplo seria a de velocidade máxima de 20 km/h onde possuíamos uma quantidade maior de imagens e assim o modelo identificava algumas vezes outras placas como sendo a de maior quantidade.
Após o treinamento podemos observar que no começo o modelo tem uma crescente exponencial na acurácia e chegando ao fim das épocas se estabiliza.
Isso se aplica ao contrário para o gráfico de perda onde no começo o modelo tem uma perda maior e obtém uma decrescente exponencial e se mantém nos níveis mais baixos nas épocas finais do treinamento.
Linha de treinamento representada em azul e a de validação em laranja.

![perda_precisao_img](https://github.com/joaopedromendesjava/pythonProjectRecognition/assets/90357555/f720d800-3d26-4ffa-afa5-f649352b56ec)

Com a função citada anteriormente, conseguimos obter o histórico do treinamento que nos mostra exatamente o que o gráfico representa, na imagem abaixo podemos observar que nas últimas épocas o valor da acurácia, representado pela variável val_accuracy, se manteve próximo ao 1 com poucas variações sempre oscilando entre 0.98 e 0.99 (98% a 99%).

![epoca_img](https://github.com/joaopedromendesjava/pythonProjectRecognition/assets/90357555/3a469591-cb09-4503-8f35-a4853e6382f9)

![20km_precisao_img](https://github.com/joaopedromendesjava/pythonProjectRecognition/assets/90357555/9dc7aedc-6741-49c0-ae46-dc2f45f6461b)

Aqui podemos observar um teste feito para validação com a placa de 20 Km/h.



6. Conclusões Finais
Nós do grupo usaremos esse último tópico para mostrarmos nossas considerações e todas as nossas conclusões finais a respeito deste trabalho.
No começo do trabalho durante a fase de coleta dos dados e montagem do data set, acabamos não nos atentando a nacionalidade das placas e de todas as imagens que havíamos coletado somente 8 tipos delas estavam em acordo com o CONTRAN, isso foi algo que nós percebemos somente durante a última fase de Avaliação do Modelo.
Outro erro que cometemos na etapa de coleta dos dados e que só percebemos na etapa de avaliação, foi não nos atentar ao balanceamento das imagens, captando mais imagens para alguns tipos de placas do que para outros e isso impactou diretamente em algumas questões.
A primeira foi que devido a isso a precisão do nosso modelo ficou bastante desbalanceada em algumas placas, por exemplo: obtivemos mais de 98% de precisão em diversas placas como a 80km/h, porém, em placas com menos imagens, como a de 20km/h obtivemos apenas 95% de precisão.
A segunda questão foi que esse balanceamento ocasionou em um “overfitting” do nosso modelo, usando os conhecimento adquiridos em sala de aula e com nossa própria pesquisa descobrimos que isso aconteceu pois muitas placas possuíam fundos parecidos e nosso modelo acabou levando em consideração esses fundos na hora do reconhecimento fazendo com que o reconhecimento ficasse mais difícil para placas que não possuíam esse tipo fundo.
Como informado na descrição do trabalho a escolha da arquitetura que fizemos foi da Lenet e obrigatoriamente a Yolo v8, porém percebemos diferenças muito grandes entre as duas e gostaríamos de pontuar algumas dessas diferenças. Primeiro porque o LeNet é uma rede neural convolucional simples (classificador de imagens), já o Yolo é próprio para detecção de objetos, o que seria mais indicado para o nosso trabalho. Outro ponto é que com a LeNet o trabalho ficou bem mais difícil e demorado, muito pelo fato de que nela tivemos que fazer todo o pré processamento das imagens, equalização, entre outras coisas que não foram necessárias no Yolo v8.

Só percebemos essas diferenças ao finalizar o trabalho com o Lenet, pois na hora de fazer com o Yolo v8, apenas pegamos o dataset do nosso próprio Roboflow, geramos uma espécie de API que é consumida pelo próprio Yolo onde ele acessa nossas imagens e começa a treinar automaticamente o modelo.
