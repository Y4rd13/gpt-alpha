# Construyendo un modelo similar al GPT desde cero con teoría detallada e implementación de código

## La era de los Transformadores

Los Transformadores están revolucionando el mundo de la inteligencia artificial. Esta poderosa arquitectura de red neuronal, introducida en 2017, se ha convertido rápidamente en la opción preferida para el procesamiento del lenguaje natural, la inteligencia artificial generativa y más. Con la ayuda de los transformadores, hemos visto la creación de productos de inteligencia artificial de vanguardia como BERT, GPT-x, DALL-E y AlphaFold, que están cambiando la forma en que interactuamos con el lenguaje y resolvemos problemas complejos como el plegamiento de proteínas. Y las posibilidades emocionantes no terminan ahí: los transformadores también están causando revuelo en el campo de la visión por computadora con el advenimiento de los Vision Transformers.

Antes de los Transformadores, el mundo de la IA estaba dominado por las redes neuronales convolucionales para la visión por computadora (VGGNet, AlexNet), las redes neuronales recurrentes para el procesamiento del lenguaje natural y problemas de secuenciación (LSTMs, GRUs) y las redes generativas adversarias (GANs) para la IA generativa. En 2017, las CNN y las RNN representaban el 10,2% y el 29,4%, respectivamente, de los documentos publicados sobre reconocimiento de patrones. Pero ya en 2021, los investigadores de Stanford llamaron a los transformadores "modelos fundamentales" porque los ven impulsando un cambio de paradigma en la IA.

Los Transformadores combinan algunos de los beneficios que tradicionalmente se ven en las redes neuronales convolucionales (CNN) y recurrentes (RNN).

Un beneficio que los Transformadores comparten con las CNN es la capacidad de procesar la entrada de manera jerárquica, con cada capa del modelo aprendiendo características cada vez más complejas. Esto permite que los Transformadores aprendan representaciones altamente estructuradas de los datos de entrada, similares a cómo las CNN aprenden representaciones estructuradas de imágenes.

Otro beneficio que los Transformadores comparten con las RNN es la capacidad de capturar dependencias entre elementos en una secuencia. A diferencia de las CNN, que procesan la entrada en una ventana de longitud fija, los Transformadores utilizan mecanismos de autoatención que permiten al modelo relacionar directamente diferentes elementos de entrada entre sí, independientemente de su distancia en la secuencia. Esto permite que los Transformadores capturen dependencias de largo alcance entre los elementos de entrada, lo que es particularmente importante para tareas como el procesamiento del lenguaje natural, donde es necesario considerar el contexto de una palabra para comprenderla con precisión.

Andrej Karpathy ha resumido el poder de los Transformadores como una computadora diferenciable general de propósito. Es muy poderoso en un pase hacia adelante, porque puede expresar una computación muy general. Los nodos almacenan vectores y estos nodos se miran entre sí y pueden ver lo que es importante para su cálculo en los otros nodos. En el paso hacia atrás es optimizable con las conexiones residuales, la normalización de capa y la atención softmax. Por lo tanto, se puede optimizar utilizando métodos de primer orden como el descenso de gradiente. Y por último, se puede ejecutar eficientemente en hardware moderno (GPUs), que prefiere mucho paralelismo.

El Transformador es una magnífica arquitectura de red neuronal porque es una computadora diferenciable de propósito general. es simultáneamente:

1. expresivo (en el pase hacia adelante)
2. optimizable (mediante retropropagación + descenso de gradiente)
3. eficiente (gráfico de cómputo de alto paralelismo)

Sorprendentemente, la arquitectura de Transformer es muy resistente con el tiempo. El Transformer que salió en 2017 es básicamente el mismo que el actual, excepto que reorganizas algunas de las normalizaciones de las capas. Los investigadores están haciendo que los conjuntos de datos de entrenamiento sean mucho más grandes, la evaluación mucho más grande, pero mantienen la arquitectura intacta, lo cual es notable.

## Attention is all you need

La nueva arquitectura se propuso en el artículo Attention Is All You Need, publicado en 2017. El artículo describía un nuevo tipo de arquitectura de red neuronal llamada Transformador, que se basa en mecanismos de autoatención a diferencia de las redes neuronales convolucionales o recurrentes tradicionales.

El artículo original describe una arquitectura de codificador-decodificador (encoder-decoder) para que Transformer resuelva el problema de la traducción.

En este artículo, sin embargo, nos centraremos en la arquitectura de solo decodificador (decoder), ya que construiremos un modelo similar a GPT, es decir, estamos resolviendo una tarea de completar oraciones en lugar de una tarea de traducción descrita en el documento.

Para la tarea de traducción, primero debe codificar el texto de entrada y tener una capa de atención-cruzada (cross-attention layer) con el texto traducido. En la finalización de oraciones solo es necesaria la parte del decodificador, así que centrémonos en ella.

## Encoder-decorder stack

Para empezar, recordemos la arquitectura codificador-decodificador (encoder-decoder). Los modelos de lenguaje podrían clasificarse aproximadamente en solo Encoder (BERT): solo tienen los bloques Encoders, solo Decoders (GPT-x, PaLM, OPT), solo los bloques decodificadores y Encoder-Decoder (BART, T0/T5). Esencialmente, la diferencia se reduce a la tarea que está tratando de resolver:

1. Las arquitecturas de solo Encoder (como BERT) resuelven la tarea de predecir la palabra enmascarada en una oración (masked word in a sentence). Entonces la atención puede ver todas las palabras antes y después de esta palabra enmascarada. Esto tiene un término "modelado de lenguaje enmascarado" ("masked language modelling"). La arquitectura del Encoder generalmente se usa para tareas de modelado de lenguaje que involucran Encoding de una secuencia de tokens de entrada y la producción de una representación de longitud fija (fixed-length), también conocida como Context Vector o [Embedding](#intro-a-embeddings), que resume la entrada. Este vector de contexto puede ser utilizado por una tarea posterior, como la traducción automática o el resumen de texto. El modelo BERT es un buen ejemplo.
2. Para las arquitecturas de Decoder y Encoder-Decoder, la tarea es predecir el siguiente token o conjunto de tokens, es decir, los tokens dados [0, ..., n-1] predicen n. Estas arquitecturas se utilizan para tareas de modelado de lenguaje que implican generar una secuencia de tokens de salida basados en un vector de contexto de entrada (input context vector).

En el artículo Attention is all you need, el Encoder y el Decoder están representados por seis capas (el número puede ser cualquiera, por ejemplo, en BERT hay 24 bloques Encoders).

Cada Encoder consta de dos capas: Self-Atention y Feed Forward Neural Network. Los datos de entrada del Encoder pasan primero por la capa de Self-Atention. Esta capa ayuda al Encoder a buscar otras palabras en la oración de entrada cuando codifica una palabra en particular. Los resultados de esta capa se envían a la capa completamente conectada (red neuronal de alimentación directa (Feed Forward Neural Network)).

A medida que el modelo procesa cada token (cada palabra en la secuencia de entrada), la capa de Self-Attention le permite buscar pistas en los otros tokens en la secuencia de entrada que pueden ayudar a mejorar la Encoding de esa palabra. En la capa totalmente conectada, la red neuronal de alimentación directa (Feed Forward Neural Network) no interactúa con otras palabras y, por lo tanto, se pueden ejecutar varias cadenas en paralelo a medida que pasan por esta capa. Esta es una de las principales características de Transformers, que les permite procesar todas las palabras del texto de entrada en paralelo.

[ver Encoder stack imagen 1](https://habrastorage.org/r/w1560/getpro/habr/upload_files/112/b04/0dc/112b040dc5607959986bc0d48d3c6124.png)

### <a name="intro-a-embeddings"></a>Breve introducción a los Embeddings (incrustaciones)

Una matriz de Embeddings en Python es simplemente una tabla de números que representa las palabras de entrada en un modelo de lenguaje natural. Cada fila de la matriz representa una palabra diferente y cada columna representa una dimensión de características diferentes.

Por ejemplo, si tenemos las palabras de entrada "hello" y "world", y queremos crear una matriz de embeddings con dos dimensiones de características, la matriz podría verse así:

```
    | dim1 | dim2 |
-------------------
hello| 0.2  | 0.1  |
-------------------
world| 0.3  | 0.5  |
```

En esta matriz, la fila "hello" representa el vector de embedding para la palabra "hello", y contiene dos valores que representan las dos dimensiones de características/features (en este caso, 0.2 y 0.1). De manera similar, la fila "world" representa el vector de embedding para la palabra "world", y contiene dos valores que representan las dos dimensiones de características/features (en este caso, 0.3 y 0.5).

Una analogía para entender mejor las matrices de embeddings podría ser pensar en ellas como una tabla de coordenadas que representa la ubicación de cada palabra en un espacio de características/features. Cada palabra se representa como un punto en este espacio de características/features y las coordenadas de cada punto se corresponden con los valores de su vector de embedding en cada dimensión de características/features.

Las características (o features) son las diferentes propiedades o aspectos de una palabra que se utilizan para representarla en un modelo de lenguaje natural. En el ejemplo anterior de la matriz de embeddings para las palabras "hello" y "world", las características o features podrían ser cualquier cosa que se utilice para representar estas palabras, como por ejemplo la longitud de la palabra, el número de vocales, la frecuencia con la que aparece en un corpus de texto, entre otros.

Por ejemplo, si utilizamos la longitud de la palabra y la frecuencia con la que aparece en un corpus de texto como nuestras características, la matriz de embeddings para las palabras "hello" y "world" podría verse así:

```
    | Longitud | Frecuencia |
-----------------------------
hello|    5    |   0.002    |
-----------------------------
world|    5    |   0.003    |
```

En esta matriz, la columna "Longitud" representa la longitud de cada palabra, mientras que la columna "Frecuencia" representa la frecuencia con la que cada palabra aparece en un corpus de texto. Los valores en cada celda representan la longitud o la frecuencia de cada palabra, normalizados o escalados de alguna manera para ser utilizados como features en el modelo.

Estas características se utilizan para representar cada palabra de entrada como un vector de números, es decir, como un punto en un espacio de características. En este caso, el vector de embedding para la palabra "hello" podría ser `(5, 0.002)`, mientras que el vector de embedding para la palabra "world" podría ser `(5, 0.003)`. Estos vectores de embeddings se utilizan como entrada para el modelo de lenguaje natural y se procesan para generar la salida deseada, como la predicción de la siguiente palabra en una oración o la clasificación de un texto.

Para crear embeddings utilizando "hello world" como input word en un modelo Transformer, podemos utilizar diversas librerías de procesamiento de lenguaje natural en Python, como por ejemplo spaCy, TensorFlow, PyTorch, Gensim, etc.

A continuación, te mostraré un ejemplo utilizando la librería spaCy para crear embeddings de las palabras "hello" y "world":

```python
import spacy

# Cargamos el modelo de lenguaje de spaCy en inglés
nlp = spacy.load('en_core_web_sm')

# Definimos las palabras de entrada
input_words = ['hello', 'world']

# Creamos un arreglo de vectores de embeddings para cada palabra de entrada
embeddings = []

# Generamos los embeddings para cada palabra utilizando el modelo de spaCy
for word in input_words:
    embedding = nlp(word).vector
    embeddings.append(embedding)

# Mostramos los embeddings generados para cada palabra de entrada
print(embeddings)


# El resultado de este código sería un arreglo de vectores de embeddings de dos elementos, cada uno con una longitud de 300 (que es la dimensión de características utilizada por el modelo de spaCy en inglés):
[array([-1.8676002 ,  0.74291   , ...,  1.2458001 , -0.30633   ],
      dtype=float32),
 array([ 0.16164 ,  0.22146 , ..., -0.047529, -0.16768 ],
      dtype=float32)]
```

## Algunas palabras más sobre la atención

De hecho, calculamos las "atenciones" inconscientemente mientras hablamos. Te preocupas constantemente por lo que se refiere "ella", "eso", "el" o "eso". Imagina una frase "Nunca he estado en la Antártida, pero eso está en mi lista de deseos". Sabemos que "eso" se refiere a "Antártida".

Los autores del artículo Attention Is All You Need introducen un concepto de vectores Consulta/Clave/Valor (Query/Key/Value vectors). Una buena analogía es un sistema de recuperación. Por ejemplo, cuando busca videos en Youtube, el motor de búsqueda mapeará su Consulta/Query (texto en la barra de búsqueda) contra un conjunto de Claves/Keys (título del video, descripción, etc.) asociadas con los videos en su base de datos, y le dará un resultado con los mejores videos coincidentes (Valores/Values).

En nuestro ejemplo, digamos que queremos calcular la autoatención (Self-Atenttion) de la palabra "eso" en la oración "Nunca he estado en la Antártida, pero eso está en mi lista de deseos". El proceso involucraría la creación de matrices Q, K y V para la oración de entrada, donde q (fila en la matriz Q) representa el vector de consulta (Q) para la palabra "eso", K representa la matriz clave (key matrix) para todas las palabras en la oración y V representa la matriz de valores (value matrix) para todas las palabras de la oración.

La autoatención (self-attention) para "eso" se calcularía entonces como el producto escalar de Q y K, dividido por la raíz cuadrada de la longitud de K, seguido de un producto escalar con V. Esto daría como resultado una suma ponderada de los valores en V, con los pesos determinados por la relevancia de cada palabra en la oración para la consulta "eso". En este caso, la autoatención por "eso" probablemente sería mayor por la palabra "Antártida", ya que están relacionados.

Este es el cálculo básico para la operación de autoatención en los modelos Transformer.

$softmax(\frac{QK^T}{\sqrt{d_k}})V = \sum_{i=1}^{n} softmax(\frac{qk_i^T}{\sqrt{d_k}})v_i$

Donde:

- Q: Matriz de consulta (misma que K y V en la autoatención)
- K: Matriz de clave (misma que Q y V en la autoatención)
- V: Matriz de valor (misma que Q y K en la autoatención)
- $d_k$: dimensión de K
- $softmax$: función de softmax que normaliza los pesos en la multiplicación de matrices en un rango de 0 a 1, de tal forma que la suma de todos los valores normalizados sea igual a 1. Esto asegura que los valores normalizados puedan utilizarse como pesos de atención en la operación de atención, ya que los pesos deben sumar 1 para que la suma ponderada de los valores de la matriz de valor V tenga sentido.

  $softmax(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$ / Donde $z_i$: Valor en el vector de entrada que se está normalizando y $n$: Longitud del vector de entrada

- $qk_i$: producto de punto entre la consulta Q y la clave K para la palabra i
- $v_i$: valor correspondiente a la palabra i en la matriz de valor V
- $n$: longitud de la oración
- T: en la expresión matemática K^T, la letra T indica la operación de transposición de la matriz K. La transposición de una matriz consiste en intercambiar las filas y las columnas de la matriz, es decir, si K es una matriz de tamaño (m x n), entonces la transpuesta de K, denotada por K^T, es una matriz de tamaño (n x m), donde los elementos de la columna i de K se convierten en los elementos de la fila i de K^T. Por lo tanto, en la operación QK^T, Q es una matriz de tamaño (n x d) y K es una matriz de tamaño (m x d), donde d es la dimensión de los vectores de entrada. Después de la transposición, K^T se convierte en una matriz de tamaño (d x m), lo que permite realizar la operación de producto de punto (o producto escalar/vectorial) entre Q y K^T para obtener una matriz de tamaño (n x m) que contiene la similitud entre las consultas en Q y las claves en K.

En codigo utilizando PyTorch:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definimos la oración de entrada
input_sentence = "Nunca he estado en la Antártida, pero eso está en mi lista de deseos"

# Convertimos la oración en una lista de palabras
input_words = input_sentence.split()

# Definimos la dimensión de los vectores de entrada
d = 128

# Creamos una instancia de nn.Embedding para generar los vectores de embedding
embedding = nn.Embedding(len(input_words), d)

# Generamos los vectores de embedding para cada palabra de entrada
input_vectors = embedding(torch.LongTensor(range(len(input_words))))

# Definimos las matrices de consulta, clave y valor
Q = input_vectors
K = input_vectors
V = input_vectors

# Calculamos la atención utilizando softmax
scores = torch.matmul(Q, K.t()) / torch.sqrt(torch.tensor(d))
attention_weights = F.softmax(scores, dim=1)
output_vectors = torch.matmul(attention_weights, V)

# Mostramos el resultado de la operación de atención
print(output_vectors)
tensor([[-0.9265,  1.0233, -0.2897,  ...,  0.2196,  1.6644, -0.6473],
        [ 1.0860,  0.2047, -0.3145,  ..., -0.5502, -0.6240, -0.4559],
        [-1.2225, -1.3630,  0.3759,  ..., -0.7909,  0.3532,  1.1266],
        ...,
        [ 0.1062,  1.1196, -1.0840,  ..., -0.1806, -0.1345, -1.3477],
        [-0.7551,  0.5022,  0.3807,  ..., -0.2712, -0.2242,  0.4842],
        [ 0.6968,  0.2501, -0.8668,  ..., -0.5559, -0.0033, -1.7683]],
       grad_fn=<MmBackward0>)
```

En codigo, con TensorFlow:

```python
import tensorflow as tf

# Definimos la oración de entrada
input_sentence = "Nunca he estado en la Antártida, pero eso está en mi lista de deseos"

# Convertimos la oración en una lista de palabras
input_words = input_sentence.split()

# Definimos la dimensión de los vectores de entrada
d = 128

# Creamos una instancia de tf.keras.layers.Embedding para generar los vectores de embedding
embedding = tf.keras.layers.Embedding(len(input_words), d)

# Generamos los vectores de embedding para cada palabra de entrada
input_vectors = embedding(tf.range(len(input_words)))

# Definimos las matrices de consulta, clave y valor
Q = input_vectors
K = input_vectors
V = input_vectors

# Calculamos la atención utilizando softmax
scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.constant(d, dtype=tf.float32))
attention_weights = tf.nn.softmax(scores, axis=1)
output_vectors = tf.matmul(attention_weights, V)

# Mostramos el resultado de la operación de atención
print(output_vectors)
tf.Tensor(
[[ 0.0033139   0.00507426  0.00547205 ...  0.00202452 -0.01837145
  -0.003456  ]
 [ 0.00333045  0.005109    0.00544784 ...  0.00200186 -0.01841413
  -0.00350031]
 [ 0.0033486   0.00509828  0.00545489 ...  0.00204012 -0.01839903
  -0.00349796]
 ...
 [ 0.00333579  0.0051146   0.00548671 ...  0.00202211 -0.01841822
  -0.00345222]
 [ 0.00336279  0.00509567  0.0054412  ...  0.00204145 -0.01842242
  -0.00347955]
 [ 0.00331269  0.00505507  0.00544502 ...  0.00203352 -0.01842565
  -0.00350634]], shape=(14, 128), dtype=float32)
```

## Como Query (Q), Key (K) and Value (V) vectors son generados?

Una respuesta simple es mediante la multiplicación de vector a matriz con tres matrices diferentes ($W^Q, W^K, W^V$).

La transformación se realiza mediante la multiplicación matricial de cada una de las matrices de entrada con sus correspondientes matrices de pesos, lo que da como resultado nuevas matrices de consulta $Q'$, clave $K'$ y valor $V'$ que se utilizan para calcular los pesos de atención.

La fórmula matemática para la transformación de las matrices de entrada es la siguiente:

$Q' = QW^Q$

$K' = KW^K$

$V' = VW^V$

Donde:

- $Q, K, V$: son las matrices de entrada
- $W^Q$, $W^K$, $W^V$: son las matrices de pesos
- $Q'$, $K'$, $V'$: son las matrices transformadas

El vector $X_{n}$ en este caso sería una palabra incrustada del token de entrada (o salida de la capa de atención anterior). Las matrices $W^Q, W^K, W^V$ son inicialmente matrices de peso inicializadas aleatoriamente. Su tarea es remodelar el vector de entrada en una forma más pequeña que represente los vectores de Query, Key y Value ($q, k, v$).

[ver imagen 2.](https://habrastorage.org/r/w1560/getpro/habr/upload_files/691/b5f/10a/691b5f10aff6ff1e26f224b7e0dab9b4.png)

Volvamos a un prompt más breve "Hola mundo" y veamos cómo lo procesará la arquitectura de Transformer. Tenemos dos palabras de entrada incrustadas (embedded input words): $x_{0}$ como Embedding (vector de contexto) de "Hola" y $x_{1}$ como Embedding de "mundo".

Y tenemos vectores ($q, k, v$) para cada una de estas dos palabras, es decir: ($q_{0}, k_{0}, v_{0}$) y ($q_{1}, k_{1}, v_{1}$). Estos vectores se generan multiplicando el vector de Embedding de la palabra y la matriz de peso $W$. Lo que se puede representar como:

$q_{0} = x_{0}W^{Q}$, $k_{0} = x_{0}W^{K}$, $v_{0} = x_{0}W^{V}$

$q_{1} = x_{1}W^{Q}$, $k_{1} = x_{1}W^{K}$, $v_{1} = x_{1}W^{V}$

Donde:

- $x_{0}$ y $x_{1}$ son los vectores de embedding para las palabras "Hola" y "mundo", respectivamente.
- $W^{Q}$, $W^{K}$ y $W^{V}$ son las matrices de pesos para la generación de los vectores de consulta, clave y valor.

Los vectores de consulta y clave (Query y Key) se utilizan para calcular los pesos de atención entre las dos palabras, mientras que los vectores de valor (Value) se utilizan para calcular la representación final de la entrada.

[ver imagen 3.](https://habrastorage.org/r/w1560/getpro/habr/upload_files/1f4/1c4/349/1f41c434987bcfa2bd5f23448c8aded7.png)

## Misma lógica pero en forma de matriz

Podemos presentar los mismos vectores de Query, Key y Value ($q, k, v$) en forma de matriz para todos los tokens de entrada (palabras) a la vez. Y ese es el poder de la arquitectura Transformer: podemos procesar todas las palabras de entrada a la vez.

$X[N × d_{model}]$ - es una matriz de Embeddings para las palabras de entrada (en nuestro caso, embeddings de las dos palabras "Hola" y "mundo").

$W[d_{model} × d_{k}]$ - matrices de peso inicializadas aleatoriamente.

$Q[N × d_{q}]$ - la matriz de Query vectors.

$K[N × d_{k}]$ - la matriz de Key vectors.

$V[N × d_{v}]$ - la matriz de Value vectors.

$N$ - número de tokens en una secuencia de entrada (input sequence).

$d_{v}$ - dimensión de los vectores de valor (Value).

$d_{k}$ = $d_{q}$ - dimensión de los vectores de consulta (Query) y clave (Key).

$d_{model}$ - dimensión de las hidden layers o de las dimensiones de los token embeddings.

$h$ = número de cabezas de multi-atención (number of heads of multi-head attention)

$Q = XW^{Q}$, $K = XW^{K}$, $V = XW^{V}$ - la transformación de las matrices de entrada.

Para la imagen 4, $x0$ es el vector de embedding para la palabra "Hola", $x1$ es el vector de embedding para la palabra "mundo".

En el articulo el $d_{model} = 512$ (en nuestra imagen 4 - 5 cuadrados), $d_{k}$ = $d_{q}$ = $d_{v}$ = $d_{model}$/h = 64 (3 cuadrados).

[ver imagen 4.](https://habrastorage.org/r/w1560/getpro/habr/upload_files/c52/1f9/5bd/c521f95bdb84708ee8098a6055809ba1.png)

## Ahora calculemos la matriz de Atención

Una vez que obtengamos nuestras matrices $Q$, $K$ y $V$, ahora podemos calcular la matriz para las puntuaciones de atención. La puntuación determina cuánto enfoque aplicamos en otras partes de la oración de entrada a medida que codificamos una palabra en una posición determinada.

[ver imagen 5](https://habrastorage.org/r/w1560/getpro/habr/upload_files/ff4/abe/48b/ff4abe48b19ae376f05fb4745c719537.png)

La fórmula matemática para calcular la matriz de atención es la siguiente:

$A_(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_{k}}})V = Z$

Donde:

- $Q$, $K$ y $V$ son matrices de consulta, clave y valor, respectivamente.
- $d_k$ es la dimensión de la matriz de clave $K$.
- $softmax$ es la función de normalización utilizada para obtener los pesos de atención normalizados.
- $Z$ es la matriz resultante de la operación de atención.
