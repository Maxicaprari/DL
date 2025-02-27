
## Jugando con Byte Pair Encoding (BPE)
En esta notebook se  explora el algoritmo Byte Pair Encoding (BPE), una técnica de tokenización utilizada en modelos de lenguaje. A través de un enfoque práctico, implementamos desde cero los pasos clave del algoritmo, incluyendo:

Conversión de texto a Unicode mediante la función ord().
  
Cálculo de la frecuencia de pares consecutivos en una secuencia de tokens.
  
Fusión de los pares más frecuentes para construir un vocabulario optimizado.
  
Iteración del proceso hasta alcanzar el tamaño de vocabulario deseado.
  
El código permite visualizar el proceso de compresión progresiva del texto y comprender cómo BPE aprende representaciones eficientes de palabras a partir de sus componentes más frecuentes.



## BASIC TOKENIZER 

El **BasicTokenizer** es una implementación desde cero de un tokenizador que convierte texto en secuencias de tokens (representados como enteros) y viceversa.
# Componentes clave
**Clase Tokenizer**
Clase base que define la interfaz y funcionalidad común:

Inicialización de vocabulario básico (256 bytes)
Métodos abstractos para entrenamiento, codificación y decodificación
Funciones para guardar/cargar modelos de tokenización

**Clase BasicTokenizer**
Implementación concreta que:

Procesa el texto de entrada convirtiéndolo a bytes UTF-8
Cuenta la frecuencia de pares consecutivos
Fusiona iterativamente los pares más frecuentes
Crea nuevos tokens para representar estas fusiones

**Funciones auxiliares**

get_stats(): Cuenta la frecuencia de pares consecutivos en una lista de tokens
merge(): Reemplaza ocurrencias de un par específico con un nuevo token

**Notas técnicas**

El algoritmo implementa una versión simplificada del Byte-Pair Encoding (BPE)
Utiliza codificación UTF-8 para manejar texto en múltiples idiomas
El vocabulario base consiste en 256 tokens correspondientes a todos los posibles valores de byte




## GPT4Tokenizer
Esta clase implementa un tokenizador compatible con el tokenizador utilizado por GPT-4, basado en el algoritmo cl100k_base de OpenAI.
**Características técnicas**

Extiende la clase RegexTokenizer utilizando el patrón de segmentación específico de GPT-4
Utiliza el algoritmo de tokenización BPE (Byte-Pair Encoding) con una configuración compatible con el tokenizador oficial
Incluye manejo especial para la permutación de bytes que existe en el tokenizador original de OpenAI
Incorpora tokens especiales utilizados por GPT-4 (<|endoftext|>, <|fim_prefix|>, etc.)

**Componentes clave**
Funciones auxiliares

bpe(): Función auxiliar que reconstruye el bosque de fusiones para un token dado
recover_merges(): Recupera los pares originales de fusiones a partir de los tokens ya fusionados

Constantes

GPT4_SPLIT_PATTERN: Expresión regular utilizada para segmentar el texto antes de la tokenización
GPT4_SPECIAL_TOKENS: Diccionario de tokens especiales utilizados por GPT-4

**Métodos principales**

__init__(): Inicializa el tokenizador usando tiktoken para obtener los rangos de fusión oficiales
_encode_chunk(): Codifica fragmentos de texto aplicando la permutación específica de bytes
decode(): Decodifica IDs de tokens a texto, invirtiendo la permutación de bytes
save_vocab(): Guarda el vocabulario en un formato legible para visualización

**Notas de implementación**

Este tokenizador no está diseñado para ser entrenado (train() no implementado)
No soporta guardar/cargar el modelo directamente (save()/load() no implementados)
Gestiona una peculiaridad histórica: los tokens correspondientes a bytes individuales están permutados en un orden diferente



