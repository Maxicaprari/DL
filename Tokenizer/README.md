
## Jugando con Byte Pair Encoding (BPE)
En esta notebook se  explora el algoritmo Byte Pair Encoding (BPE), una técnica de tokenización utilizada en modelos de lenguaje. A través de un enfoque práctico, implementamos desde cero los pasos clave del algoritmo, incluyendo:

* Conversión de texto a Unicode mediante la función ord().
  
* Cálculo de la frecuencia de pares consecutivos en una secuencia de tokens.
  
* Fusión de los pares más frecuentes para construir un vocabulario optimizado.
  
* Iteración del proceso hasta alcanzar el tamaño de vocabulario deseado.
  
* El código permite visualizar el proceso de compresión progresiva del texto y comprender cómo BPE aprende representaciones eficientes de palabras a partir de sus componentes más frecuentes.



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



