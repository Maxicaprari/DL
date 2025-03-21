# Modelo de código para comparar diferentes optimizadores en aprendizaje profundo
# Ejemplos para TensorFlow/Keras y PyTorch

# -------------------- TensorFlow/Keras --------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
import numpy as np
import matplotlib.pyplot as plt

# Función para crear un modelo simple de red neuronal convolucional
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Cargar y preparar los datos MNIST
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Normalizar y dar formato correcto
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # One-hot encoding para las etiquetas
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Función para entrenar y evaluar el modelo con diferentes optimizadores
def compare_optimizers():
    # Cargar datos
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Definir los optimizadores a comparar con sus configuraciones
    optimizers = {
        'SGD': SGD(learning_rate=0.01, momentum=0.9),
        'SGD+Nesterov': SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        'RMSprop': RMSprop(learning_rate=0.001, rho=0.9),
        'Adagrad': Adagrad(learning_rate=0.01),
        'Adadelta': Adadelta(learning_rate=1.0, rho=0.95),
        'Adam': Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        'Adamax': Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999),
        'Nadam': Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
    }
    
    # Histórico de precisión para cada optimizador
    history = {}
    
    # Ejecutar el entrenamiento para cada optimizador
    for name, optimizer in optimizers.items():
        print(f"Entrenando con {name}...")
        model = create_model()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar el modelo
        hist = model.fit(
            x_train, y_train,
            epochs=5,  # Usar más épocas en un escenario real
            batch_size=64,
            validation_split=0.2,
            verbose=1
        )
        
        # Evaluar el modelo
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"{name} - Precisión en prueba: {test_acc:.4f}")
        
        # Guardar historial
        history[name] = hist.history
    
    return history

# Visualizar los resultados
def plot_results(history):
    plt.figure(figsize=(12, 6))
    
    for name, hist in history.items():
        plt.plot(hist['val_accuracy'], label=name)
    
    plt.title('Comparación de Optimizadores - Precisión de Validación')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    plt.show()

# Ejecutar la comparación
# history = compare_optimizers()
# plot_results(history)

# -------------------- PyTorch --------------------
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Definir un modelo simple de CNN con PyTorch
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

# Función para cargar datos MNIST en PyTorch
def load_pytorch_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader

# Función para entrenar modelo con diferentes optimizadores en PyTorch
def train_pytorch_model(model, optimizer, criterion, trainloader, device, epochs=5):
    model.train()
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Resetear gradientes
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        print(f'Época {epoch+1}, Pérdida: {epoch_loss:.4f}, Precisión: {epoch_acc:.4f}')
    
    return history

# Función para comparar optimizadores en PyTorch
def compare_pytorch_optimizers():
    # Configuración
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = load_pytorch_data()
    criterion = nn.CrossEntropyLoss()
    
    # Definir los optimizadores a comparar
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=0.01, momentum=0.9),
        'SGD+Nesterov': lambda params: optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True),
        'Adam': lambda params: optim.Adam(params, lr=0.001, betas=(0.9, 0.999)),
        'RMSprop': lambda params: optim.RMSprop(params, lr=0.001, alpha=0.99),
        'Adagrad': lambda params: optim.Adagrad(params, lr=0.01),
        'Adadelta': lambda params: optim.Adadelta(params, lr=1.0, rho=0.9),
        'Adamax': lambda params: optim.Adamax(params, lr=0.002, betas=(0.9, 0.999))
    }
    
    results = {}
    
    for name, optimizer_fn in optimizers.items():
        print(f"\nEntrenando con {name}...")
        model = SimpleCNN().to(device)
        optimizer = optimizer_fn(model.parameters())
        history = train_pytorch_model(model, optimizer, criterion, trainloader, device)
        results[name] = history
    
    return results

# Ejemplo de uso
# pytorch_results = compare_pytorch_optimizers()

# -------------------- Explicación de los Optimizadores --------------------
"""
Resumen de los optimizadores implementados:

1. SGD (Stochastic Gradient Descent):
   - El optimizador más básico que actualiza los pesos en la dirección opuesta al gradiente
   - Parámetros clave: learning_rate (tasa de aprendizaje), momentum (impulso)
   - Ventajas: Simple, bien entendido
   - Desventajas: Convergencia lenta, puede quedar atrapado en mínimos locales

2. SGD con Nesterov Momentum:
   - Variante de SGD que calcula el gradiente después de aplicar el impulso
   - Parámetros clave: Los mismos que SGD, con nesterov=True
   - Ventajas: Mejor convergencia que SGD estándar, especialmente en problemas con alta curvatura
   - Desventajas: Sigue siendo sensible a la elección de la tasa de aprendizaje

3. RMSprop (Root Mean Square Propagation):
   - Mantiene un promedio móvil del cuadrado de los gradientes y divide el gradiente por la raíz de este promedio
   - Parámetros clave: learning_rate, rho (factor de decaimiento)
   - Ventajas: Adaptativo, buen rendimiento en problemas no estacionarios
   - Desventajas: Puede tener problemas en configuraciones específicas

4. Adagrad:
   - Adapta las tasas de aprendizaje a los parámetros utilizando un historial acumulado de gradientes
   - Parámetros clave: learning_rate inicial
   - Ventajas: Bueno para datos dispersos, ajusta automáticamente las tasas de aprendizaje
   - Desventajas: La tasa de aprendizaje disminuye demasiado con el tiempo, lo que puede detener el aprendizaje

5. Adadelta:
   - Extensión de Adagrad que busca resolver el problema de la disminución de la tasa de aprendizaje
   - Parámetros clave: learning_rate (generalmente 1.0), rho (factor de decaimiento)
   - Ventajas: No necesita configurar una tasa de aprendizaje inicial, adaptativo
   - Desventajas: Puede ser más lento en algunas configuraciones

6. Adam (Adaptive Moment Estimation):
   - Combina las ideas de RMSprop y momentum, manteniendo promedios móviles tanto del gradiente como del cuadrado del gradiente
   - Parámetros clave: learning_rate, beta_1 (decaimiento para el promedio del gradiente), beta_2 (decaimiento para el promedio del cuadrado del gradiente)
   - Ventajas: Rendimiento generalmente bueno, adaptativo, funciona bien en la mayoría de los problemas
   - Desventajas: Puede tener problemas de generalización en algunos casos

7. Adamax:
   - Variante de Adam basada en la norma infinita
   - Parámetros clave: Similares a Adam
   - Ventajas: Más estable que Adam en algunos casos
   - Desventajas: No siempre mejora sobre Adam

8. Nadam (Nesterov-accelerated Adaptive Moment Estimation):
   - Combina Adam con el momentum de Nesterov
   - Parámetros clave: Similares a Adam
   - Ventajas: Puede proporcionar mejor convergencia que Adam en algunos casos
   - Desventajas: Computacionalmente más costoso
"""
