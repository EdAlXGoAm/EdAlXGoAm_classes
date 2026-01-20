# 🎯 Core Window Manager - Documentación

## 📋 Descripción

El **Core Window Manager** es una clase independiente y reutilizable para la gestión de ventanas en Windows. Diseñada para ser utilizada en cualquier tipo de aplicación (CLI, GUI, web APIs, etc.) sin dependencias específicas de frameworks de interfaz gráfica.

## 🚀 Características Principales

- **Independiente de GUI**: No depende de tkinter, PyQt u otras librerías de interfaz gráfica
- **Sistema de Callbacks**: Permite extensibilidad y notificaciones personalizadas
- **Múltiples Estrategias**: Diferentes enfoques para traer ventanas al frente
- **API Completa**: Operaciones básicas y avanzadas para gestión de ventanas
- **Estadísticas del Sistema**: Información detallada sobre ventanas y procesos
- **Filtros Especializados**: Métodos específicos para File Explorer y otros procesos

## 🔧 Instalación y Dependencias

```bash
pip install pywin32 psutil
```

## 💡 Uso Básico

### Importar la Clase

```python
from core_window_manager import WindowManagerCore, WindowStrategy
```

### Crear una Instancia

```python
# Crear gestor con debug desactivado
wm = WindowManagerCore(debug_mode=False)

# Crear gestor con debug activado (útil para desarrollo)
wm = WindowManagerCore(debug_mode=True)
```

### Operaciones Básicas

```python
# Obtener todas las ventanas
ventanas = wm.get_all_windows()

# Obtener ventanas de un proceso específico
ventanas_notepad = wm.get_windows_by_process('notepad.exe')

# Obtener ventanas del File Explorer
ventanas_explorer = wm.get_file_explorer_windows()

# Obtener procesos agrupados
procesos = wm.get_windows_grouped_by_process()

# Obtener estadísticas del sistema
stats = wm.get_statistics()
```

### Traer Ventanas al Frente

```python
# Traer todas las ventanas de un proceso al frente
resultado = wm.bring_process_windows_to_front('notepad.exe', WindowStrategy.MINIMIZE_FIRST)

# Traer ventanas del File Explorer al frente
resultado = wm.bring_file_explorer_to_front(WindowStrategy.MINIMIZE_FIRST)

# Traer ventanas específicas al frente
ventanas = wm.get_windows_by_process('chrome.exe')
resultado = wm.bring_windows_to_front_batch(ventanas, WindowStrategy.MINIMIZE_FIRST)
```

## 🎯 Estrategias Disponibles

### `WindowStrategy.SIMPLE`
- Método básico y rápido
- Ideal para casos simples
- Menor tiempo de ejecución

### `WindowStrategy.MINIMIZE_FIRST` (Recomendado)
- Minimiza todas las ventanas primero
- Luego las restaura y trae al frente
- Mayor tasa de éxito

### `WindowStrategy.FORCE_FOREGROUND`
- Método más agresivo
- Para casos donde otros métodos fallan
- Puede ser más intrusivo

## 🔄 Sistema de Callbacks

### Callbacks Disponibles

- `on_window_found`: Se dispara cuando se encuentra una ventana
- `on_window_brought_to_front`: Se dispara cuando una ventana se trae al frente
- `on_operation_complete`: Se dispara cuando una operación batch se completa
- `on_error`: Se dispara cuando ocurre un error

### Ejemplo de Uso de Callbacks

```python
def mi_callback_ventana_encontrada(window):
    print(f"Encontrada ventana: {window['title']}")

def mi_callback_operacion_completa(result):
    print(f"Operación completada: {result['success_count']}/{result['total_count']}")

# Añadir callbacks
wm.add_callback('on_window_found', mi_callback_ventana_encontrada)
wm.add_callback('on_operation_complete', mi_callback_operacion_completa)

# Los callbacks se ejecutarán automáticamente durante las operaciones
resultado = wm.bring_process_windows_to_front('notepad.exe')
```

## 📊 Información de Ventanas

Cada ventana devuelta contiene la siguiente información:

```python
{
    'hwnd': 123456,                          # Handle de la ventana
    'title': 'Documento - Notepad',         # Título de la ventana
    'process_name': 'notepad.exe',          # Nombre del proceso
    'pid': 1234,                            # ID del proceso
    'exe_path': 'C:\\Windows\\notepad.exe', # Ruta del ejecutable
    'window_class': 'Notepad',              # Clase de la ventana
    'is_minimized': False,                  # ¿Está minimizada?
    'is_maximized': False                   # ¿Está maximizada?
}
```

## 🌐 Ejemplos de Uso en Diferentes Aplicaciones

### 1. Aplicación CLI

```python
from core_window_manager import WindowManagerCore, WindowStrategy

def app_cli():
    wm = WindowManagerCore(debug_mode=True)
    
    print("Procesos disponibles:")
    procesos = wm.get_windows_grouped_by_process()
    for nombre, ventanas in procesos.items():
        print(f"  - {nombre}: {len(ventanas)} ventana(s)")
    
    proceso = input("¿Qué proceso quieres traer al frente? ")
    resultado = wm.bring_process_windows_to_front(proceso)
    print(f"Resultado: {resultado}")

if __name__ == "__main__":
    app_cli()
```

### 2. Aplicación GUI (tkinter)

```python
import tkinter as tk
from core_window_manager import WindowManagerCore, WindowStrategy

class MiGUI:
    def __init__(self):
        self.wm = WindowManagerCore()
        self.root = tk.Tk()
        
        # Configurar callbacks
        self.wm.add_callback('on_operation_complete', self.mostrar_resultado)
        
        # Crear interfaz
        btn = tk.Button(self.root, text="Traer Explorer", 
                       command=self.traer_explorer)
        btn.pack()
    
    def mostrar_resultado(self, result):
        print(f"Operación completada: {result}")
    
    def traer_explorer(self):
        self.wm.bring_file_explorer_to_front(WindowStrategy.MINIMIZE_FIRST)
    
    def run(self):
        self.root.mainloop()

app = MiGUI()
app.run()
```

### 3. API Web (Flask)

```python
from flask import Flask, jsonify
from core_window_manager import WindowManagerCore

app = Flask(__name__)
wm = WindowManagerCore()

@app.route('/api/windows')
def get_windows():
    ventanas = wm.get_all_windows()
    return jsonify({'ventanas': ventanas, 'count': len(ventanas)})

@app.route('/api/bring-explorer', methods=['POST'])
def bring_explorer():
    resultado = wm.bring_file_explorer_to_front()
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(port=5000)
```

## 🔧 Funciones de Conveniencia

Para casos de uso simples, se incluyen funciones de conveniencia:

```python
from core_window_manager import bring_explorer_to_front, bring_process_to_front

# Traer File Explorer al frente (función rápida)
resultado = bring_explorer_to_front(debug=True)

# Traer proceso específico al frente (función rápida)
resultado = bring_process_to_front('notepad.exe', debug=True)
```

## 📋 API Completa

### Métodos de Enumeración
- `get_all_windows()`: Todas las ventanas visibles
- `get_windows_by_process(process_name)`: Ventanas por proceso
- `get_windows_by_title(title, exact_match=False)`: Ventanas por título
- `get_windows_grouped_by_process()`: Ventanas agrupadas por proceso
- `get_file_explorer_windows()`: Solo ventanas del File Explorer

### Métodos de Manipulación
- `minimize_window(hwnd)`: Minimizar ventana específica
- `restore_window(hwnd)`: Restaurar ventana específica
- `bring_window_to_front(hwnd, strategy)`: Traer ventana específica al frente

### Métodos de Alto Nivel
- `bring_windows_to_front_batch(windows, strategy)`: Operación batch
- `bring_process_windows_to_front(process_name, strategy)`: Por proceso
- `bring_file_explorer_to_front(strategy)`: File Explorer específico

### Métodos de Información
- `get_window_info(hwnd)`: Información detallada de ventana
- `get_statistics()`: Estadísticas del sistema

### Gestión de Callbacks
- `add_callback(event_type, callback)`: Añadir callback

## 🎯 Casos de Uso Recomendados

1. **Aplicaciones de Productividad**: Cambio rápido entre aplicaciones
2. **Herramientas de Desarrollo**: Gestión de ventanas de IDEs y navegadores
3. **Sistemas de Automatización**: Scripts para organización de escritorio
4. **APIs de Escritorio**: Servicios web para control remoto
5. **Aplicaciones de Monitoreo**: Supervisión de aplicaciones activas

## ⚠️ Consideraciones Importantes

- Requiere Windows (usa win32gui API)
- Algunos antivirus pueden requerir permisos especiales
- Las operaciones de traer al frente pueden ser bloqueadas por políticas del sistema
- Usar `debug_mode=True` durante desarrollo para mejor troubleshooting

## 🚀 Rendimiento

- La enumeración de ventanas es rápida (~10-50ms típico)
- Las operaciones batch usan delays configurable para estabilidad
- Los callbacks permiten operaciones asíncronas sin bloquear la aplicación

## 📄 Ejemplo Completo

Ver los archivos de ejemplo incluidos:
- `demo_cli_usage.py`: Aplicación CLI completa
- `demo_web_api.py`: API REST completa
- `window_manager_gui.py`: Integración con tkinter
