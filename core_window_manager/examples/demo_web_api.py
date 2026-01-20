#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de uso del Core Window Manager en aplicaciones web/API
Demuestra cómo usar la clase para crear APIs REST
"""

from flask import Flask, jsonify, request
from core_window_manager import WindowManagerCore, WindowStrategy
import threading
import time

app = Flask(__name__)

# Instancia global del window manager
wm = WindowManagerCore(debug_mode=False)

# Storage para resultados de operaciones asíncronas
operation_results = {}

def async_operation(operation_id, operation_func):
    """Ejecuta una operación de forma asíncrona y guarda el resultado"""
    try:
        result = operation_func()
        operation_results[operation_id] = {
            'status': 'completed',
            'result': result,
            'error': None
        }
    except Exception as e:
        operation_results[operation_id] = {
            'status': 'error',
            'result': None,
            'error': str(e)
        }

@app.route('/api/windows', methods=['GET'])
def get_all_windows():
    """Obtiene todas las ventanas del sistema"""
    try:
        windows = wm.get_all_windows()
        return jsonify({
            'success': True,
            'data': windows,
            'count': len(windows)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/windows/process/<process_name>', methods=['GET'])
def get_windows_by_process(process_name):
    """Obtiene ventanas de un proceso específico"""
    try:
        windows = wm.get_windows_by_process(process_name)
        return jsonify({
            'success': True,
            'process_name': process_name,
            'data': windows,
            'count': len(windows)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/windows/title/<title>', methods=['GET'])
def get_windows_by_title(title):
    """Obtiene ventanas por título"""
    try:
        exact_match = request.args.get('exact', 'false').lower() == 'true'
        windows = wm.get_windows_by_title(title, exact_match=exact_match)
        return jsonify({
            'success': True,
            'title_pattern': title,
            'exact_match': exact_match,
            'data': windows,
            'count': len(windows)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/processes', methods=['GET'])
def get_processes_grouped():
    """Obtiene procesos agrupados con sus ventanas"""
    try:
        grouped = wm.get_windows_grouped_by_process()
        return jsonify({
            'success': True,
            'data': grouped,
            'process_count': len(grouped)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Obtiene estadísticas del sistema"""
    try:
        stats = wm.get_statistics()
        return jsonify({
            'success': True,
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/windows/bring-to-front/process/<process_name>', methods=['POST'])
def bring_process_to_front(process_name):
    """Trae ventanas de un proceso al frente"""
    try:
        # Obtener estrategia del request
        strategy_name = request.json.get('strategy', 'minimize_first') if request.json else 'minimize_first'
        strategy = WindowStrategy(strategy_name)
        
        # Ejecutar operación
        result = wm.bring_process_windows_to_front(process_name, strategy)
        
        return jsonify({
            'success': True,
            'process_name': process_name,
            'strategy': strategy.value,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/windows/bring-to-front/explorer', methods=['POST'])
def bring_explorer_to_front():
    """Trae ventanas del File Explorer al frente"""
    try:
        # Obtener estrategia del request
        strategy_name = request.json.get('strategy', 'minimize_first') if request.json else 'minimize_first'
        strategy = WindowStrategy(strategy_name)
        
        # Ejecutar operación
        result = wm.bring_file_explorer_to_front(strategy)
        
        return jsonify({
            'success': True,
            'strategy': strategy.value,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/windows/bring-to-front/async/process/<process_name>', methods=['POST'])
def bring_process_to_front_async(process_name):
    """Trae ventanas de un proceso al frente de forma asíncrona"""
    try:
        # Generar ID único para la operación
        operation_id = f"{process_name}_{int(time.time() * 1000)}"
        
        # Obtener estrategia del request
        strategy_name = request.json.get('strategy', 'minimize_first') if request.json else 'minimize_first'
        strategy = WindowStrategy(strategy_name)
        
        # Marcar operación como iniciada
        operation_results[operation_id] = {
            'status': 'running',
            'result': None,
            'error': None
        }
        
        # Ejecutar en hilo separado
        operation_func = lambda: wm.bring_process_windows_to_front(process_name, strategy)
        thread = threading.Thread(target=async_operation, args=(operation_id, operation_func))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'operation_id': operation_id,
            'status': 'running',
            'check_url': f'/api/operations/{operation_id}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/operations/<operation_id>', methods=['GET'])
def get_operation_status(operation_id):
    """Obtiene el estado de una operación asíncrona"""
    if operation_id not in operation_results:
        return jsonify({
            'success': False,
            'error': 'Operación no encontrada'
        }), 404
    
    operation = operation_results[operation_id]
    return jsonify({
        'success': True,
        'operation_id': operation_id,
        'status': operation['status'],
        'result': operation['result'],
        'error': operation['error']
    })

@app.route('/api/window/<int:hwnd>/info', methods=['GET'])
def get_window_info(hwnd):
    """Obtiene información detallada de una ventana específica"""
    try:
        window_info = wm.get_window_info(hwnd)
        if window_info:
            return jsonify({
                'success': True,
                'data': window_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Ventana no encontrada'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/window/<int:hwnd>/minimize', methods=['POST'])
def minimize_window(hwnd):
    """Minimiza una ventana específica"""
    try:
        success = wm.minimize_window(hwnd)
        return jsonify({
            'success': success
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/window/<int:hwnd>/restore', methods=['POST'])
def restore_window(hwnd):
    """Restaura una ventana específica"""
    try:
        success = wm.restore_window(hwnd)
        return jsonify({
            'success': success
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/window/<int:hwnd>/bring-to-front', methods=['POST'])
def bring_window_to_front(hwnd):
    """Trae una ventana específica al frente"""
    try:
        # Obtener estrategia del request
        strategy_name = request.json.get('strategy', 'simple') if request.json else 'simple'
        strategy = WindowStrategy(strategy_name)
        
        success = wm.bring_window_to_front(hwnd, strategy)
        return jsonify({
            'success': success,
            'strategy': strategy.value
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Página de inicio con documentación de la API"""
    return '''
    <h1>🎯 Window Manager API</h1>
    <p>API REST para gestión de ventanas usando Core Window Manager</p>
    
    <h2>📋 Endpoints Disponibles:</h2>
    <ul>
        <li><strong>GET /api/windows</strong> - Obtener todas las ventanas</li>
        <li><strong>GET /api/windows/process/&lt;process_name&gt;</strong> - Ventanas por proceso</li>
        <li><strong>GET /api/windows/title/&lt;title&gt;</strong> - Ventanas por título</li>
        <li><strong>GET /api/processes</strong> - Procesos agrupados</li>
        <li><strong>GET /api/statistics</strong> - Estadísticas del sistema</li>
        <li><strong>POST /api/windows/bring-to-front/process/&lt;process_name&gt;</strong> - Traer proceso al frente</li>
        <li><strong>POST /api/windows/bring-to-front/explorer</strong> - Traer File Explorer al frente</li>
        <li><strong>POST /api/windows/bring-to-front/async/process/&lt;process_name&gt;</strong> - Traer proceso al frente (asíncrono)</li>
        <li><strong>GET /api/operations/&lt;operation_id&gt;</strong> - Estado de operación asíncrona</li>
        <li><strong>GET /api/window/&lt;hwnd&gt;/info</strong> - Info de ventana específica</li>
        <li><strong>POST /api/window/&lt;hwnd&gt;/minimize</strong> - Minimizar ventana</li>
        <li><strong>POST /api/window/&lt;hwnd&gt;/restore</strong> - Restaurar ventana</li>
        <li><strong>POST /api/window/&lt;hwnd&gt;/bring-to-front</strong> - Traer ventana al frente</li>
    </ul>
    
    <h2>🔧 Estrategias Disponibles:</h2>
    <ul>
        <li><strong>simple</strong> - Método simple</li>
        <li><strong>minimize_first</strong> - Minimizar primero (recomendado)</li>
        <li><strong>force_foreground</strong> - Forzar al frente</li>
    </ul>
    
    <h2>💡 Ejemplo de uso:</h2>
    <pre>
    # Obtener todas las ventanas
    curl http://localhost:5000/api/windows
    
    # Traer Notepad al frente
    curl -X POST http://localhost:5000/api/windows/bring-to-front/process/notepad.exe \
         -H "Content-Type: application/json" \
         -d '{"strategy": "minimize_first"}'
    
    # Traer File Explorer al frente
    curl -X POST http://localhost:5000/api/windows/bring-to-front/explorer
    </pre>
    '''

if __name__ == "__main__":
    print("🚀 Iniciando Window Manager API...")
    print("📋 Documentación disponible en: http://localhost:5000/")
    print("🔧 Ejemplo: curl http://localhost:5000/api/statistics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
