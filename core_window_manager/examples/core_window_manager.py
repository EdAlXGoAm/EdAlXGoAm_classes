#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Window Manager - Clase independiente para gestión de ventanas
Diseñada para ser reutilizada en cualquier tipo de aplicación (GUI, CLI, web, etc.)
"""

import win32gui
import win32con
import win32process
import psutil
import time
from typing import List, Dict, Optional, Callable, Any
from collections import defaultdict
from enum import Enum

class WindowStrategy(Enum):
    """Estrategias para traer ventanas al frente"""
    SIMPLE = "simple"
    MINIMIZE_FIRST = "minimize_first"
    FORCE_FOREGROUND = "force_foreground"

# ==========================================
# FUNCIÓN GLOBAL PARA CALLBACK DE ENUMERACIÓN
# ==========================================
def _global_enum_windows_callback(hwnd, windows_list):
    """Callback global para enumerar ventanas (requerido por win32gui)"""
    try:
        if win32gui.IsWindowVisible(hwnd):
            window_title = win32gui.GetWindowText(hwnd)
            if window_title:  # Solo ventanas con título
                try:
                    _, pid = win32process.GetWindowThreadProcessId(hwnd)
                    process = psutil.Process(pid)
                    process_name = process.name()
                    exe_path = process.exe() if hasattr(process, 'exe') else 'N/A'
                except (psutil.NoSuchProcess, psutil.AccessDenied, Exception):
                    process_name = 'Unknown'
                    pid = 0
                    exe_path = 'N/A'
                
                try:
                    window_class = win32gui.GetClassName(hwnd)
                except:
                    window_class = 'Unknown'
                
                try:
                    is_minimized = win32gui.IsIconic(hwnd)
                    is_maximized = win32gui.IsZoomed(hwnd)
                except:
                    is_minimized = False
                    is_maximized = False
                
                window_info = {
                    'hwnd': hwnd,
                    'title': window_title,
                    'process_name': process_name,
                    'pid': pid,
                    'exe_path': exe_path,
                    'window_class': window_class,
                    'is_minimized': is_minimized,
                    'is_maximized': is_maximized
                }
                
                windows_list.append(window_info)
                
    except Exception as e:
        pass  # Ignorar errores para continuar enumeración
    return True

class WindowManagerCore:
    """
    Clase central para gestión de ventanas en Windows.
    Completamente independiente de GUI - puede usarse en aplicaciones CLI, web, etc.
    """
    
    def __init__(self, debug_mode: bool = False):
        """
        Inicializa el gestor de ventanas.
        
        Args:
            debug_mode: Si True, imprime información de depuración
        """
        self.debug_mode = debug_mode
        self._callbacks = {
            'on_window_found': [],
            'on_window_brought_to_front': [],
            'on_operation_complete': [],
            'on_error': []
        }
    
    # ==========================================
    # MÉTODOS DE CALLBACK PARA EXTENSIBILIDAD
    # ==========================================
    
    def add_callback(self, event_type: str, callback: Callable):
        """
        Añade un callback para eventos específicos.
        
        Args:
            event_type: Tipo de evento ('on_window_found', 'on_window_brought_to_front', etc.)
            callback: Función a llamar cuando ocurra el evento
        """
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    def _trigger_callback(self, event_type: str, **kwargs):
        """Dispara callbacks para un tipo de evento específico"""
        for callback in self._callbacks.get(event_type, []):
            try:
                callback(**kwargs)
            except Exception as e:
                if self.debug_mode:
                    print(f"Error en callback {event_type}: {e}")
    
    def _log(self, message: str, level: str = "info"):
        """Log interno (solo si debug_mode está activo)"""
        if self.debug_mode:
            prefix = {
                "info": "i",
                "success": "OK",
                "warning": "WARN",
                "error": "ERROR"
            }.get(level, "")
            print(f"{prefix} {message}")
    
    # ==========================================
    # MÉTODOS CORE DE ENUMERACIÓN DE VENTANAS
    # ==========================================
    
    def get_all_windows(self) -> List[Dict]:
        """
        Obtiene información de todas las ventanas visibles.
        
        Returns:
            Lista de diccionarios con información de ventanas
        """
        windows_list = []
        win32gui.EnumWindows(_global_enum_windows_callback, windows_list)
        
        # Disparar callbacks para cada ventana encontrada
        for window in windows_list:
            self._trigger_callback('on_window_found', window=window)
        
        self._log(f"Encontradas {len(windows_list)} ventanas")
        return windows_list
    
    def get_windows_by_process(self, process_name: str) -> List[Dict]:
        """
        Encuentra ventanas por nombre de proceso.
        
        Args:
            process_name: Nombre del proceso (ej: 'notepad.exe')
            
        Returns:
            Lista de ventanas del proceso especificado
        """
        all_windows = self.get_all_windows()
        matching_windows = [w for w in all_windows 
                          if process_name.lower() in w['process_name'].lower()]
        
        self._log(f"Encontradas {len(matching_windows)} ventanas de {process_name}")
        return matching_windows
    
    def get_windows_by_title(self, title_pattern: str, exact_match: bool = False) -> List[Dict]:
        """
        Encuentra ventanas por título.
        
        Args:
            title_pattern: Patrón del título a buscar
            exact_match: Si True, búsqueda exacta; si False, contiene
            
        Returns:
            Lista de ventanas que coinciden con el patrón
        """
        all_windows = self.get_all_windows()
        
        if exact_match:
            matching_windows = [w for w in all_windows if w['title'] == title_pattern]
        else:
            matching_windows = [w for w in all_windows 
                              if title_pattern.lower() in w['title'].lower()]
        
        self._log(f"Encontradas {len(matching_windows)} ventanas con título '{title_pattern}'")
        return matching_windows
    
    def get_windows_grouped_by_process(self) -> Dict[str, List[Dict]]:
        """
        Obtiene ventanas agrupadas por proceso.
        
        Returns:
            Diccionario donde las claves son nombres de proceso y valores son listas de ventanas
        """
        all_windows = self.get_all_windows()
        grouped = defaultdict(list)
        
        for window in all_windows:
            process_name = window['process_name']
            if process_name and process_name.lower() != 'dwm.exe':  # Excluir Desktop Window Manager
                grouped[process_name].append(window)
        
        return dict(grouped)
    
    # ==========================================
    # MÉTODOS ESPECÍFICOS PARA EXPLORER
    # ==========================================
    
    def get_file_explorer_windows(self) -> List[Dict]:
        """
        Encuentra SOLO ventanas reales del File Explorer.
        Aplica filtros estrictos para evitar ventanas del sistema.
        
        Returns:
            Lista de ventanas del File Explorer
        """
        all_windows = self.get_all_windows()
        explorer_windows = []
        
        for window in all_windows:
            # 1. Debe ser del proceso explorer.exe
            if window['process_name'].lower() != 'explorer.exe':
                continue
                
            # 2. Debe terminar específicamente con "- File Explorer"
            if not window['title'].endswith(' - File Explorer'):
                continue
                
            # 3. Excluir ventanas especiales del sistema
            if window['title'] in ['Program Manager', 'Desktop', 'Taskbar', 'Start', '']:
                continue
                
            # 4. Verificar que tenga las características de una ventana de explorador
            if window['window_class'] != 'CabinetWClass':  # Clase específica del File Explorer
                continue
                
            explorer_windows.append(window)
        
        self._log(f"Encontradas {len(explorer_windows)} ventanas del File Explorer")
        return explorer_windows
    
    # ==========================================
    # MÉTODOS DE MANIPULACIÓN DE VENTANAS
    # ==========================================
    
    def minimize_window(self, hwnd: int) -> bool:
        """
        Minimiza una ventana específica.
        
        Args:
            hwnd: Handle de la ventana
            
        Returns:
            True si se minimizó exitosamente
        """
        try:
            if not win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
                self._log(f"Ventana minimizada: {hwnd}")
                return True
            return True  # Ya estaba minimizada
        except Exception as e:
            self._log(f"Error minimizando ventana {hwnd}: {e}", "error")
            return False
    
    def restore_window(self, hwnd: int) -> bool:
        """
        Restaura una ventana desde estado minimizado.
        
        Args:
            hwnd: Handle de la ventana
            
        Returns:
            True si se restauró exitosamente
        """
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            self._log(f"Ventana restaurada: {hwnd}")
            return True
        except Exception as e:
            self._log(f"Error restaurando ventana {hwnd}: {e}", "error")
            return False
    
    def bring_window_to_front(self, hwnd: int, strategy: WindowStrategy = WindowStrategy.SIMPLE) -> bool:
        """
        Trae una ventana al frente usando la estrategia especificada.
        
        Args:
            hwnd: Handle de la ventana
            strategy: Estrategia a usar
            
        Returns:
            True si se trajo al frente exitosamente
        """
        try:
            # Verificar que la ventana sigue siendo válida
            if not win32gui.IsWindow(hwnd):
                self._log(f"Ventana {hwnd} ya no es válida", "warning")
                return False
            
            # Si está minimizada, restaurar primero
            if win32gui.IsIconic(hwnd):
                self.restore_window(hwnd)
                time.sleep(0.1)
            
            # Aplicar estrategia específica
            success = False
            
            if strategy == WindowStrategy.SIMPLE:
                success = self._bring_to_front_simple(hwnd)
            elif strategy == WindowStrategy.FORCE_FOREGROUND:
                success = self._bring_to_front_force(hwnd)
            else:  # DEFAULT: MINIMIZE_FIRST se maneja a nivel superior
                success = self._bring_to_front_simple(hwnd)
            
            if success:
                self._trigger_callback('on_window_brought_to_front', hwnd=hwnd)
                self._log(f"Ventana traída al frente: {hwnd}")
            
            return success
            
        except Exception as e:
            self._log(f"Error trayendo ventana {hwnd} al frente: {e}", "error")
            self._trigger_callback('on_error', error=str(e), hwnd=hwnd)
            return False
    
    def _bring_to_front_simple(self, hwnd: int) -> bool:
        """Estrategia simple para traer ventana al frente"""
        try:
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                win32gui.BringWindowToTop(hwnd)
                return True
            except Exception:
                try:
                    win32gui.SetActiveWindow(hwnd)
                    return True
                except Exception:
                    return False
    
    def _bring_to_front_force(self, hwnd: int) -> bool:
        """Estrategia forzada para traer ventana al frente"""
        try:
            # Método más agresivo
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, 0, 0, 0, 0,
                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            win32gui.SetForegroundWindow(hwnd)
            return True
        except Exception:
            return self._bring_to_front_simple(hwnd)
    
    # ==========================================
    # MÉTODOS DE ALTO NIVEL PARA OPERACIONES COMPLEJAS
    # ==========================================
    
    def bring_windows_to_front_batch(self, 
                                   windows: List[Dict], 
                                   strategy: WindowStrategy = WindowStrategy.MINIMIZE_FIRST,
                                   delay_between_windows: float = 0.25) -> Dict[str, Any]:
        """
        Trae múltiples ventanas al frente usando la estrategia especificada.
        
        Args:
            windows: Lista de ventanas a traer al frente
            strategy: Estrategia a usar
            delay_between_windows: Tiempo de espera entre ventanas
            
        Returns:
            Diccionario con estadísticas de la operación
        """
        if not windows:
            return {'success_count': 0, 'total_count': 0, 'success_rate': 0.0}
        
        total_count = len(windows)
        success_count = 0
        
        self._log(f"Iniciando operación batch con {total_count} ventanas usando estrategia {strategy.value}")
        
        # Aplicar estrategia MINIMIZE_FIRST
        if strategy == WindowStrategy.MINIMIZE_FIRST:
            self._log("Fase 1: Minimizando todas las ventanas")
            minimized_count = 0
            for window in windows:
                if self.minimize_window(window['hwnd']):
                    minimized_count += 1
                time.sleep(0.1)
            
            self._log(f"Minimizadas {minimized_count}/{total_count} ventanas")
            time.sleep(0.5)  # Pausa estratégica
            
            self._log("Fase 2: Restaurando y trayendo al frente")
        
        # Traer ventanas al frente
        for i, window in enumerate(windows, 1):
            self._log(f"Procesando ventana {i}/{total_count}: {window['title'][:50]}...")
            
            if self.bring_window_to_front(window['hwnd'], 
                                        WindowStrategy.SIMPLE if strategy == WindowStrategy.MINIMIZE_FIRST 
                                        else strategy):
                success_count += 1
            
            time.sleep(delay_between_windows)
        
        # Calcular estadísticas
        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        
        result = {
            'success_count': success_count,
            'total_count': total_count,
            'success_rate': success_rate,
            'strategy_used': strategy.value
        }
        
        self._log(f"Operación completada: {success_count}/{total_count} ventanas ({success_rate:.1f}% éxito)")
        self._trigger_callback('on_operation_complete', result=result)
        
        return result
    
    def bring_process_windows_to_front(self, 
                                     process_name: str, 
                                     strategy: WindowStrategy = WindowStrategy.MINIMIZE_FIRST) -> Dict[str, Any]:
        """
        Trae todas las ventanas de un proceso específico al frente.
        
        Args:
            process_name: Nombre del proceso
            strategy: Estrategia a usar
            
        Returns:
            Diccionario con estadísticas de la operación
        """
        windows = self.get_windows_by_process(process_name)
        return self.bring_windows_to_front_batch(windows, strategy)
    
    def bring_file_explorer_to_front(self, 
                                   strategy: WindowStrategy = WindowStrategy.MINIMIZE_FIRST) -> Dict[str, Any]:
        """
        Trae todas las ventanas del File Explorer al frente.
        
        Args:
            strategy: Estrategia a usar
            
        Returns:
            Diccionario con estadísticas de la operación
        """
        explorer_windows = self.get_file_explorer_windows()
        return self.bring_windows_to_front_batch(explorer_windows, strategy)
    
    # ==========================================
    # MÉTODOS DE UTILIDAD Y INFORMACIÓN
    # ==========================================
    
    def get_window_info(self, hwnd: int) -> Optional[Dict]:
        """
        Obtiene información detallada de una ventana específica.
        
        Args:
            hwnd: Handle de la ventana
            
        Returns:
            Diccionario con información de la ventana o None si no existe
        """
        try:
            if not win32gui.IsWindow(hwnd):
                return None
                
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            
            return {
                'hwnd': hwnd,
                'title': win32gui.GetWindowText(hwnd),
                'process_name': process.name(),
                'pid': pid,
                'exe_path': process.exe() if hasattr(process, 'exe') else 'N/A',
                'window_class': win32gui.GetClassName(hwnd),
                'is_visible': win32gui.IsWindowVisible(hwnd),
                'is_minimized': win32gui.IsIconic(hwnd),
                'is_maximized': win32gui.IsZoomed(hwnd),
                'rect': win32gui.GetWindowRect(hwnd)
            }
        except Exception as e:
            self._log(f"Error obteniendo información de ventana {hwnd}: {e}", "error")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del sistema de ventanas.
        
        Returns:
            Diccionario con estadísticas
        """
        all_windows = self.get_all_windows()
        grouped = self.get_windows_grouped_by_process()
        
        return {
            'total_windows': len(all_windows),
            'total_processes': len(grouped),
            'explorer_windows': len(self.get_file_explorer_windows()),
            'minimized_windows': len([w for w in all_windows if w['is_minimized']]),
            'maximized_windows': len([w for w in all_windows if w['is_maximized']]),
            'processes_summary': {name: len(windows) for name, windows in grouped.items()}
        }

# ==========================================
# FUNCIONES DE CONVENIENCIA PARA USO RÁPIDO
# ==========================================

def create_window_manager(debug: bool = False) -> WindowManagerCore:
    """Función de conveniencia para crear una instancia del gestor"""
    return WindowManagerCore(debug_mode=debug)

def bring_explorer_to_front(debug: bool = False) -> Dict[str, Any]:
    """Función de conveniencia para traer el File Explorer al frente"""
    wm = create_window_manager(debug)
    return wm.bring_file_explorer_to_front()

def bring_process_to_front(process_name: str, debug: bool = False) -> Dict[str, Any]:
    """Función de conveniencia para traer un proceso al frente"""
    wm = create_window_manager(debug)
    return wm.bring_process_windows_to_front(process_name)
