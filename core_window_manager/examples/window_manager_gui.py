#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI para gestionar ventanas usando advanced_window_manager.py
Interfaz gráfica compacta con botones por proceso
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from collections import defaultdict
from typing import Dict, List

# Importar nuestro gestor core
from core_window_manager import WindowManagerCore, WindowStrategy

class WindowManagerGUI:
    """Interfaz gráfica para el gestor avanzado de ventanas"""
    
    def __init__(self):
        self.window_manager = WindowManagerCore(debug_mode=False)
        self.root = tk.Tk()
        self.process_buttons = {}
        self.status_var = tk.StringVar()
        
        # Configurar callbacks para mostrar información en la GUI
        self._setup_window_manager_callbacks()
        
        self.setup_gui()
        self.refresh_processes()
    
    def setup_gui(self):
        """Configura la interfaz gráfica"""
        self.root.title("🎯 Gestor de Ventanas - Procesos")
        self.root.geometry("400x600")
        self.root.resizable(True, True)
        
        # Configurar estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal con scroll
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(
            main_frame, 
            text="🎯 Gestor de Ventanas por Proceso",
            font=("Segoe UI", 14, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Frame para botones de control
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Botón refresh
        refresh_btn = ttk.Button(
            control_frame,
            text="🔄 Actualizar Procesos",
            command=self.refresh_processes
        )
        refresh_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón File Explorer directo
        explorer_btn = ttk.Button(
            control_frame,
            text="📁 File Explorer",
            command=self.bring_file_explorer,
            style="Accent.TButton"
        )
        explorer_btn.pack(side=tk.LEFT)
        
        # Separador
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill=tk.X, pady=10)
        
        # Frame con scroll para los procesos
        self.canvas = tk.Canvas(main_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Frame de estado
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_var.set("🔄 Listo para usar")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack()
        
        # Bind mousewheel al canvas
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        self.root.bind("<MouseWheel>", _on_mousewheel)
    
    def _setup_window_manager_callbacks(self):
        """Configura callbacks del window manager para actualizar la GUI"""
        def on_operation_complete(result):
            success_count = result['success_count']
            total_count = result['total_count']
            success_rate = result['success_rate']
            
            if success_count == total_count:
                self.status_var.set(f"✅ {success_count}/{total_count} ventanas traídas exitosamente")
            elif success_rate >= 80:
                self.status_var.set(f"🎯 {success_count}/{total_count} ventanas traídas ({success_rate:.1f}% éxito)")
            elif success_count > 0:
                self.status_var.set(f"⚠️ {success_count}/{total_count} ventanas traídas ({success_rate:.1f}% éxito)")
            else:
                self.status_var.set(f"❌ No se pudieron traer las ventanas al frente")
        
        def on_error(error, **kwargs):
            if 'hwnd' in kwargs:
                self.status_var.set(f"❌ Error con ventana {kwargs['hwnd']}: {error}")
            else:
                self.status_var.set(f"❌ Error: {error}")
        
        self.window_manager.add_callback('on_operation_complete', on_operation_complete)
        self.window_manager.add_callback('on_error', on_error)
    
    def get_processes_with_windows(self) -> Dict[str, List[dict]]:
        """Obtiene procesos agrupados con sus ventanas"""
        return self.window_manager.get_windows_grouped_by_process()
    
    def refresh_processes(self):
        """Actualiza la lista de procesos y recrea los botones"""
        self.status_var.set("🔄 Escaneando procesos...")
        self.root.update()
        
        # Limpiar botones existentes
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Obtener procesos
        processes = self.get_processes_with_windows()
        
        if not processes:
            no_processes_label = ttk.Label(
                self.scrollable_frame,
                text="❌ No se encontraron procesos con ventanas",
                font=("Segoe UI", 10)
            )
            no_processes_label.pack(pady=20)
            self.status_var.set("❌ Sin procesos")
            return
        
        # Ordenar alfabéticamente
        sorted_processes = sorted(processes.items())
        
        # Crear botones para cada proceso
        for i, (process_name, windows) in enumerate(sorted_processes):
            window_count = len(windows)
            
            # Crear frame para cada botón
            btn_frame = ttk.Frame(self.scrollable_frame)
            btn_frame.pack(fill=tk.X, pady=2, padx=5)
            
            # Determinar estilo del botón según el proceso
            if process_name.lower() == 'explorer.exe':
                btn_style = "Accent.TButton"
                icon = "📁"
            elif 'chrome' in process_name.lower():
                icon = "🌐"
                btn_style = "TButton"
            elif 'notepad' in process_name.lower():
                icon = "📝"
                btn_style = "TButton"
            elif 'code' in process_name.lower() or 'vscode' in process_name.lower():
                icon = "💻"
                btn_style = "TButton"
            else:
                icon = "🔷"
                btn_style = "TButton"
            
            # Crear el botón
            btn_text = f"{icon} {process_name} ({window_count} ventana{'s' if window_count != 1 else ''})"
            
            process_btn = ttk.Button(
                btn_frame,
                text=btn_text,
                command=lambda p=process_name, w=windows: self.bring_process_windows(p, w),
                width=50
            )
            
            if btn_style == "Accent.TButton":
                try:
                    process_btn.configure(style=btn_style)
                except:
                    pass  # Si el estilo no existe, usar el predeterminado
            
            process_btn.pack(fill=tk.X)
        
        # Actualizar tamaño de la ventana
        self.update_window_size(len(sorted_processes))
        
        self.status_var.set(f"✅ {len(sorted_processes)} procesos encontrados")
    
    def update_window_size(self, process_count):
        """Actualiza el tamaño de la ventana según el número de procesos"""
        # Calcular altura basada en número de procesos
        base_height = 200  # Altura mínima
        button_height = 35  # Altura estimada por botón
        max_height = 800   # Altura máxima
        
        calculated_height = base_height + (process_count * button_height)
        final_height = min(calculated_height, max_height)
        
        self.root.geometry(f"420x{final_height}")
    
    def bring_process_windows(self, process_name: str, windows: List[dict]):
        """Trae todas las ventanas de un proceso específico al frente"""
        self.status_var.set(f"🔄 Trayendo ventanas de {process_name}...")
        self.root.update()
        
        def bring_windows_thread():
            try:
                if process_name.lower() == 'explorer.exe':
                    # Para explorer.exe, usar método específico de File Explorer
                    result = self.window_manager.bring_file_explorer_to_front(WindowStrategy.MINIMIZE_FIRST)
                    # El callback se encarga de actualizar el status
                else:
                    # Para otros procesos, usar método general
                    result = self.window_manager.bring_process_windows_to_front(process_name, WindowStrategy.MINIMIZE_FIRST)
                    # El callback se encarga de actualizar el status
                
            except Exception as e:
                self.status_var.set(f"❌ Error: {str(e)}")
        
        # Ejecutar en hilo separado para no bloquear la GUI
        thread = threading.Thread(target=bring_windows_thread, daemon=True)
        thread.start()
    
    def bring_file_explorer(self):
        """Método directo para traer ventanas del File Explorer"""
        self.status_var.set("🔄 Trayendo ventanas del File Explorer...")
        self.root.update()
        
        def bring_explorer_thread():
            try:
                result = self.window_manager.bring_file_explorer_to_front(WindowStrategy.MINIMIZE_FIRST)
                if result['total_count'] == 0:
                    self.status_var.set("❌ No hay ventanas del File Explorer abiertas")
                # El callback se encarga de actualizar el status para otros casos
            except Exception as e:
                self.status_var.set(f"❌ Error: {str(e)}")
        
        thread = threading.Thread(target=bring_explorer_thread, daemon=True)
        thread.start()
    
    def run(self):
        """Ejecuta la interfaz gráfica"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\n👋 Aplicación cerrada")

def main():
    """Función principal"""
    try:
        app = WindowManagerGUI()
        app.run()
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")
        input("Presiona Enter para salir...")

if __name__ == "__main__":
    main()
