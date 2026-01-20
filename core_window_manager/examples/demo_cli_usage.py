#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ejemplo de uso del Core Window Manager en aplicaciones CLI
Demuestra cómo usar la clase independiente de GUI
"""

from core_window_manager import WindowManagerCore, WindowStrategy, bring_explorer_to_front, bring_process_to_front

def demo_cli_usage():
    """Demostración del uso de la clase en aplicación CLI"""
    print("=== DEMO CLI DEL CORE WINDOW MANAGER ===\n")
    
    # Crear instancia del gestor con debug activado
    wm = WindowManagerCore(debug_mode=True)
    
    print("1. Obteniendo estadísticas del sistema:")
    stats = wm.get_statistics()
    print(f"   - Total de ventanas: {stats['total_windows']}")
    print(f"   - Total de procesos: {stats['total_processes']}")
    print(f"   - Ventanas del Explorer: {stats['explorer_windows']}")
    print(f"   - Ventanas minimizadas: {stats['minimized_windows']}")
    
    print("\n2. Procesos con más ventanas:")
    for proc, count in sorted(stats['processes_summary'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   - {proc}: {count} ventana(s)")
    
    print("\n3. Ejemplo de callbacks personalizados:")
    
    # Añadir callbacks personalizados
    def on_window_found(window):
        if 'notepad' in window['process_name'].lower():
            print(f"   🗒️ Encontrado Notepad: {window['title']}")
    
    def on_operation_complete(result):
        print(f"   🎉 Operación completada: {result['success_count']}/{result['total_count']} ventanas")
    
    wm.add_callback('on_window_found', on_window_found)
    wm.add_callback('on_operation_complete', on_operation_complete)
    
    # Buscar ventanas de Notepad (esto disparará los callbacks)
    notepad_windows = wm.get_windows_by_process('notepad.exe')
    
    if notepad_windows:
        print(f"\n4. Trayendo {len(notepad_windows)} ventana(s) de Notepad al frente...")
        result = wm.bring_process_windows_to_front('notepad.exe', WindowStrategy.MINIMIZE_FIRST)
    else:
        print("\n4. No hay ventanas de Notepad abiertas")
    
    print("\n=== FIN DE LA DEMO ===")

def demo_convenience_functions():
    """Demostración de funciones de conveniencia"""
    print("\n=== DEMO DE FUNCIONES DE CONVENIENCIA ===\n")
    
    print("1. Trayendo File Explorer al frente (función de conveniencia):")
    result = bring_explorer_to_front(debug=True)
    print(f"   Resultado: {result}")
    
    print("\n2. Trayendo Chrome al frente (función de conveniencia):")
    result = bring_process_to_front('chrome.exe', debug=True)
    print(f"   Resultado: {result}")

def interactive_demo():
    """Demo interactivo para que el usuario seleccione qué hacer"""
    wm = WindowManagerCore(debug_mode=True)
    
    while True:
        print("\n=== DEMO INTERACTIVO DEL CORE WINDOW MANAGER ===")
        print("1. Ver estadísticas del sistema")
        print("2. Listar todos los procesos con ventanas")
        print("3. Traer File Explorer al frente")
        print("4. Buscar y traer proceso específico al frente")
        print("5. Ver ventanas de un proceso específico")
        print("6. Salir")
        
        try:
            choice = input("\nSelecciona una opción (1-6): ").strip()
            
            if choice == "1":
                stats = wm.get_statistics()
                print(f"\n📊 ESTADÍSTICAS DEL SISTEMA:")
                print(f"   • Total de ventanas: {stats['total_windows']}")
                print(f"   • Total de procesos: {stats['total_processes']}")
                print(f"   • Ventanas del Explorer: {stats['explorer_windows']}")
                print(f"   • Ventanas minimizadas: {stats['minimized_windows']}")
                print(f"   • Ventanas maximizadas: {stats['maximized_windows']}")
                
            elif choice == "2":
                grouped = wm.get_windows_grouped_by_process()
                print(f"\n📁 PROCESOS CON VENTANAS ({len(grouped)}):")
                for proc, windows in sorted(grouped.items()):
                    print(f"   • {proc}: {len(windows)} ventana(s)")
                    
            elif choice == "3":
                print("\n🔄 Trayendo File Explorer al frente...")
                result = wm.bring_file_explorer_to_front(WindowStrategy.MINIMIZE_FIRST)
                if result['total_count'] == 0:
                    print("   ❌ No hay ventanas del File Explorer abiertas")
                else:
                    success_rate = result['success_rate']
                    print(f"   ✅ Resultado: {result['success_count']}/{result['total_count']} ventanas ({success_rate:.1f}% éxito)")
                    
            elif choice == "4":
                process_name = input("Nombre del proceso (ej: notepad.exe): ").strip()
                if process_name:
                    print(f"\n🔄 Trayendo ventanas de {process_name} al frente...")
                    result = wm.bring_process_windows_to_front(process_name, WindowStrategy.MINIMIZE_FIRST)
                    if result['total_count'] == 0:
                        print(f"   ❌ No se encontraron ventanas de {process_name}")
                    else:
                        success_rate = result['success_rate']
                        print(f"   ✅ Resultado: {result['success_count']}/{result['total_count']} ventanas ({success_rate:.1f}% éxito)")
                        
            elif choice == "5":
                process_name = input("Nombre del proceso (ej: notepad.exe): ").strip()
                if process_name:
                    windows = wm.get_windows_by_process(process_name)
                    if windows:
                        print(f"\n🔍 VENTANAS DE {process_name.upper()} ({len(windows)}):")
                        for i, window in enumerate(windows, 1):
                            status = "🔽" if window['is_minimized'] else "🔼" if window['is_maximized'] else "🔲"
                            print(f"   {i}. {status} {window['title']}")
                    else:
                        print(f"   ❌ No se encontraron ventanas de {process_name}")
                        
            elif choice == "6":
                print("👋 ¡Hasta luego!")
                break
                
            else:
                print("❌ Opción inválida. Por favor selecciona 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Selecciona el tipo de demo:")
    print("1. Demo automático")
    print("2. Demo de funciones de conveniencia") 
    print("3. Demo interactivo")
    
    choice = input("Opción (1-3): ").strip()
    
    if choice == "1":
        demo_cli_usage()
    elif choice == "2":
        demo_convenience_functions()
    elif choice == "3":
        interactive_demo()
    else:
        print("Opción inválida, ejecutando demo interactivo...")
        interactive_demo()
