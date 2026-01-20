"""
Calco de imagen (Image Trace) - Convierte imágenes rasterizadas a vectores SVG/DXF.
Similar a la función Image Trace de Adobe Illustrator.

Usa OpenCV para detección de contornos rápida.
Soporta exportación a SVG y DXF (AutoCAD).
"""
import os
import io
import threading
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageGrab, ImageTk
import numpy as np

# Intentar importar OpenCV para mejor rendimiento
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    print("Advertencia: OpenCV no está instalado. El procesamiento será más lento.")
    print("Instalar con: pip install opencv-python")

# Importar ezdxf para generar archivos DXF válidos
try:
    import ezdxf
    from ezdxf import units
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    print("Advertencia: ezdxf no está instalado. No se podrán generar archivos DXF.")
    print("Instalar con: pip install ezdxf")

# Importar potrace para vectorización profesional
try:
    import potrace
    HAS_POTRACE = True
except ImportError:
    HAS_POTRACE = False
    print("Advertencia: potrace no está instalado. Usando algoritmo básico.")
    print("Instalar con: pip install potracer")


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


class TraceMode(Enum):
    """Modos de calco disponibles."""
    BLACK_WHITE = "Blanco y Negro"
    GRAYSCALE = "Escala de grises"
    COLOR = "Color"
    OUTLINE = "Solo contornos"


class OutputFormat(Enum):
    """Formatos de salida disponibles."""
    SVG = "SVG"
    DXF = "DXF"


@dataclass
class TraceSettings:
    """Configuración para el calco de imagen."""
    mode: TraceMode = TraceMode.BLACK_WHITE
    threshold: int = 128  # 0-255, umbral para blanco/negro
    color_count: int = 6  # Número de colores para modo color
    detail_level: float = 1.0  # 0.1 a 10, más alto = más detalle
    smoothness: float = 1.0  # 0.0 a 2.0, suavizado de curvas
    corner_threshold: float = 1.0  # 0.0 a 1.34, umbral para esquinas
    ignore_white: bool = True  # Ignorar áreas blancas
    invert: bool = False  # Invertir colores
    # Dimensiones DXF en cm
    use_physical_dims: bool = True  # Usar dimensiones físicas para DXF
    dim_mode: str = "width"  # "width" o "height" - qué dimensión se especifica
    dim_value_cm: float = 10.0  # Valor de la dimensión especificada en cm


def is_image_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def ensure_unique_output_path(base_path: Path) -> Path:
    """Si el archivo existe, agrega sufijos _1, _2, ... hasta que no exista."""
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def image_to_grayscale_array(
    img: Image.Image,
    settings: TraceSettings,
    apply_smoothing: bool = True,
) -> np.ndarray:
    """Convierte una imagen PIL a array numpy en escala de grises."""
    if img.mode != "L":
        gray = img.convert("L")
    else:
        gray = img.copy()
    
    # Aplicar suavizado si se requiere
    if apply_smoothing and settings.smoothness > 0:
        blur_radius = settings.smoothness * 2
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Invertir si se requiere
    if settings.invert:
        gray = ImageOps.invert(gray)
    
    return np.array(gray, dtype=np.uint8)


def threshold_image(gray_array: np.ndarray, threshold: int) -> np.ndarray:
    """Aplica umbral a una imagen en escala de grises."""
    if HAS_OPENCV:
        _, binary = cv2.threshold(gray_array, threshold, 255, cv2.THRESH_BINARY)
        return binary
    else:
        return np.where(gray_array > threshold, 255, 0).astype(np.uint8)


def preprocess_for_trace(img: Image.Image, settings: TraceSettings) -> Tuple[Image.Image, float, float]:
    """
    Preprocesa la imagen para mejorar el trazado y aumentar el costo de procesamiento.
    Devuelve la imagen procesada y los factores de escala (x, y).
    """
    original_width, original_height = img.size

    # Escalar según nivel de detalle: entre 2x y 4x
    scale = 1.5 + (settings.detail_level * 0.5)
    scale = max(2.0, min(4.0, scale))

    work = img.convert("L") if img.mode != "L" else img.copy()
    new_width = max(1, int(round(original_width * scale)))
    new_height = max(1, int(round(original_height * scale)))

    if new_width != original_width or new_height != original_height:
        work = work.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if HAS_OPENCV:
        arr = np.array(work)
        sigma = int(20 + (settings.smoothness * 40))
        arr = cv2.bilateralFilter(arr, d=9, sigmaColor=sigma, sigmaSpace=sigma)

        if settings.detail_level <= 2:
            ksize = 3
        elif settings.detail_level <= 4:
            ksize = 5
        else:
            ksize = 7
        arr = cv2.medianBlur(arr, ksize)
        work = Image.fromarray(arr)
    else:
        if settings.detail_level <= 2:
            ksize = 3
        elif settings.detail_level <= 4:
            ksize = 5
        else:
            ksize = 7
        work = work.filter(ImageFilter.MedianFilter(size=ksize))

        blur_radius = 1.0 + settings.smoothness * 1.5
        work = work.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if settings.smoothness > 0:
        work = ImageOps.autocontrast(work, cutoff=1)

    scale_x = work.width / original_width
    scale_y = work.height / original_height
    return work, scale_x, scale_y


# ============================================================================
# FUNCIONES DE POTRACE - Vectorización profesional
# ============================================================================

def trace_with_potrace(
    img: Image.Image,
    settings: TraceSettings,
) -> Tuple[str, List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """
    Vectoriza una imagen usando Potrace (algoritmo profesional).
    
    Args:
        img: Imagen PIL
        settings: Configuración de trazado
        
    Returns:
        Tuple de (contenido SVG, lista de splines para DXF, polilíneas para preview)
    """
    if not HAS_POTRACE:
        raise RuntimeError("Potrace no está instalado. Instalar con: pip install potracer")

    original_width, original_height = img.size
    pre_img, scale_x, scale_y = preprocess_for_trace(img, settings)
    
    # Convertir a escala de grises y aplicar umbral
    gray_array = image_to_grayscale_array(pre_img, settings, apply_smoothing=False)
    binary = threshold_image(gray_array, settings.threshold)
    
    # Potrace espera valores 0-255:
    # - Internamente aplica: data > (255 * blacklevel) -> luego invert()
    # - Así que valores bajos (negro/oscuro) serán trazados
    # - Pasamos binary directamente (0=negro, 255=blanco)
    
    # Crear bitmap de Potrace
    bmp = potrace.Bitmap(binary)
    
    # Trazar con parámetros optimizados
    # turdsize: ignora manchas menores a este tamaño (elimina ruido)
    # alphamax: 0=solo esquinas, 1.34=curvas muy suaves (estilo Illustrator)
    # opticurve: optimiza las curvas Bezier
    # opttolerance: tolerancia de optimización
    turdsize = max(2, int(10 / settings.detail_level))
    
    # Vincular smoothness (0.0-2.0) a alphamax (0.5-1.34)
    # smoothness=0.0 -> alphamax=0.5 (más esquinas, polígonos)
    # smoothness=1.0 -> alphamax=0.92 (balance)
    # smoothness=2.0 -> alphamax=1.34 (máximas curvas, estilo Illustrator)
    alphamax = 0.5 + (settings.smoothness / 2.0) * 0.84
    
    path = bmp.trace(
        turdsize=turdsize,
        alphamax=alphamax,
        opticurve=True,
        opttolerance=0.2
    )
    
    height, width = binary.shape
    scale_down_x = original_width / width
    scale_down_y = original_height / height
    
    # Convertir a SVG
    svg_content = potrace_path_to_svg(path, original_width, original_height, scale_down_x, scale_down_y)
    
    # Convertir a puntos para SPLINE en DXF (curvas suaves reales)
    spline_points = potrace_path_to_spline_points(path, height, scale_down_x, scale_down_y)

    # Polilíneas densas para preview (muestran suavidad)
    preview_polylines = potrace_path_to_polylines(path, height, settings.detail_level, scale_down_x, scale_down_y)
    
    return svg_content, spline_points, preview_polylines


def potrace_path_to_svg(
    path,
    width: int,
    height: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> str:
    """Convierte un path de Potrace a contenido SVG con curvas Bezier."""
    svg_paths = []
    
    for curve in path:
        start_x = curve.start_point.x * scale_x
        start_y = curve.start_point.y * scale_y
        path_d = f"M {start_x:.2f},{start_y:.2f}"
        
        for segment in curve:
            if segment.is_corner:
                # Segmento de esquina: línea recta pasando por punto de control
                c_x = segment.c.x * scale_x
                c_y = segment.c.y * scale_y
                end_x = segment.end_point.x * scale_x
                end_y = segment.end_point.y * scale_y
                path_d += f" L {c_x:.2f},{c_y:.2f}"
                path_d += f" L {end_x:.2f},{end_y:.2f}"
            else:
                # Segmento Bezier cúbico
                c1_x = segment.c1.x * scale_x
                c1_y = segment.c1.y * scale_y
                c2_x = segment.c2.x * scale_x
                c2_y = segment.c2.y * scale_y
                end_x = segment.end_point.x * scale_x
                end_y = segment.end_point.y * scale_y
                path_d += f" C {c1_x:.2f},{c1_y:.2f}"
                path_d += f" {c2_x:.2f},{c2_y:.2f}"
                path_d += f" {end_x:.2f},{end_y:.2f}"
        
        path_d += " Z"
        svg_paths.append(path_d)
    
    # Construir SVG completo
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{width}" 
     height="{height}" 
     viewBox="0 0 {width} {height}">
  <g fill="black" fill-rule="evenodd">
'''
    
    for path_d in svg_paths:
        svg_content += f'    <path d="{path_d}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    return svg_content


def potrace_path_to_spline_points(
    path,
    height: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[List[Tuple[float, float]]]:
    """
    Convierte un path de Potrace a puntos para SPLINE en DXF.
    Extrae los puntos clave de las curvas Bezier sin discretizar.
    Estos puntos servirán como "fit points" para el SPLINE.
    """
    spline_curves = []
    
    for curve in path:
        points = []
        current_point = (
            curve.start_point.x * scale_x,
            (height - curve.start_point.y) * scale_y,
        )
        points.append(current_point)
        
        for segment in curve:
            if segment.is_corner:
                # Esquina: añadir punto de control y punto final
                c = (segment.c.x * scale_x, (height - segment.c.y) * scale_y)
                end = (segment.end_point.x * scale_x, (height - segment.end_point.y) * scale_y)
                points.append(c)
                points.append(end)
                current_point = end
            else:
                # Curva Bezier: extraer puntos intermedios para mejor ajuste del spline
                # Usamos puntos de control como guía para el SPLINE
                p0 = current_point
                p1 = (segment.c1.x * scale_x, (height - segment.c1.y) * scale_y)
                p2 = (segment.c2.x * scale_x, (height - segment.c2.y) * scale_y)
                p3 = (segment.end_point.x * scale_x, (height - segment.end_point.y) * scale_y)
                
                # Añadir puntos a t=0.25, 0.5, 0.75 de la curva Bezier
                # Esto da suficiente información al SPLINE para recrear la curva
                for t in [0.25, 0.5, 0.75]:
                    x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
                    y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
                    points.append((x, y))
                
                # Añadir punto final
                points.append(p3)
                current_point = p3
        
        if len(points) >= 3:
            spline_curves.append(points)
    
    return spline_curves


def potrace_path_to_polylines(
    path,
    height: int,
    detail_level: float,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[List[Tuple[float, float]]]:
    """
    Convierte un path de Potrace a polilíneas para DXF (fallback).
    Las curvas Bezier se convierten a puntos discretos.
    """
    polylines = []
    
    # Número de puntos para aproximar cada curva Bezier
    # Más detalle = más puntos
    bezier_steps = int(10 * detail_level)
    bezier_steps = max(4, min(50, bezier_steps))
    
    for curve in path:
        points = []
        current_point = (
            curve.start_point.x * scale_x,
            (height - curve.start_point.y) * scale_y,
        )
        points.append(current_point)
        
        for segment in curve:
            if segment.is_corner:
                # Esquina: añadir punto de control y punto final
                c = (segment.c.x * scale_x, (height - segment.c.y) * scale_y)
                end = (segment.end_point.x * scale_x, (height - segment.end_point.y) * scale_y)
                points.append(c)
                points.append(end)
                current_point = end
            else:
                # Curva Bezier cúbica: discretizar
                p0 = current_point
                p1 = (segment.c1.x * scale_x, (height - segment.c1.y) * scale_y)
                p2 = (segment.c2.x * scale_x, (height - segment.c2.y) * scale_y)
                p3 = (segment.end_point.x * scale_x, (height - segment.end_point.y) * scale_y)
                
                # Generar puntos a lo largo de la curva Bezier
                for i in range(1, bezier_steps + 1):
                    t = i / bezier_steps
                    # Fórmula de Bezier cúbica
                    x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
                    y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
                    points.append((x, y))
                
                current_point = p3
        
        if len(points) >= 3:
            polylines.append(points)
    
    return polylines


def find_contours_cv(binary: np.ndarray) -> List[np.ndarray]:
    """Encuentra contornos usando OpenCV (muy rápido)."""
    if HAS_OPENCV:
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    else:
        # Fallback muy simplificado para cuando no hay OpenCV
        return find_contours_simple(binary)


def find_contours_simple(binary: np.ndarray) -> List[np.ndarray]:
    """
    Implementación simple de detección de contornos.
    Solo para fallback cuando OpenCV no está disponible.
    """
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    contours = []
    
    # Buscar bordes simples
    edges = np.zeros_like(binary)
    edges[1:, :] |= (binary[1:, :] != binary[:-1, :])
    edges[:, 1:] |= (binary[:, 1:] != binary[:, :-1])
    
    # Encontrar puntos de borde
    edge_points = np.argwhere(edges & (binary == 255))
    
    if len(edge_points) == 0:
        return []
    
    # Simplificar: tomar puntos cada N píxeles
    step = max(1, len(edge_points) // 1000)
    sampled = edge_points[::step]
    
    if len(sampled) > 2:
        # Convertir a formato de contorno OpenCV
        contour = np.array([[p[1], p[0]] for p in sampled], dtype=np.int32).reshape(-1, 1, 2)
        contours.append(contour)
    
    return contours


def simplify_contour(contour: np.ndarray, epsilon_factor: float) -> np.ndarray:
    """Simplifica un contorno usando Douglas-Peucker."""
    if HAS_OPENCV:
        # epsilon es un porcentaje del perímetro
        perimeter = cv2.arcLength(contour, True)
        epsilon = epsilon_factor * perimeter * 0.001
        return cv2.approxPolyDP(contour, epsilon, True)
    else:
        # Sin OpenCV, devolver el contorno tal cual
        return contour


def contours_to_svg_paths(contours: List[np.ndarray], settings: TraceSettings) -> List[str]:
    """Convierte contornos a paths SVG."""
    paths = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Simplificar según nivel de detalle (valores más bajos = más detalle)
        epsilon_factor = 1.0 / settings.detail_level
        simplified = simplify_contour(contour, epsilon_factor)
        
        if len(simplified) < 3:
            continue
        
        # Construir path SVG
        points = simplified.reshape(-1, 2)
        path_d = f"M{points[0][0]},{points[0][1]}"
        
        for point in points[1:]:
            path_d += f" L{point[0]},{point[1]}"
        
        path_d += " Z"
        paths.append(path_d)
    
    return paths


def contours_to_dxf_entities(contours: List[np.ndarray], settings: TraceSettings, height: int) -> List[List[Tuple[float, float]]]:
    """Convierte contornos a listas de puntos para DXF."""
    polylines = []
    
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Simplificar según nivel de detalle (valores más bajos = más detalle)
        # Con detail_level=2.0, epsilon_factor=0.5 (mantiene más detalle)
        # Con detail_level=5.0, epsilon_factor=0.2 (máximo detalle)
        epsilon_factor = 1.0 / settings.detail_level
        simplified = simplify_contour(contour, epsilon_factor)
        
        if len(simplified) < 3:
            continue
        
        # Convertir a lista de puntos (invertir Y para DXF)
        points = simplified.reshape(-1, 2)
        polyline = [(float(p[0]), float(height - p[1])) for p in points]
        polyline.append(polyline[0])  # Cerrar el polígono
        polylines.append(polyline)
    
    return polylines


def generate_dxf_ezdxf(polylines: List[List[Tuple[float, float]]], width: int, height: int, 
                       scale: float = 1.0, width_cm: float = 0.0, height_cm: float = 0.0,
                       use_spline: bool = True) -> bytes:
    """Genera archivo DXF usando ezdxf (formato R2000, compatible con Fusion 360).
    
    Args:
        polylines: Lista de polilíneas/splines (cada una es una lista de puntos (x, y))
        width: Ancho en píxeles
        height: Alto en píxeles
        scale: Factor de escala para convertir píxeles a cm
        width_cm: Ancho final en cm
        height_cm: Alto final en cm
        use_spline: Si True, usa SPLINE (curvas suaves). Si False, usa LWPOLYLINE.
        
    Returns:
        Contenido del archivo DXF como bytes
    """
    if not HAS_EZDXF:
        raise RuntimeError("ezdxf no está instalado. Instalar con: pip install ezdxf")
    
    # Crear documento DXF en formato R2000 (compatible y soporta SPLINE/LWPOLYLINE)
    doc = ezdxf.new('R2000')
    
    # Configurar unidades explícitamente en el header
    # $INSUNITS: 0=sin unidades, 1=pulgadas, 2=pies, 4=mm, 5=cm, 6=metros
    if width_cm > 0:
        doc.header['$INSUNITS'] = 5  # Centímetros
        doc.header['$LUNITS'] = 2    # Decimal
        doc.header['$LUPREC'] = 4    # 4 decimales de precisión
    
    # Obtener el modelspace para añadir entidades
    msp = doc.modelspace()
    
    # Calcular dimensiones finales para el bounding box
    final_width = width_cm if width_cm > 0 else width * scale
    final_height = height_cm if height_cm > 0 else height * scale
    
    # Añadir cada curva
    for curve_points in polylines:
        if len(curve_points) < 2:
            continue
        
        # Escalar puntos a cm si es necesario
        scaled_points = [(x * scale, y * scale) for x, y in curve_points]
        
        if use_spline:
            # Usar SPLINE para curvas suaves matemáticas
            # Los puntos son "fit points" - la curva pasará por ellos
            # Añadir el primer punto al final para cerrar la curva
            closed_points = scaled_points + [scaled_points[0]]
            msp.add_spline(closed_points, degree=3)
        else:
            # Fallback: usar LWPOLYLINE (polilíneas discretas)
            msp.add_lwpolyline(scaled_points, close=True)
    
    # Configurar los límites del dibujo (ayuda a Fusion 360 a interpretar el tamaño)
    doc.header['$EXTMIN'] = (0, 0, 0)
    doc.header['$EXTMAX'] = (final_width, final_height, 0)
    doc.header['$LIMMIN'] = (0, 0)
    doc.header['$LIMMAX'] = (final_width, final_height)
    
    # Guardar a string en memoria y luego convertir a bytes
    stream = io.StringIO()
    doc.write(stream)
    stream.seek(0)
    return stream.getvalue().encode('utf-8')


def generate_dxf(polylines: List[List[Tuple[float, float]]], width: int, height: int, 
                 scale: float = 1.0, width_cm: float = 0.0, height_cm: float = 0.0,
                 use_spline: bool = True) -> bytes:
    """Genera archivo DXF compatible con Fusion 360.
    
    Usa ezdxf si está disponible, sino genera un DXF básico manualmente.
    
    Args:
        polylines: Lista de polilíneas (cada una es una lista de puntos (x, y))
        width: Ancho en píxeles
        height: Alto en píxeles
        scale: Factor de escala para convertir píxeles a cm
        width_cm: Ancho final en cm
        height_cm: Alto final en cm
        use_spline: Si True, usa SPLINE (curvas suaves). Si False, usa LWPOLYLINE.
        
    Returns:
        Contenido del archivo DXF como bytes
    """
    if HAS_EZDXF:
        return generate_dxf_ezdxf(polylines, width, height, scale, width_cm, height_cm, use_spline)
    
    # Fallback: generar DXF R12 manualmente (muy básico)
    # Formato R12 es el más compatible con todos los programas CAD
    
    dxf_content = """0
SECTION
2
HEADER
9
$ACADVER
1
AC1009
0
ENDSEC
0
SECTION
2
TABLES
0
TABLE
2
LTYPE
70
1
0
LTYPE
2
CONTINUOUS
70
0
3
Solid line
72
65
73
0
40
0.0
0
ENDTAB
0
TABLE
2
LAYER
70
1
0
LAYER
2
0
70
0
62
7
6
CONTINUOUS
0
ENDTAB
0
ENDSEC
0
SECTION
2
ENTITIES
"""
    
    # Añadir cada polilínea como POLYLINE (formato R12)
    for polyline in polylines:
        if len(polyline) < 2:
            continue
        
        # Inicio de POLYLINE
        dxf_content += """0
POLYLINE
8
0
66
1
70
1
"""
        
        # Vértices
        for x, y in polyline:
            x_scaled = x * scale
            y_scaled = y * scale
            dxf_content += f"""0
VERTEX
8
0
10
{x_scaled:.6f}
20
{y_scaled:.6f}
"""
        
        # Fin de POLYLINE
        dxf_content += """0
SEQEND
8
0
"""
    
    dxf_content += """0
ENDSEC
0
EOF
"""
    return dxf_content.encode('utf-8')


def trace_bitmap_to_svg(img: Image.Image, settings: TraceSettings) -> str:
    """Convierte una imagen a SVG en modo blanco y negro."""
    gray_array = image_to_grayscale_array(img, settings)
    binary = threshold_image(gray_array, settings.threshold)
    
    height, width = binary.shape
    
    # Encontrar contornos
    contours = find_contours_cv(binary)
    
    # Convertir a paths SVG
    svg_paths = contours_to_svg_paths(contours, settings)
    
    # Generar SVG
    fill_color = "black"
    fill_rule = "evenodd" if settings.ignore_white else "nonzero"
    
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{width}" 
     height="{height}" 
     viewBox="0 0 {width} {height}">
  <g fill="{fill_color}" fill-rule="{fill_rule}">
'''
    
    for path_d in svg_paths:
        svg_content += f'    <path d="{path_d}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    return svg_content


def trace_outline_to_svg(img: Image.Image, settings: TraceSettings) -> str:
    """Detecta bordes y los convierte a SVG."""
    gray_array = image_to_grayscale_array(img, settings)
    
    height, width = gray_array.shape
    
    if HAS_OPENCV:
        # Usar Canny edge detection (muy rápido y buenos resultados)
        edges = cv2.Canny(gray_array, settings.threshold // 2, settings.threshold)
    else:
        # Fallback: usar filtro de bordes de PIL
        gray_img = Image.fromarray(gray_array)
        edges_img = gray_img.filter(ImageFilter.FIND_EDGES)
        edges = np.array(edges_img)
        edges = np.where(edges > 30, 255, 0).astype(np.uint8)
    
    # Encontrar contornos en los bordes
    contours = find_contours_cv(edges)
    
    # Convertir a paths SVG
    svg_paths = contours_to_svg_paths(contours, settings)
    
    # Generar SVG con líneas en lugar de relleno
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{width}" 
     height="{height}" 
     viewBox="0 0 {width} {height}">
  <g fill="none" stroke="black" stroke-width="1">
'''
    
    for path_d in svg_paths:
        svg_content += f'    <path d="{path_d}"/>\n'
    
    svg_content += '''  </g>
</svg>'''
    
    return svg_content


def quantize_colors(img: Image.Image, num_colors: int) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """Reduce el número de colores de una imagen usando k-means."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    if HAS_OPENCV:
        # Usar k-means de OpenCV (muy rápido)
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(height, width, 3)
        
        colors = [tuple(c) for c in centers]
    else:
        # Fallback: usar cuantización de PIL
        quantized_pil = img.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT)
        quantized = np.array(quantized_pil.convert("RGB"))
        
        # Obtener colores únicos
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)
        colors = [tuple(c) for c in unique_colors[:num_colors]]
    
    return quantized, colors


def trace_color_to_svg(img: Image.Image, settings: TraceSettings) -> str:
    """Calca una imagen a color generando capas SVG."""
    quantized, colors = quantize_colors(img, settings.color_count)
    height, width = quantized.shape[:2]
    
    svg_paths = []
    
    for color in colors:
        # Saltar blanco si se requiere
        if settings.ignore_white and sum(color) > 750:  # Casi blanco
            continue
        
        # Crear máscara para este color
        mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
        
        # Encontrar contornos
        contours = find_contours_cv(mask)
        
        # Convertir a paths
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        paths = contours_to_svg_paths(contours, settings)
        
        for path_d in paths:
            svg_paths.append((hex_color, path_d))
    
    # Generar SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{width}" 
     height="{height}" 
     viewBox="0 0 {width} {height}">
'''
    
    for color, path_d in svg_paths:
        svg_content += f'  <path fill="{color}" d="{path_d}"/>\n'
    
    svg_content += '</svg>'
    
    return svg_content


def trace_grayscale_to_svg(img: Image.Image, settings: TraceSettings) -> str:
    """Calca una imagen en escala de grises con múltiples niveles."""
    gray_array = image_to_grayscale_array(img, settings)
    height, width = gray_array.shape
    
    num_levels = settings.color_count
    svg_paths = []
    
    for level in range(num_levels):
        low = int(255 * level / num_levels)
        high = int(255 * (level + 1) / num_levels)
        
        # Saltar niveles muy claros si se requiere
        if settings.ignore_white and high >= 240:
            continue
        
        # Crear máscara para este nivel
        mask = ((gray_array >= low) & (gray_array < high)).astype(np.uint8) * 255
        
        # Encontrar contornos
        contours = find_contours_cv(mask)
        
        # Color del nivel
        mid_gray = (low + high) // 2
        hex_color = '#{:02x}{:02x}{:02x}'.format(mid_gray, mid_gray, mid_gray)
        
        paths = contours_to_svg_paths(contours, settings)
        
        for path_d in paths:
            svg_paths.append((hex_color, path_d))
    
    # Generar SVG
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{width}" 
     height="{height}" 
     viewBox="0 0 {width} {height}">
'''
    
    for color, path_d in svg_paths:
        svg_content += f'  <path fill="{color}" d="{path_d}"/>\n'
    
    svg_content += '</svg>'
    
    return svg_content


@dataclass
class TraceResult:
    """Resultado del calco con datos para ambos formatos."""
    svg_content: str
    dxf_content: bytes  # DXF como bytes para compatibilidad con ezdxf
    width: int
    height: int
    path_count: int
    # Dimensiones físicas en cm (para DXF)
    width_cm: float = 0.0
    height_cm: float = 0.0
    # Polilíneas para preview (curvas densas)
    preview_polylines: Optional[List[List[Tuple[float, float]]]] = None


def extract_contours_from_image(img: Image.Image, settings: TraceSettings) -> Tuple[List[np.ndarray], int, int]:
    """Extrae contornos de una imagen según el modo."""
    all_contours = []
    
    if settings.mode == TraceMode.BLACK_WHITE:
        gray_array = image_to_grayscale_array(img, settings)
        binary = threshold_image(gray_array, settings.threshold)
        height, width = binary.shape
        all_contours = find_contours_cv(binary)
    
    elif settings.mode == TraceMode.OUTLINE:
        gray_array = image_to_grayscale_array(img, settings)
        height, width = gray_array.shape
        if HAS_OPENCV:
            edges = cv2.Canny(gray_array, settings.threshold // 2, settings.threshold)
        else:
            gray_img = Image.fromarray(gray_array)
            edges = np.array(gray_img.filter(ImageFilter.FIND_EDGES))
            edges = np.where(edges > 30, 255, 0).astype(np.uint8)
        all_contours = find_contours_cv(edges)
    
    elif settings.mode == TraceMode.GRAYSCALE:
        gray_array = image_to_grayscale_array(img, settings)
        height, width = gray_array.shape
        num_levels = settings.color_count
        for level in range(num_levels):
            low = int(255 * level / num_levels)
            high = int(255 * (level + 1) / num_levels)
            if settings.ignore_white and high >= 240:
                continue
            mask = ((gray_array >= low) & (gray_array < high)).astype(np.uint8) * 255
            contours = find_contours_cv(mask)
            all_contours.extend(contours)
    
    elif settings.mode == TraceMode.COLOR:
        quantized, colors = quantize_colors(img, settings.color_count)
        height, width = quantized.shape[:2]
        for color in colors:
            if settings.ignore_white and sum(color) > 750:
                continue
            mask = np.all(quantized == color, axis=2).astype(np.uint8) * 255
            contours = find_contours_cv(mask)
            all_contours.extend(contours)
    else:
        width, height = img.size
    
    return all_contours, width, height


def trace_image(img: Image.Image, settings: TraceSettings) -> TraceResult:
    """Función principal de calco de imagen. Devuelve ambos formatos."""
    width, height = img.size
    
    # Usar Potrace para vectorización profesional si está disponible
    # Potrace funciona mejor para imágenes B/N y contornos
    use_potrace = HAS_POTRACE and settings.mode in [TraceMode.BLACK_WHITE, TraceMode.OUTLINE]
    
    if use_potrace:
        # Vectorización con Potrace (profesional)
        svg_content, potrace_splines, preview_polylines = trace_with_potrace(img, settings)
        polylines_for_dxf = potrace_splines
    else:
        # Fallback al algoritmo básico para colores/escala de grises
        contours, width, height = extract_contours_from_image(img, settings)
        
        # Generar SVG usando las funciones existentes
        if settings.mode == TraceMode.BLACK_WHITE:
            svg_content = trace_bitmap_to_svg(img, settings)
        elif settings.mode == TraceMode.OUTLINE:
            svg_content = trace_outline_to_svg(img, settings)
        elif settings.mode == TraceMode.GRAYSCALE:
            svg_content = trace_grayscale_to_svg(img, settings)
        elif settings.mode == TraceMode.COLOR:
            svg_content = trace_color_to_svg(img, settings)
        else:
            svg_content = ""
        
        # Convertir contornos para DXF
        polylines_for_dxf = contours_to_dxf_entities(contours, settings, height)
        preview_polylines = None
    
    # Calcular escala para DXF en cm
    scale = 1.0
    width_cm = 0.0
    height_cm = 0.0
    
    if settings.use_physical_dims and settings.dim_value_cm > 0:
        if settings.dim_mode == "width":
            # El usuario especificó el ancho
            scale = settings.dim_value_cm / width
            width_cm = settings.dim_value_cm
            height_cm = height * scale
        else:
            # El usuario especificó el alto
            scale = settings.dim_value_cm / height
            height_cm = settings.dim_value_cm
            width_cm = width * scale
    
    # Generar DXF
    dxf_content = generate_dxf(
        polylines_for_dxf,
        width,
        height,
        scale,
        width_cm,
        height_cm,
        use_spline=use_potrace,
    )
    
    # Contar paths
    path_count = svg_content.count('<path')
    
    return TraceResult(
        svg_content=svg_content,
        dxf_content=dxf_content,
        width=width,
        height=height,
        path_count=path_count,
        width_cm=width_cm,
        height_cm=height_cm,
        preview_polylines=preview_polylines
    )


class ResultDialog(tk.Toplevel):
    """Diálogo para mostrar y gestionar el resultado del calco."""
    
    def __init__(self, parent: tk.Tk, trace_result: TraceResult, original_image: Image.Image, settings: TraceSettings) -> None:
        super().__init__(parent)
        self.title("Resultado - Calco de imagen")
        self.trace_result = trace_result
        self.svg_content = trace_result.svg_content  # Para compatibilidad
        self.dxf_content = trace_result.dxf_content
        self.original_image = original_image
        self.settings = settings
        
        # Calcular dimensiones para la ventana
        max_preview_size = 400
        img_width, img_height = original_image.size
        
        if img_width > max_preview_size or img_height > max_preview_size:
            scale = min(max_preview_size / img_width, max_preview_size / img_height)
            preview_width = int(img_width * scale)
            preview_height = int(img_height * scale)
        else:
            preview_width = img_width
            preview_height = img_height
        
        window_width = preview_width * 2 + 80
        window_height = preview_height + 180
        self.geometry(f"{window_width}x{window_height}")
        self.resizable(False, False)
        
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.winfo_screenheight() // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        self._build_ui(preview_width, preview_height)
        
        self.transient(parent)
        self.grab_set()
    
    def _build_ui(self, preview_width: int, preview_height: int) -> None:
        # Información
        info_frame = ttk.Frame(self)
        info_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        info_text = f"Modo: {self.settings.mode.value} | Tamaño: {self.original_image.width}×{self.original_image.height}px"
        ttk.Label(info_frame, text=info_text, font=("TkDefaultFont", 9, "bold")).pack()
        
        # Frame para comparación
        compare_frame = ttk.Frame(self)
        compare_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Original
        orig_frame = ttk.LabelFrame(compare_frame, text="Original")
        orig_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        preview_img = self.original_image.copy()
        if preview_img.mode not in ("RGB", "RGBA"):
            preview_img = preview_img.convert("RGB")
        preview_img = preview_img.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        
        self.photo_orig = ImageTk.PhotoImage(preview_img)
        ttk.Label(orig_frame, image=self.photo_orig).pack(padx=10, pady=10)
        
        # Resultado (preview del procesamiento)
        result_frame = ttk.LabelFrame(compare_frame, text="Calco vectorial")
        result_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))
        
        # Crear preview del resultado
        preview_result = self._create_preview()
        preview_result = preview_result.resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        
        self.photo_result = ImageTk.PhotoImage(preview_result)
        ttk.Label(result_frame, image=self.photo_result).pack(padx=10, pady=10)
        
        # Info de formatos
        svg_size = len(self.svg_content.encode('utf-8'))
        dxf_size = len(self.dxf_content)  # Ya es bytes
        path_count = self.trace_result.path_count
        size_text = f"SVG: {svg_size:,} bytes | DXF: {dxf_size:,} bytes | Paths: {path_count}"
        ttk.Label(self, text=size_text, font=("TkDefaultFont", 8), foreground="gray").pack()
        
        # Info de dimensiones DXF en cm
        if self.trace_result.width_cm > 0 and self.trace_result.height_cm > 0:
            dims_text = f"DXF: {self.trace_result.width_cm:.2f} × {self.trace_result.height_cm:.2f} cm"
            ttk.Label(self, text=dims_text, font=("TkDefaultFont", 9, "bold"), foreground="blue").pack()
        
        # Botones de acción
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        ttk.Button(
            btn_frame,
            text="📋 Copiar SVG",
            command=self._copy_to_clipboard
        ).pack(side="left", padx=(0, 5), expand=True, fill="x")
        
        ttk.Button(
            btn_frame,
            text="💾 Guardar SVG",
            command=self._save_svg
        ).pack(side="left", padx=5, expand=True, fill="x")
        
        ttk.Button(
            btn_frame,
            text="💾 Guardar DXF",
            command=self._save_dxf
        ).pack(side="left", padx=5, expand=True, fill="x")
        
        ttk.Button(
            btn_frame,
            text="Cerrar",
            command=self.destroy
        ).pack(side="left", padx=(5, 0), expand=True, fill="x")
    
    def _create_preview(self) -> Image.Image:
        """Crea una imagen de preview mostrando los contornos vectoriales reales."""
        img = self.original_image.copy()
        width, height = img.size
        
        # Crear imagen blanca del mismo tamaño
        preview = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Si hay polilíneas densas de preview (Potrace), dibujarlas
        if self.trace_result.preview_polylines:
            if HAS_OPENCV:
                for polyline in self.trace_result.preview_polylines:
                    if len(polyline) < 2:
                        continue
                    pts = np.array(
                        [[int(round(x)), int(round(y))] for x, y in polyline],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2)
                    cv2.polylines(preview, [pts], isClosed=True, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                return Image.fromarray(preview)

            preview_img = Image.fromarray(preview)
            draw = ImageDraw.Draw(preview_img)
            for polyline in self.trace_result.preview_polylines:
                if len(polyline) < 2:
                    continue
                pts = [(int(round(x)), int(round(y))) for x, y in polyline]
                draw.line(pts + [pts[0]], fill=(0, 0, 0), width=1)
            return preview_img

        # Extraer contornos de la misma manera que para el DXF (fallback)
        contours, _, _ = extract_contours_from_image(img, self.settings)

        if HAS_OPENCV:
            # Dibujar cada contorno simplificado (igual que en el DXF)
            for contour in contours:
                if len(contour) < 3:
                    continue

                # Aplicar la misma simplificación que se usa para el DXF
                epsilon_factor = 1.0 / self.settings.detail_level
                simplified = simplify_contour(contour, epsilon_factor)

                if len(simplified) < 3:
                    continue

                cv2.drawContours(preview, [simplified], -1, (0, 0, 0), 1, cv2.LINE_AA)

            return Image.fromarray(preview)

        preview_img = Image.fromarray(preview)
        draw = ImageDraw.Draw(preview_img)
        for contour in contours:
            if len(contour) < 3:
                continue

            epsilon_factor = 1.0 / self.settings.detail_level
            simplified = simplify_contour(contour, epsilon_factor)

            if len(simplified) < 3:
                continue

            points = simplified.reshape(-1, 2)
            pts = [(int(p[0]), int(p[1])) for p in points]
            draw.line(pts + [pts[0]], fill=(0, 0, 0), width=1)

        return preview_img
    
    def _copy_to_clipboard(self) -> None:
        """Copia el SVG al portapapeles."""
        try:
            self.clipboard_clear()
            self.clipboard_append(self.svg_content)
            messagebox.showinfo("Copiado", "¡SVG copiado al portapapeles!")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo copiar:\n{str(exc)}")
    
    def _save_svg(self) -> None:
        """Guarda el calco como archivo SVG."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Guardar SVG",
                defaultextension=".svg",
                filetypes=[
                    ("SVG - Scalable Vector Graphics", "*.svg"),
                    ("Todos los archivos", "*.*")
                ],
                initialfile=f"calco_{self.settings.mode.name.lower()}.svg"
            )
            
            if not file_path:
                return
            
            output_path = Path(file_path)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.svg_content)
            
            messagebox.showinfo("Guardado", f"SVG guardado exitosamente como:\n{output_path.name}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar:\n{str(exc)}")
    
    def _save_dxf(self) -> None:
        """Guarda el calco como archivo DXF (compatible con Fusion 360)."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Guardar DXF",
                defaultextension=".dxf",
                filetypes=[
                    ("DXF - AutoCAD Drawing Exchange", "*.dxf"),
                    ("Todos los archivos", "*.*")
                ],
                initialfile=f"calco_{self.settings.mode.name.lower()}.dxf"
            )
            
            if not file_path:
                return
            
            output_path = Path(file_path)
            
            # DXF es bytes, escribir en modo binario
            with open(output_path, 'wb') as f:
                f.write(self.dxf_content)
            
            # Mostrar información de dimensiones si están disponibles
            if self.trace_result.width_cm > 0:
                dims_info = f"\n\nDimensiones: {self.trace_result.width_cm:.2f} × {self.trace_result.height_cm:.2f} cm"
            else:
                dims_info = ""
            
            messagebox.showinfo("Guardado", f"DXF guardado exitosamente como:\n{output_path.name}{dims_info}")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar:\n{str(exc)}")


class CalcoImagenApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Calco de imagen (Image Trace)")
        self.geometry("700x500")
        self.resizable(False, False)
        
        # Estado
        self.selected_image = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=TraceMode.BLACK_WHITE.value)
        self.threshold_var = tk.IntVar(value=128)
        self.color_count_var = tk.IntVar(value=6)
        self.detail_var = tk.DoubleVar(value=2.0)
        self.smoothness_var = tk.DoubleVar(value=0.5)
        self.corner_var = tk.DoubleVar(value=1.0)
        self.ignore_white_var = tk.BooleanVar(value=True)
        self.invert_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Listo.")
        self.clipboard_image: Optional[Image.Image] = None
        
        # Estado para dimensiones DXF en cm
        self.use_physical_dims_var = tk.BooleanVar(value=True)
        self.dim_mode_var = tk.StringVar(value="width")  # "width" o "height"
        self.dim_value_var = tk.DoubleVar(value=10.0)  # Valor en cm
        self.dim_calculated_var = tk.StringVar(value="")  # Dimensión calculada
        
        self._build_ui()
        self._check_opencv()
    
    def _check_opencv(self) -> None:
        """Verifica si OpenCV está disponible."""
        if not HAS_OPENCV:
            self.status_var.set("⚠ OpenCV no instalado. Instalar con: pip install opencv-python")
    
    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 6}
        
        # Selección de imagen
        frame_image = ttk.LabelFrame(self, text="Imagen")
        frame_image.pack(fill="x", **padding)
        
        btn_browse = ttk.Button(frame_image, text="Elegir imagen...", command=self._on_browse)
        btn_browse.pack(side="left", padx=(10, 8), pady=10)
        
        btn_paste = ttk.Button(frame_image, text="Pegar desde portapapeles", command=self._on_paste_clipboard)
        btn_paste.pack(side="left", padx=(0, 8), pady=10)
        
        self.lbl_image = ttk.Label(frame_image, text="(ninguna)", width=50)
        self.lbl_image.pack(side="left", padx=(0, 10))
        
        # Modo de calco
        frame_mode = ttk.LabelFrame(self, text="Modo de calco")
        frame_mode.pack(fill="x", **padding)
        
        for mode in TraceMode:
            ttk.Radiobutton(
                frame_mode,
                text=mode.value,
                variable=self.mode_var,
                value=mode.value,
                command=self._on_mode_change
            ).pack(side="left", padx=10, pady=8)
        
        # Parámetros
        frame_params = ttk.LabelFrame(self, text="Parámetros")
        frame_params.pack(fill="x", **padding)
        
        # Fila 1: Umbral y Colores
        row1 = ttk.Frame(frame_params)
        row1.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(row1, text="Umbral (B/N):").pack(side="left")
        self.spn_threshold = ttk.Spinbox(
            row1, from_=0, to=255, increment=5,
            textvariable=self.threshold_var, width=8
        )
        self.spn_threshold.pack(side="left", padx=(5, 20))
        
        ttk.Label(row1, text="Núm. colores:").pack(side="left")
        self.spn_colors = ttk.Spinbox(
            row1, from_=2, to=32, increment=1,
            textvariable=self.color_count_var, width=8
        )
        self.spn_colors.pack(side="left", padx=(5, 20))
        self.spn_colors.configure(state="disabled")
        
        ttk.Checkbutton(
            row1, text="Ignorar blanco",
            variable=self.ignore_white_var
        ).pack(side="left", padx=10)
        
        ttk.Checkbutton(
            row1, text="Invertir",
            variable=self.invert_var
        ).pack(side="left", padx=10)
        
        # Fila 2: Detalle y Suavizado
        row2 = ttk.Frame(frame_params)
        row2.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(row2, text="Nivel de detalle:").pack(side="left")
        self.scale_detail = ttk.Scale(
            row2, from_=0.5, to=5.0,
            variable=self.detail_var, orient="horizontal", length=120
        )
        self.scale_detail.pack(side="left", padx=(5, 5))
        self.lbl_detail = ttk.Label(row2, text="2.0", width=4)
        self.lbl_detail.pack(side="left", padx=(0, 20))
        self.detail_var.trace_add("write", self._update_detail_label)
        
        ttk.Label(row2, text="Curvas:").pack(side="left")
        self.scale_smooth = ttk.Scale(
            row2, from_=0.0, to=2.0,
            variable=self.smoothness_var, orient="horizontal", length=120
        )
        self.scale_smooth.pack(side="left", padx=(5, 5))
        self.lbl_smooth = ttk.Label(row2, text="0.5", width=4)
        self.lbl_smooth.pack(side="left")
        self.smoothness_var.trace_add("write", self._update_smooth_label)
        
        # Dimensiones DXF
        frame_dims = ttk.LabelFrame(self, text="Dimensiones DXF (cm)")
        frame_dims.pack(fill="x", **padding)
        
        row_dims = ttk.Frame(frame_dims)
        row_dims.pack(fill="x", padx=10, pady=5)
        
        self.chk_physical = ttk.Checkbutton(
            row_dims, text="Usar dimensiones físicas",
            variable=self.use_physical_dims_var,
            command=self._on_dims_toggle
        )
        self.chk_physical.pack(side="left", padx=(0, 15))
        
        ttk.Label(row_dims, text="Especificar:").pack(side="left")
        self.cmb_dim_mode = ttk.Combobox(
            row_dims, textvariable=self.dim_mode_var,
            values=["Ancho", "Alto"], state="readonly", width=8
        )
        self.cmb_dim_mode.set("Ancho")
        self.cmb_dim_mode.pack(side="left", padx=(5, 10))
        self.cmb_dim_mode.bind("<<ComboboxSelected>>", self._on_dim_mode_change)
        
        self.spn_dim_value = ttk.Spinbox(
            row_dims, from_=0.1, to=1000.0, increment=0.5,
            textvariable=self.dim_value_var, width=8
        )
        self.spn_dim_value.pack(side="left", padx=(0, 5))
        ttk.Label(row_dims, text="cm").pack(side="left", padx=(0, 15))
        
        self.lbl_dim_calculated = ttk.Label(row_dims, textvariable=self.dim_calculated_var, foreground="gray")
        self.lbl_dim_calculated.pack(side="left")
        
        # Información de ayuda
        info_frame = ttk.Frame(self)
        info_frame.pack(fill="x", padx=10, pady=2)
        
        opencv_status = "✓ OpenCV disponible" if HAS_OPENCV else "⚠ Sin OpenCV (lento)"
        ttk.Label(
            info_frame,
            text=f"Convierte imágenes a gráficos vectoriales SVG | {opencv_status}",
            font=("TkDefaultFont", 8),
            foreground="green" if HAS_OPENCV else "orange"
        ).pack(side="left")
        
        # Acciones
        frame_actions = ttk.Frame(self)
        frame_actions.pack(fill="x", **padding)
        
        self.btn_process = ttk.Button(
            frame_actions,
            text="Calcar imagen",
            command=self._on_process
        )
        self.btn_process.pack(side="left", padx=(10, 8))
        
        self.progress = ttk.Progressbar(
            frame_actions,
            orient="horizontal",
            mode="indeterminate"
        )
        self.progress.pack(fill="x", expand=True, padx=(0, 10))
        
        # Estado
        frame_status = ttk.Frame(self)
        frame_status.pack(fill="x", **padding)
        self.lbl_status = ttk.Label(frame_status, textvariable=self.status_var, anchor="w")
        self.lbl_status.pack(fill="x", padx=10)
    
    def _update_detail_label(self, *args) -> None:
        self.lbl_detail.configure(text=f"{self.detail_var.get():.1f}")
    
    def _update_smooth_label(self, *args) -> None:
        self.lbl_smooth.configure(text=f"{self.smoothness_var.get():.1f}")
    
    def _on_dims_toggle(self) -> None:
        """Habilita/deshabilita controles de dimensiones."""
        enabled = self.use_physical_dims_var.get()
        state = "readonly" if enabled else "disabled"
        state_spn = "normal" if enabled else "disabled"
        self.cmb_dim_mode.configure(state=state)
        self.spn_dim_value.configure(state=state_spn)
        if enabled:
            self._update_calculated_dim()
        else:
            self.dim_calculated_var.set("")
    
    def _on_dim_mode_change(self, event=None) -> None:
        """Actualiza el modo de dimensión (ancho/alto)."""
        mode = self.cmb_dim_mode.get()
        self.dim_mode_var.set("width" if mode == "Ancho" else "height")
        self._update_calculated_dim()
    
    def _update_calculated_dim(self) -> None:
        """Calcula y muestra la dimensión complementaria."""
        if not self.use_physical_dims_var.get():
            self.dim_calculated_var.set("")
            return
        
        # Obtener dimensiones de la imagen actual
        img_width, img_height = self._get_current_image_size()
        if img_width == 0 or img_height == 0:
            self.dim_calculated_var.set("(carga una imagen)")
            return
        
        try:
            value_cm = self.dim_value_var.get()
            if value_cm <= 0:
                self.dim_calculated_var.set("")
                return
            
            if self.dim_mode_var.get() == "width":
                # Ancho especificado, calcular alto
                scale = value_cm / img_width
                calculated = img_height * scale
                self.dim_calculated_var.set(f"→ Alto: {calculated:.2f} cm")
            else:
                # Alto especificado, calcular ancho
                scale = value_cm / img_height
                calculated = img_width * scale
                self.dim_calculated_var.set(f"→ Ancho: {calculated:.2f} cm")
        except Exception:
            self.dim_calculated_var.set("")
    
    def _get_current_image_size(self) -> Tuple[int, int]:
        """Obtiene las dimensiones de la imagen actual."""
        if self.clipboard_image is not None:
            return self.clipboard_image.size
        
        image_path = self.selected_image.get().strip()
        if image_path and os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    return img.size
            except Exception:
                pass
        
        return (0, 0)
    
    def _on_mode_change(self) -> None:
        mode_str = self.mode_var.get()
        
        if mode_str == TraceMode.BLACK_WHITE.value or mode_str == TraceMode.OUTLINE.value:
            self.spn_threshold.configure(state="normal")
            self.spn_colors.configure(state="disabled")
        else:
            self.spn_threshold.configure(state="disabled")
            self.spn_colors.configure(state="normal")
    
    def _on_browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )
        if not selected:
            return
        self.selected_image.set(selected)
        self.clipboard_image = None
        self.lbl_image.configure(text=self._ellipsize_path(selected, max_chars=50))
        self.status_var.set("Imagen seleccionada. Ajusta los parámetros y pulsa Calcar.")
        self._update_calculated_dim()
    
    def _on_paste_clipboard(self) -> None:
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showwarning(
                    "Sin imagen", 
                    "No hay imagen en el portapapeles.\n\nCopia una imagen (Ctrl+C) e intenta de nuevo."
                )
                return
            
            if not isinstance(img, Image.Image):
                messagebox.showwarning(
                    "No es una imagen",
                    "El contenido del portapapeles no es una imagen válida."
                )
                return
            
            self.clipboard_image = img
            self.selected_image.set("")
            self.lbl_image.configure(text=f"(portapapeles: {img.width}×{img.height}px)")
            self.status_var.set("Imagen del portapapeles cargada. Ajusta los parámetros y pulsa Calcar.")
            self._update_calculated_dim()
            
        except Exception as exc:
            messagebox.showerror(
                "Error",
                f"No se pudo obtener la imagen del portapapeles:\n{str(exc)}"
            )
    
    def _get_current_settings(self) -> TraceSettings:
        mode_str = self.mode_var.get()
        mode = TraceMode.BLACK_WHITE
        for m in TraceMode:
            if m.value == mode_str:
                mode = m
                break
        
        return TraceSettings(
            mode=mode,
            threshold=self.threshold_var.get(),
            color_count=self.color_count_var.get(),
            detail_level=self.detail_var.get(),
            smoothness=self.smoothness_var.get(),
            corner_threshold=self.corner_var.get(),
            ignore_white=self.ignore_white_var.get(),
            invert=self.invert_var.get(),
            use_physical_dims=self.use_physical_dims_var.get(),
            dim_mode=self.dim_mode_var.get(),
            dim_value_cm=self.dim_value_var.get(),
        )
    
    def _on_process(self) -> None:
        image_path = self.selected_image.get().strip()
        
        if not image_path and self.clipboard_image is None:
            messagebox.showwarning("Falta imagen", "Selecciona primero una imagen o pega una desde el portapapeles.")
            return
        
        settings = self._get_current_settings()
        
        if self.clipboard_image is not None:
            self.progress.start()
            self._set_ui_enabled(False)
            self.status_var.set("Procesando imagen del portapapeles...")
            
            thread = threading.Thread(
                target=self._process_clipboard_in_background,
                args=(self.clipboard_image.copy(), settings),
                daemon=True,
            )
            thread.start()
            return
        
        if not os.path.exists(image_path):
            messagebox.showerror("Error", "La imagen seleccionada no existe.")
            return
        
        input_path = Path(image_path)
        if not is_image_file(input_path):
            messagebox.showwarning("Archivo inválido", "El archivo seleccionado no es una imagen válida.")
            return
        
        self.progress.start()
        self._set_ui_enabled(False)
        self.status_var.set("Procesando imagen...")
        
        thread = threading.Thread(
            target=self._process_file_in_background,
            args=(input_path, settings),
            daemon=True,
        )
        thread.start()
    
    def _process_file_in_background(self, input_path: Path, settings: TraceSettings) -> None:
        try:
            with Image.open(input_path) as img:
                trace_result = trace_image(img, settings)
                original = img.copy()
            
            def _finish() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                self.status_var.set("¡Completado!")
                ResultDialog(self, trace_result, original, settings)
            
            self.after(0, _finish)
        except Exception as exc:
            def _error() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                error_msg = f"Error al procesar la imagen: {str(exc)}"
                self.status_var.set(error_msg)
                messagebox.showerror("Error", error_msg)
            
            self.after(0, _error)
    
    def _process_clipboard_in_background(self, img: Image.Image, settings: TraceSettings) -> None:
        try:
            trace_result = trace_image(img, settings)
            
            def _finish() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                self.status_var.set("¡Completado!")
                ResultDialog(self, trace_result, img, settings)
            
            self.after(0, _finish)
        except Exception as exc:
            def _error() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                error_msg = f"Error al procesar la imagen: {str(exc)}"
                self.status_var.set(error_msg)
                messagebox.showerror("Error", error_msg)
            
            self.after(0, _error)
    
    def _set_ui_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_process.configure(state=state)
    
    @staticmethod
    def _ellipsize_path(path_str: str, max_chars: int = 70) -> str:
        if len(path_str) <= max_chars:
            return path_str
        head = path_str[: max_chars // 2 - 2]
        tail = path_str[-(max_chars // 2 - 3) :]
        return f"{head}...{tail}"


def main() -> None:
    app = CalcoImagenApp()
    app.mainloop()


if __name__ == "__main__":
    main()
