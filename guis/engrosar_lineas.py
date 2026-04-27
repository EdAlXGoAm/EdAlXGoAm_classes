import os
import io
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageFilter, ImageOps, ImageGrab, ImageTk
import numpy as np


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


def thicken_lines_image(img: Image.Image, percentage: float) -> Image.Image:
    """Engrosa las líneas de una imagen PIL en blanco y negro según un porcentaje.
    
    Args:
        img: Imagen PIL de entrada
        percentage: Porcentaje de engrosamiento (ej: 150 = 150%, engrosa 1.5x)
        
    Returns:
        Imagen PIL procesada en modo "1" (blanco y negro)
    """
    # Convertir a escala de grises si no lo está
    if img.mode != "L":
        img = img.convert("L")
    
    # Convertir a numpy array
    img_array = np.array(img, dtype=np.uint8)
    
    # Determinar si las líneas son negras o blancas
    # Si la mayoría de píxeles son claros, las líneas son oscuras
    mean_value = np.mean(img_array)
    if mean_value > 127:
        # Fondo claro, líneas oscuras - invertir para trabajar con líneas blancas
        img_array = 255 - img_array
        lines_are_white = True
    else:
        # Fondo oscuro, líneas claras - ya están como líneas blancas
        lines_are_white = False
    
    # Convertir a binario: 1 para líneas (blancas), 0 para fondo (negro)
    threshold = 128
    binary = (img_array >= threshold).astype(np.uint8)
    
    # Calcular el tamaño del kernel de dilatación basado en el porcentaje
    # El porcentaje se interpreta como un factor multiplicativo
    # Por ejemplo, 150% significa engrosar 1.5 veces más
    width, height = img_array.shape
    min_dimension = min(width, height)
    
    # El radio base es una fracción pequeña de la dimensión mínima
    # Por ejemplo, 1% de la dimensión mínima como base
    base_radius = max(1, int(min_dimension * 0.01))
    
    # Aplicar el porcentaje como factor multiplicativo
    # Si percentage = 100, no cambia; si = 200, duplica el grosor
    radius = max(1, int(base_radius * (percentage / 100.0)))
    
    # Implementar dilatación morfológica
    # Crear kernel circular usando numpy vectorizado
    kernel_size = radius * 2 + 1
    center = kernel_size // 2
    y_coords, x_coords = np.ogrid[:kernel_size, :kernel_size]
    dist_sq = (x_coords - center) ** 2 + (y_coords - center) ** 2
    kernel = (dist_sq <= radius ** 2).astype(bool)
    
    # Intentar usar scipy para mejor rendimiento, si está disponible
    try:
        from scipy import ndimage
        dilated = ndimage.binary_dilation(binary.astype(bool), structure=kernel).astype(np.uint8)
    except ImportError:
        # Fallback: implementación manual más eficiente
        dilated = np.zeros_like(binary)
        pad = radius
        padded = np.pad(binary.astype(bool), pad, mode='constant', constant_values=False)
        
        # Usar operaciones vectorizadas donde sea posible
        for y in range(height):
            for x in range(width):
                y_start, y_end = y, y + kernel_size
                x_start, x_end = x, x + kernel_size
                neighborhood = padded[y_start:y_end, x_start:x_end]
                # Si hay algún píxel de línea donde el kernel es True, dilatar
                if np.any(neighborhood & kernel):
                    dilated[y, x] = 1
    
    # Convertir de vuelta: 1 -> blanco (255), 0 -> negro (0)
    result_array = dilated.astype(np.uint8) * 255
    
    # Si invertimos antes, invertir de vuelta
    if lines_are_white:
        result_array = 255 - result_array
    
    result_img = Image.fromarray(result_array, mode='L')
    
    # Convertir a modo 1 (blanco y negro puro)
    result_img = result_img.convert("1")
    
    return result_img


def thicken_lines(
    input_path: Path,
    percentage: float,
    output_path: Path,
) -> None:
    """Engrosa las líneas de una imagen en blanco y negro según un porcentaje.
    
    Args:
        input_path: Ruta de la imagen de entrada
        percentage: Porcentaje de engrosamiento (ej: 150 = 150%, engrosa 1.5x)
        output_path: Ruta donde guardar la imagen procesada
    """
    with Image.open(input_path) as img:
        original_format = img.format
        
        # Procesar la imagen
        result_img = thicken_lines_image(img, percentage)
        
        # Preparar para guardar
        save_kwargs = {}
        exif_bytes = img.info.get("exif")
        icc_profile = img.info.get("icc_profile")
        if exif_bytes:
            save_kwargs["exif"] = exif_bytes
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile
        
        # Para JPEG, usar alta calidad
        if original_format and original_format.upper() in {"JPEG", "JPG"}:
            save_kwargs.update({
                "quality": 95,
                "subsampling": 0,
                "optimize": True,
            })
        
        # Asegurar ruta de salida única
        output_path = ensure_unique_output_path(output_path)
        
        # Convertir a RGB si es necesario para JPEG
        to_save = result_img
        if original_format and original_format.upper() in {"JPEG", "JPG"}:
            to_save = to_save.convert("RGB")
        
        # Guardar con el formato original si es conocido
        if original_format:
            to_save.save(output_path, format=original_format, **save_kwargs)
        else:
            to_save.save(output_path, **save_kwargs)


def _circular_structure(radius: int) -> np.ndarray:
    """Elemento estructurante circular (disco) para morfología."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    k = radius * 2 + 1
    c = k // 2
    y, x = np.ogrid[:k, :k]
    return ((x - c) ** 2 + (y - c) ** 2 <= radius * radius)


def _binary_fill_holes_numpy(mask: np.ndarray) -> np.ndarray:
    """Rellena agujeros en una máscara booleana (True = tinta). Sin scipy."""
    mask = mask.astype(bool)
    if not mask.any():
        return mask.copy()
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for x in range(w):
        if not mask[0, x]:
            stack.append((0, x))
        if not mask[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if not mask[y, 0]:
            stack.append((y, 0))
        if not mask[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if visited[y, x] or mask[y, x]:
            continue
        visited[y, x] = True
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and not mask[ny, nx]:
                stack.append((ny, nx))
    holes = (~mask) & (~visited)
    filled = mask.copy()
    filled[holes] = True
    return filled


def _binary_fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
        return ndimage.binary_fill_holes(mask.astype(bool))
    except ImportError:
        return _binary_fill_holes_numpy(mask)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    struct = _circular_structure(radius)
    m = mask.astype(bool)
    try:
        from scipy import ndimage
        return ndimage.binary_dilation(m, structure=struct)
    except ImportError:
        return _binary_dilate_manual(m, struct)


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    struct = _circular_structure(radius)
    m = mask.astype(bool)
    try:
        from scipy import ndimage
        return ndimage.binary_erosion(m, structure=struct)
    except ImportError:
        return _binary_erode_manual(m, struct)


def _binary_dilate_manual(mask: np.ndarray, struct: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    kh, kw = struct.shape
    cy, cx = kh // 2, kw // 2
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(struct)
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            for dy, dx in zip(ys - cy, xs - cx):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    out[ny, nx] = True
    return out


def _binary_erode_manual(mask: np.ndarray, struct: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    kh, kw = struct.shape
    cy, cx = kh // 2, kw // 2
    out = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(struct)
    for y in range(h):
        for x in range(w):
            ok = True
            for dy, dx in zip(ys - cy, xs - cx):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w and mask[ny, nx]):
                    ok = False
                    break
            if ok:
                out[y, x] = True
    return out


def image_to_ink_mask(img: Image.Image) -> np.ndarray:
    """Máscara True donde hay tinta (líneas), independiente de fondo claro/oscuro."""
    arr = np.array(img.convert("L"), dtype=np.uint8)
    mean_value = float(np.mean(arr))
    if mean_value > 127:
        arr = 255 - arr
    return arr >= 128


def apply_outer_offset_contours(
    base_img: Image.Image,
    silhouette_source: Image.Image,
    ref_axis: str,
    ref_value_mm: float,
    active_offsets: list[tuple[float, float]],
) -> tuple[Image.Image, int]:
    """Dibuja contornos de offset alrededor de la silueta exterior rellena.

    Args:
        active_offsets: lista de (distancia_mm, grosor_mm). La distancia es
            la separación entre la silueta y el borde interno de la línea
            (línea siempre dibujada hacia afuera).

    Returns:
        (imagen RGB con contornos, padding simétrico aplicado en píxeles).
    """
    if ref_value_mm <= 0 or not active_offsets:
        return base_img.convert("RGB") if base_img.mode != "RGB" else base_img.copy(), 0

    dim_px = base_img.width if ref_axis == "width" else base_img.height
    px_per_mm = dim_px / ref_value_mm

    bands_px: list[tuple[int, int]] = []
    for dist_mm, thick_mm in active_offsets:
        r_in = max(0, int(round(dist_mm * px_per_mm)))
        r_out = int(round((dist_mm + thick_mm) * px_per_mm))
        if r_out > r_in:
            bands_px.append((r_in, r_out))
    if not bands_px:
        rgb = base_img.convert("RGB") if base_img.mode != "RGB" else base_img.copy()
        return rgb, 0

    max_r = max(r_out for _, r_out in bands_px)
    pad = max_r + 3

    mask = image_to_ink_mask(silhouette_source)
    if mask.shape != (base_img.height, base_img.width):
        raise ValueError("La silueta y la imagen base deben tener el mismo tamaño")
    solid = _binary_fill_holes(mask)

    base_rgb = base_img.convert("RGB")
    arr = np.array(base_rgb, dtype=np.uint8)
    arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="constant", constant_values=255)

    solid_p = np.pad(solid, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)

    dilation_cache: dict[int, np.ndarray] = {0: solid_p}

    def _dil(r: int) -> np.ndarray:
        if r not in dilation_cache:
            dilation_cache[r] = _binary_dilate(solid_p, r) if r > 0 else solid_p
        return dilation_cache[r]

    for r_in, r_out in sorted(bands_px, key=lambda b: b[1], reverse=True):
        outer = _dil(r_out)
        inner = _dil(r_in)
        band = outer & (~inner)
        arr[band] = (0, 0, 0)

    return Image.fromarray(arr, mode="RGB"), pad


class ResultDialog(tk.Toplevel):
    """Diálogo para mostrar y gestionar el resultado de procesar desde portapapeles."""
    
    def __init__(self, parent: tk.Tk, original_image: Image.Image, initial_percentage: float) -> None:
        super().__init__(parent)
        self.title("Resultado - Imagen procesada")
        self.original_image = original_image
        self.percentage_var = tk.DoubleVar(value=float(initial_percentage))
        self.blur_radius_var = tk.DoubleVar(value=0.0)
        self.blur_value_var = tk.StringVar(value="0.0 px")
        self.percentage_value_var = tk.StringVar()
        self.info_var = tk.StringVar()
        self._cached_thicken_percentage = None
        self._cached_blur_radius = None
        self._cached_thickened_image = None
        self._cached_blur_image = None
        self._cached_offsets_image: Optional[Image.Image] = None
        self._cached_offsets_key: Optional[tuple] = None
        
        # Offsets de contorno (silueta exterior)
        self._offset_ref_axis_var = tk.StringVar(value="Ancho")
        self._offset_ref_value_var = tk.StringVar(value="10.0")
        self._offset_ref_unit_var = tk.StringVar(value="cm")
        self._offset_dpi_var = tk.StringVar(value="300")
        self._offset_enable_vars = [tk.BooleanVar(value=False) for _ in range(2)]
        self._offset_dist_mm_vars = [
            tk.StringVar(value="1.0"),
            tk.StringVar(value="4.0"),
        ]
        self._offset_thick_mm_vars = [
            tk.StringVar(value="1.0"),
            tk.StringVar(value="2.0"),
        ]
        self._last_preview_image_size: Optional[tuple[int, int]] = None
        
        # Calcular dimensiones para la ventana
        max_preview_size = 600
        img_width, img_height = original_image.size
        
        # Recorte - inicializar cubriendo toda la imagen
        self.crop_box: Optional[tuple[int, int, int, int]] = (0, 0, img_width, img_height)
        self._crop_start: Optional[tuple[int, int]] = None  # Coordenadas de inicio del arrastre en preview
        self._crop_rect_id: Optional[int] = None  # ID del rectángulo dibujado en canvas
        self._crop_handles: list[int] = []  # IDs de los 8 handles
        self._active_handle: Optional[int] = None  # Índice del handle activo (0-7) o None si se mueve el rectángulo
        self._handle_size = 8  # Tamaño de los handles en píxeles del canvas
        self._drag_mode: Optional[str] = None  # 'handle', 'move', o None
        self._force_square_var = tk.BooleanVar(value=False)  # Forzar rectángulo cuadrado
        
        # Escalar para preview si es necesario
        if img_width > max_preview_size or img_height > max_preview_size:
            self._preview_scale = min(max_preview_size / img_width, max_preview_size / img_height)
            preview_width = int(img_width * self._preview_scale)
            preview_height = int(img_height * self._preview_scale)
        else:
            self._preview_scale = 1.0
            preview_width = img_width
            preview_height = img_height
        
        self._preview_size = (preview_width, preview_height)
        self._original_size = (img_width, img_height)
        self._base_scale = self._preview_scale  # Escala base calculada para que la imagen quepa
        self._zoom_level = 1.0  # Nivel de zoom manual (1.0 = 100%, 0.5 = 50%, etc.)
        self._display_scale = self._preview_scale * self._zoom_level  # Escala de visualización actual
        self._zoom_label_var = tk.StringVar(value="100%")  # Label para mostrar el nivel de zoom
        self._total_origin = (0, 0)  # Origen del área total de visualización
        self._img_offset = (0, 0)  # Offset de la imagen en el canvas
        self._center_offset = (10, 10)  # Offset de centrado en el canvas
        self._visible_bounds = (0, 0, img_width, img_height)  # Límites visibles en coordenadas de imagen
        
        self._build_ui()
        self._update_preview()
        
        # Configurar ventana
        self.update_idletasks()
        window_width = max(preview_width + 40, self.winfo_reqwidth())
        window_height = max(preview_height + 40, self.winfo_reqheight())
        self.geometry(f"{window_width}x{window_height}")
        self.resizable(True, True)
        self.minsize(min(520, window_width), min(400, window_height))
        
        # Centrar en la pantalla
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.winfo_screenheight() // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Hacer modal
        self.transient(parent)
        self.grab_set()
    
    def _build_ui(self) -> None:
        # Contenedor con scrollbar vertical para que el contenido sea desplazable
        # cuando no quepa en la altura de la ventana.
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        
        self._scroll_canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        self._scroll_canvas.pack(side="left", fill="both", expand=True)
        
        inner = ttk.Frame(self._scroll_canvas)
        self._inner_frame = inner
        self._inner_window_id = self._scroll_canvas.create_window(
            (0, 0), window=inner, anchor="nw"
        )
        
        def _on_inner_configure(_e: object) -> None:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)
        
        def _on_canvas_configure(e: tk.Event) -> None:
            # Hacer que el frame interior siga el ancho del canvas (scroll solo vertical).
            self._scroll_canvas.itemconfigure(self._inner_window_id, width=e.width)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)
        
        # Rueda del ratón: activa scroll solo cuando el cursor está fuera del preview
        # para no interferir con interacciones del lienzo de imagen.
        def _on_mousewheel(e: tk.Event) -> str:
            self._scroll_canvas.yview_scroll(int(-e.delta / 120), "units")
            return "break"
        def _bind_wheel(_e: object) -> None:
            self._scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        def _unbind_wheel(_e: object) -> None:
            self._scroll_canvas.unbind_all("<MouseWheel>")
        self._scroll_canvas.bind("<Enter>", _bind_wheel)
        self._scroll_canvas.bind("<Leave>", _unbind_wheel)
        self.bind("<Destroy>", lambda _e: self._scroll_canvas.unbind_all("<MouseWheel>"))
        
        # Información
        info_frame = ttk.Frame(inner)
        info_frame.pack(fill="x", padx=10, pady=(10, 5))
        
        ttk.Label(
            info_frame,
            textvariable=self.info_var,
            font=("TkDefaultFont", 9, "bold")
        ).pack()
        
        # Preview de la imagen
        preview_frame = ttk.LabelFrame(inner, text="Vista previa (arrastra para seleccionar, usa los puntos azules para ajustar)")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Usar Canvas en lugar de Label para permitir selección de área
        # El tamaño se ajustará dinámicamente en _update_preview
        self.preview_canvas = tk.Canvas(
            preview_frame,
            cursor="crosshair",
            bg="white"
        )
        self.preview_canvas.pack(padx=10, pady=10)
        
        # Bind eventos de mouse para selección
        self.preview_canvas.bind("<Button-1>", self._on_crop_start)
        self.preview_canvas.bind("<B1-Motion>", self._on_crop_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_crop_end)
        self.preview_canvas.bind("<Motion>", self._on_crop_motion)
        
        # Controles de engrosamiento
        thicken_frame = ttk.LabelFrame(inner, text="Engrosamiento")
        thicken_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(thicken_frame, text="Porcentaje:").grid(
            row=0, column=0, sticky="w", padx=10, pady=8
        )
        ttk.Scale(
            thicken_frame,
            from_=0.0,
            to=200.0,
            orient="horizontal",
            variable=self.percentage_var,
            command=self._on_thicken_change,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=8)
        ttk.Label(
            thicken_frame,
            textvariable=self.percentage_value_var,
            width=8,
            anchor="e"
        ).grid(row=0, column=2, sticky="e", padx=(0, 10), pady=8)
        thicken_frame.columnconfigure(1, weight=1)
        
        # Controles de desenfoque
        blur_frame = ttk.LabelFrame(inner, text="Desenfoque gaussiano")
        blur_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(blur_frame, text="Radio (px):").grid(
            row=0, column=0, sticky="w", padx=10, pady=8
        )
        ttk.Scale(
            blur_frame,
            from_=0.0,
            to=10.0,
            orient="horizontal",
            variable=self.blur_radius_var,
            command=self._on_blur_change,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=8)
        ttk.Label(
            blur_frame,
            textvariable=self.blur_value_var,
            width=8,
            anchor="e"
        ).grid(row=0, column=2, sticky="e", padx=(0, 10), pady=8)
        blur_frame.columnconfigure(1, weight=1)
        
        # Offsets de contorno (silueta exterior)
        offset_frame = ttk.LabelFrame(inner, text="Offsets de contorno (silueta exterior)")
        offset_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(offset_frame, text="Escala física:").grid(
            row=0, column=0, sticky="w", padx=10, pady=4
        )
        axis_cb = ttk.Combobox(
            offset_frame,
            textvariable=self._offset_ref_axis_var,
            values=("Ancho", "Alto"),
            width=8,
            state="readonly",
        )
        axis_cb.grid(row=0, column=1, sticky="w", padx=(0, 6), pady=4)
        axis_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_offset_change())
        
        ref_entry = ttk.Entry(offset_frame, textvariable=self._offset_ref_value_var, width=8)
        ref_entry.grid(row=0, column=2, sticky="w", padx=(0, 6), pady=4)
        ref_entry.bind("<KeyRelease>", lambda _e: self._on_offset_change())
        
        unit_cb = ttk.Combobox(
            offset_frame,
            textvariable=self._offset_ref_unit_var,
            values=("cm", "mm"),
            width=5,
            state="readonly",
        )
        unit_cb.grid(row=0, column=3, sticky="w", padx=(0, 6), pady=4)
        unit_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_offset_change())
        
        ttk.Label(offset_frame, text="DPI ref.:").grid(row=0, column=4, sticky="e", padx=(8, 4), pady=4)
        dpi_entry = ttk.Entry(offset_frame, textvariable=self._offset_dpi_var, width=6)
        dpi_entry.grid(row=0, column=5, sticky="w", padx=(0, 10), pady=4)
        dpi_entry.bind("<KeyRelease>", lambda _e: self._on_offset_change())
        
        # Cabecera de columnas para los offsets
        ttk.Label(offset_frame, text="Distancia").grid(row=1, column=2, sticky="w", padx=(0, 4), pady=(4, 0))
        ttk.Label(offset_frame, text="Grosor").grid(row=1, column=4, sticky="w", padx=(0, 4), pady=(4, 0))

        for i in range(2):
            row = 2 + i
            cb = ttk.Checkbutton(
                offset_frame,
                text=f"Offset {i + 1}:",
                variable=self._offset_enable_vars[i],
                command=self._on_offset_change,
            )
            cb.grid(row=row, column=0, sticky="w", padx=10, pady=2, columnspan=2)

            dist_entry = ttk.Entry(offset_frame, textvariable=self._offset_dist_mm_vars[i], width=8)
            dist_entry.grid(row=row, column=2, sticky="w", padx=(0, 4), pady=2)
            dist_entry.bind("<KeyRelease>", lambda _e: self._on_offset_change())
            ttk.Label(offset_frame, text="mm").grid(row=row, column=3, sticky="w", padx=(0, 10), pady=2)

            thick_entry = ttk.Entry(offset_frame, textvariable=self._offset_thick_mm_vars[i], width=8)
            thick_entry.grid(row=row, column=4, sticky="w", padx=(0, 4), pady=2)
            thick_entry.bind("<KeyRelease>", lambda _e: self._on_offset_change())
            ttk.Label(offset_frame, text="mm").grid(row=row, column=5, sticky="w", padx=(0, 10), pady=2)

        offset_frame.columnconfigure(6, weight=1)
        
        # Controles de recorte y zoom
        crop_frame = ttk.Frame(inner)
        crop_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Checkbutton(
            crop_frame,
            text="Forzar cuadrado",
            variable=self._force_square_var,
            command=self._on_force_square_change
        ).pack(side="left", padx=(10, 5))
        
        ttk.Button(
            crop_frame,
            text="Restablecer",
            command=self._clear_crop
        ).pack(side="left", padx=(5, 10))
        
        # Separador visual
        ttk.Separator(crop_frame, orient="vertical").pack(side="left", fill="y", padx=5)
        
        # Controles de zoom
        ttk.Label(crop_frame, text="Zoom:").pack(side="left", padx=(5, 2))
        
        ttk.Button(
            crop_frame,
            text="-",
            width=3,
            command=self._zoom_out
        ).pack(side="left", padx=2)
        
        ttk.Label(
            crop_frame,
            textvariable=self._zoom_label_var,
            width=6,
            anchor="center"
        ).pack(side="left", padx=2)
        
        ttk.Button(
            crop_frame,
            text="+",
            width=3,
            command=self._zoom_in
        ).pack(side="left", padx=2)
        
        ttk.Button(
            crop_frame,
            text="Reset",
            command=self._zoom_reset
        ).pack(side="left", padx=(2, 10))
        
        # Botones de acción
        btn_frame = ttk.Frame(inner)
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))
        
        ttk.Button(
            btn_frame,
            text="📋 Copiar al portapapeles",
            command=self._copy_to_clipboard
        ).pack(side="left", padx=(0, 5), expand=True, fill="x")
        
        ttk.Button(
            btn_frame,
            text="💾 Guardar como...",
            command=self._save_as
        ).pack(side="left", padx=5, expand=True, fill="x")
        
        ttk.Button(
            btn_frame,
            text="Cerrar",
            command=self.destroy
        ).pack(side="left", padx=(5, 0), expand=True, fill="x")
    
    def _get_thicken_percentage(self) -> float:
        try:
            return max(0.0, float(self.percentage_var.get()))
        except (TypeError, ValueError):
            return 0.0
    
    def _get_blur_radius(self) -> float:
        try:
            return round(float(self.blur_radius_var.get()), 1)
        except (TypeError, ValueError):
            return 0.0
    
    def _on_offset_change(self, _event: object = None) -> None:
        self._cached_offsets_image = None
        self._cached_offsets_key = None
        self._update_preview()
    
    def _parse_offset_params(self) -> dict:
        """Parámetros de offsets para caché e info. ref_mm = longitud física del eje en mm."""
        try:
            ref_val = float(str(self._offset_ref_value_var.get()).replace(",", "."))
        except (TypeError, ValueError):
            ref_val = 0.0
        unit = str(self._offset_ref_unit_var.get()).strip().lower()
        ref_mm = ref_val * 10.0 if unit == "cm" else ref_val
        axis = "width" if self._offset_ref_axis_var.get() == "Ancho" else "height"
        
        active: list[tuple[float, float]] = []
        for i in range(2):
            if not self._offset_enable_vars[i].get():
                continue
            try:
                dist_mm = float(str(self._offset_dist_mm_vars[i].get()).replace(",", "."))
            except (TypeError, ValueError):
                continue
            try:
                thick_mm = float(str(self._offset_thick_mm_vars[i].get()).replace(",", "."))
            except (TypeError, ValueError):
                continue
            if dist_mm >= 0 and thick_mm > 0:
                active.append((dist_mm, thick_mm))
        
        dpi_str = str(self._offset_dpi_var.get()).strip()
        try:
            dpi = float(dpi_str.replace(",", ".")) if dpi_str else 300.0
        except (TypeError, ValueError):
            dpi = 300.0
        
        key_tuple = (axis, ref_mm, tuple(active), dpi, ref_val, unit)
        return {
            "axis": axis,
            "ref_mm": ref_mm,
            "ref_val_raw": ref_val,
            "unit": unit,
            "active_list": active,
            "dpi": dpi,
            "key_tuple": key_tuple,
            "ref_valid": ref_mm > 0,
        }
    
    def _apply_offset_overlay_if_needed(
        self,
        base: Image.Image,
        thickened: Image.Image,
        percentage: float,
        radius: float,
    ) -> Image.Image:
        params = self._parse_offset_params()
        active = params["active_list"]
        if not active or not params["ref_valid"]:
            return base
        
        if thickened.size != base.size:
            thickened = thickened.resize(base.size, Image.Resampling.NEAREST)
        
        key = (percentage, radius, params["key_tuple"], base.size)
        if self._cached_offsets_image is not None and self._cached_offsets_key == key:
            return self._cached_offsets_image
        
        try:
            out, _pad = apply_outer_offset_contours(
                base_img=base,
                silhouette_source=thickened,
                ref_axis=params["axis"],
                ref_value_mm=params["ref_mm"],
                active_offsets=active,
            )
        except Exception:
            return base
        
        self._cached_offsets_image = out
        self._cached_offsets_key = key
        return out
    
    def _sync_crop_after_image_resize(
        self,
        prev_size: tuple[int, int],
        new_size: tuple[int, int],
    ) -> None:
        """Mantiene el recorte coherente al cambiar el tamaño (p. ej. padding de offsets)."""
        if self.crop_box is None:
            return
        x1, y1, x2, y2 = self.crop_box
        pw, ph = prev_size
        nw, nh = new_size
        if x1 == 0 and y1 == 0 and x2 == pw and y2 == ph:
            self.crop_box = (0, 0, nw, nh)
            return
        dw = (nw - pw) // 2
        dh = (nh - ph) // 2
        if dw == dh:
            self.crop_box = (x1 + dw, y1 + dh, x2 + dw, y2 + dh)
        else:
            self.crop_box = (0, 0, nw, nh)
    
    def _update_info_text(self, percentage: float, radius: float) -> None:
        crop_info = ""
        if self.crop_box is not None:
            x1, y1, x2, y2 = self.crop_box
            width = x2 - x1
            height = y2 - y1
            crop_info = f" | Recorte: {width}×{height}px"
        
        p = self._parse_offset_params()
        off_info = ""
        if p["active_list"]:
            if p["ref_valid"]:
                dim_px = (
                    self.original_image.width if p["axis"] == "width" else self.original_image.height
                )
                px_per_mm = dim_px / p["ref_mm"]
                dpi_eff = px_per_mm * 25.4
                off_str = ", ".join(
                    f"{d:g}mm@{t:g}mm" for d, t in p["active_list"]
                )
                off_info = (
                    f" | Offsets (dist@grosor): {off_str} | "
                    f"px/mm≈{px_per_mm:.3f} (~{dpi_eff:.0f} dpi) | DPI ref: {p['dpi']:.0f}"
                )
            else:
                off_info = " | Offsets activos: indica dimensión física > 0"
        
        info_text = (
            f"Imagen original: {self.original_image.width}×{self.original_image.height}px | "
            f"Engrosamiento: {percentage:.0f}% | "
            f"Desenfoque: {radius:.1f}px{off_info}{crop_info}"
        )
        self.info_var.set(info_text)
    
    def _get_current_image(
        self, 
        percentage: Optional[float] = None, 
        radius: Optional[float] = None
    ) -> Image.Image:
        if percentage is None:
            percentage = self._get_thicken_percentage()
        if radius is None:
            radius = self._get_blur_radius()
        
        # Aplicar engrosamiento primero
        if percentage <= 0:
            thickened = self.original_image
            self._cached_thicken_percentage = 0.0
            self._cached_thickened_image = None
        else:
            if (
                self._cached_thickened_image is not None
                and self._cached_thicken_percentage == percentage
            ):
                thickened = self._cached_thickened_image
            else:
                thickened = thicken_lines_image(self.original_image, percentage)
                self._cached_thicken_percentage = percentage
                self._cached_thickened_image = thickened
        
        # Desenfoque (opcional)
        if radius <= 0:
            base = thickened
            self._cached_blur_image = None
            self._cached_blur_radius = None
        else:
            if (
                self._cached_blur_image is not None
                and self._cached_blur_radius == radius
                and self._cached_thicken_percentage == percentage
            ):
                base = self._cached_blur_image
            else:
                b = thickened
                if b.mode == "1":
                    b = b.convert("L")
                blurred = b.filter(ImageFilter.GaussianBlur(radius))
                self._cached_blur_radius = radius
                self._cached_blur_image = blurred
                base = blurred
        
        return self._apply_offset_overlay_if_needed(base, thickened, percentage, radius)
    
    def _update_preview(self) -> None:
        percentage = self._get_thicken_percentage()
        radius = self._get_blur_radius()
        
        self.percentage_value_var.set(f"{percentage:.0f}%")
        self.blur_value_var.set(f"{radius:.1f} px")
        self._update_info_text(percentage, radius)
        
        prev_size = self._last_preview_image_size
        current_img = self._get_current_image(percentage, radius)
        new_size = (current_img.width, current_img.height)
        if prev_size is not None and prev_size != new_size:
            self._sync_crop_after_image_resize(prev_size, new_size)
        self._last_preview_image_size = new_size
        
        # Si el recorte sigue siendo "toda la imagen original" pero la salida es más grande (offsets), ampliar.
        ow, oh = self._original_size
        if self.crop_box is not None:
            cx1, cy1, cx2, cy2 = self.crop_box
            if cx1 == 0 and cy1 == 0 and cx2 == ow and cy2 == oh:
                if cx2 != new_size[0] or cy2 != new_size[1]:
                    self.crop_box = (0, 0, new_size[0], new_size[1])
        
        # Usar zoom manual: escala base multiplicada por el nivel de zoom
        img_width, img_height = current_img.width, current_img.height
        self._current_display_size = (img_width, img_height)
        display_scale = self._base_scale * self._zoom_level
        
        # Tamaño del canvas: cabe la imagen original o la procesada (p. ej. con padding de offsets)
        base_canvas_width = int(max(ow, img_width) * self._base_scale) + 20
        base_canvas_height = int(max(oh, img_height) * self._base_scale) + 20
        canvas_width = max(base_canvas_width, 100)
        canvas_height = max(base_canvas_height, 100)
        self.preview_canvas.config(width=canvas_width, height=canvas_height)
        
        # Calcular el área visible en coordenadas de imagen original
        # El zoom determina cuánta área cabe en el canvas
        visible_width_img = (canvas_width - 20) / display_scale
        visible_height_img = (canvas_height - 20) / display_scale
        
        # Centrar la imagen en el área visible
        # El área visible va desde -extra hasta img_size+extra
        extra_x = (visible_width_img - img_width) / 2
        extra_y = (visible_height_img - img_height) / 2
        
        # Límites del área visible (en coordenadas de imagen)
        visible_x1 = -extra_x if extra_x > 0 else 0
        visible_y1 = -extra_y if extra_y > 0 else 0
        visible_x2 = img_width + extra_x if extra_x > 0 else img_width
        visible_y2 = img_height + extra_y if extra_y > 0 else img_height
        
        # Guardar los límites para usarlos en las restricciones del crop_box
        self._visible_bounds = (visible_x1, visible_y1, visible_x2, visible_y2)
        
        # Guardar la escala de visualización para conversión de coordenadas
        self._display_scale = display_scale
        
        # Calcular tamaño de la imagen escalada
        img_preview_width = max(1, int(img_width * display_scale))
        img_preview_height = max(1, int(img_height * display_scale))
        
        # Calcular offset de centrado (para centrar la imagen)
        center_offset_x = (canvas_width - img_preview_width) // 2
        center_offset_y = (canvas_height - img_preview_height) // 2
        
        # Guardar el offset de centrado para conversión de coordenadas
        self._center_offset = (max(10, center_offset_x), max(10, center_offset_y))
        self._img_offset = (0, 0)  # Ya no necesitamos offset de imagen
        self._total_origin = (0, 0)  # El origen siempre es (0,0) de la imagen
        
        # Redimensionar la imagen para el preview
        preview_img = current_img
        if img_preview_width != img_width or img_preview_height != img_height:
            preview_img = current_img.resize((img_preview_width, img_preview_height), Image.Resampling.LANCZOS)
        
        if preview_img.mode != "RGB":
            preview_img = preview_img.convert("RGB")
        
        self.photo = ImageTk.PhotoImage(preview_img)
        
        # Limpiar canvas
        self.preview_canvas.delete("all")
        
        # Dibujar fondo de cuadros en el área visible fuera de la imagen
        if extra_x > 0 or extra_y > 0:
            # Dibujar checkerboard en toda el área visible
            self._draw_checkerboard_background(
                10, 10, canvas_width - 10, canvas_height - 10
            )
        
        # Dibujar imagen centrada en el canvas
        img_canvas_x = self._center_offset[0]
        img_canvas_y = self._center_offset[1]
        self.preview_canvas.create_image(img_canvas_x, img_canvas_y, anchor="nw", image=self.photo)
        
        # Dibujar rectángulo de recorte si existe
        if self.crop_box is not None:
            self._draw_crop_rect()
    
    def _on_thicken_change(self, _value: str) -> None:
        # Invalidar caché cuando cambia el engrosamiento
        self._cached_blur_image = None
        self._cached_offsets_image = None
        self._cached_offsets_key = None
        self._update_preview()
    
    def _on_blur_change(self, _value: str) -> None:
        self._cached_offsets_image = None
        self._cached_offsets_key = None
        self._update_preview()
    
    def _zoom_out(self) -> None:
        """Reduce el nivel de zoom."""
        # Reducir en pasos de 25%
        new_zoom = max(0.1, self._zoom_level - 0.25)
        self._zoom_level = new_zoom
        self._zoom_label_var.set(f"{int(new_zoom * 100)}%")
        self._update_preview()
    
    def _zoom_in(self) -> None:
        """Aumenta el nivel de zoom."""
        # Aumentar en pasos de 25%
        new_zoom = min(2.0, self._zoom_level + 0.25)
        self._zoom_level = new_zoom
        self._zoom_label_var.set(f"{int(new_zoom * 100)}%")
        self._update_preview()
    
    def _zoom_reset(self) -> None:
        """Resetea el zoom a 100%."""
        self._zoom_level = 1.0
        self._zoom_label_var.set("100%")
        self._update_preview()
    
    def _on_force_square_change(self) -> None:
        """Se llama cuando se activa/desactiva el switch de forzar cuadrado."""
        if self.crop_box is not None and self._force_square_var.get():
            # Si hay un rectángulo y se activa el switch, hacerlo cuadrado
            x1, y1, x2, y2 = self.crop_box
            # Mantener la esquina superior izquierda como anchor
            x1, y1, x2, y2 = self._make_square(x1, y1, x2, y2, anchor_handle=0)
            self.crop_box = (x1, y1, x2, y2)
            self._draw_crop_rect()
            # Actualizar información del texto
            percentage = self._get_thicken_percentage()
            radius = self._get_blur_radius()
            self._update_info_text(percentage, radius)
    
    def _clear_crop(self) -> None:
        """Restablece la selección de recorte a toda la imagen."""
        cur = self._get_current_image()
        self.crop_box = (0, 0, cur.width, cur.height)
        self._crop_start = None
        self._active_handle = None
        self._drag_mode = None
        self._crop_handles = []
        self.preview_canvas.delete("crop_rect")
        self._crop_rect_id = None
        self.preview_canvas.config(cursor="crosshair")
        self._update_preview()
    
    def _preview_to_original_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convierte coordenadas del canvas (preview) a coordenadas de la imagen original."""
        # Usar el offset de centrado dinámico
        center_offset = getattr(self, '_center_offset', (10, 10))
        canvas_x = x - center_offset[0]
        canvas_y = y - center_offset[1]
        
        # Usar la escala de visualización actual
        scale = getattr(self, '_display_scale', self._preview_scale)
        
        # Escalar de vuelta a la imagen original
        orig_x = int(canvas_x / scale)
        orig_y = int(canvas_y / scale)
        
        return (orig_x, orig_y)
    
    def _get_handle_positions(self) -> list[tuple[int, int]]:
        """Retorna las posiciones de los 8 handles en coordenadas del canvas."""
        if self.crop_box is None:
            return []
        
        x1, y1, x2, y2 = self.crop_box
        # Usar la escala de visualización actual
        scale = getattr(self, '_display_scale', self._preview_scale)
        # Usar el offset de centrado dinámico
        center_offset = getattr(self, '_center_offset', (10, 10))
        
        canvas_x1 = int(x1 * scale) + center_offset[0]
        canvas_y1 = int(y1 * scale) + center_offset[1]
        canvas_x2 = int(x2 * scale) + center_offset[0]
        canvas_y2 = int(y2 * scale) + center_offset[1]
        
        cx = (canvas_x1 + canvas_x2) // 2
        cy = (canvas_y1 + canvas_y2) // 2
        
        # 8 handles: esquinas (0-3) y bordes (4-7)
        # 0: esquina superior izquierda
        # 1: esquina superior derecha
        # 2: esquina inferior derecha
        # 3: esquina inferior izquierda
        # 4: borde superior
        # 5: borde derecho
        # 6: borde inferior
        # 7: borde izquierdo
        return [
            (canvas_x1, canvas_y1),  # 0: esquina superior izquierda
            (canvas_x2, canvas_y1),  # 1: esquina superior derecha
            (canvas_x2, canvas_y2),  # 2: esquina inferior derecha
            (canvas_x1, canvas_y2),  # 3: esquina inferior izquierda
            (cx, canvas_y1),         # 4: borde superior
            (canvas_x2, cy),         # 5: borde derecho
            (cx, canvas_y2),         # 6: borde inferior
            (canvas_x1, cy),         # 7: borde izquierdo
        ]
    
    def _get_handle_at_position(self, x: int, y: int) -> Optional[int]:
        """Retorna el índice del handle en la posición (x, y) o None."""
        if self.crop_box is None:
            return None
        
        handle_positions = self._get_handle_positions()
        half_size = self._handle_size // 2
        
        for i, (hx, hy) in enumerate(handle_positions):
            if abs(x - hx) <= half_size and abs(y - hy) <= half_size:
                return i
        
        return None
    
    def _is_point_in_crop_rect(self, x: int, y: int) -> bool:
        """Verifica si el punto está dentro del rectángulo de recorte (pero no en un handle)."""
        if self.crop_box is None:
            return False
        
        x1, y1, x2, y2 = self.crop_box
        # Usar la escala de visualización actual
        scale = getattr(self, '_display_scale', self._preview_scale)
        # Usar el offset de centrado dinámico
        center_offset = getattr(self, '_center_offset', (10, 10))
        
        canvas_x1 = int(x1 * scale) + center_offset[0]
        canvas_y1 = int(y1 * scale) + center_offset[1]
        canvas_x2 = int(x2 * scale) + center_offset[0]
        canvas_y2 = int(y2 * scale) + center_offset[1]
        
        return (canvas_x1 <= x <= canvas_x2 and canvas_y1 <= y <= canvas_y2)
    
    def _get_cursor_for_handle(self, handle_idx: Optional[int]) -> str:
        """Retorna el cursor apropiado para el handle."""
        if handle_idx is None:
            return "crosshair"
        
        cursors = [
            "top_left_corner",      # 0: esquina superior izquierda
            "top_right_corner",      # 1: esquina superior derecha
            "bottom_right_corner",   # 2: esquina inferior derecha
            "bottom_left_corner",    # 3: esquina inferior izquierda
            "top_side",              # 4: borde superior
            "right_side",            # 5: borde derecho
            "bottom_side",           # 6: borde inferior
            "left_side",             # 7: borde izquierdo
        ]
        return cursors[handle_idx] if handle_idx < len(cursors) else "crosshair"
    
    def _on_crop_motion(self, event) -> None:
        """Actualiza el cursor cuando el mouse se mueve sobre el canvas."""
        if self.crop_box is None:
            self.preview_canvas.config(cursor="crosshair")
            return
        
        handle_idx = self._get_handle_at_position(event.x, event.y)
        if handle_idx is not None:
            self.preview_canvas.config(cursor=self._get_cursor_for_handle(handle_idx))
        elif self._is_point_in_crop_rect(event.x, event.y):
            self.preview_canvas.config(cursor="fleur")  # Cursor para mover
        else:
            self.preview_canvas.config(cursor="crosshair")
    
    def _on_crop_start(self, event) -> None:
        """Inicia la selección o ajuste de área de recorte."""
        x, y = event.x, event.y
        
        # Verificar si hay un handle bajo el mouse
        handle_idx = self._get_handle_at_position(x, y)
        
        # Obtener límites visibles
        vb = getattr(self, '_visible_bounds', (0, 0, self._original_size[0], self._original_size[1]))
        
        if handle_idx is not None and self.crop_box is not None:
            # Ajustar handle existente
            self._active_handle = handle_idx
            self._drag_mode = "handle"
            orig_x, orig_y = self._preview_to_original_coords(x, y)
            self._crop_start = (orig_x, orig_y)
        elif self._is_point_in_crop_rect(x, y) and self.crop_box is not None:
            # Mover el rectángulo completo
            self._active_handle = None
            self._drag_mode = "move"
            orig_x, orig_y = self._preview_to_original_coords(x, y)
            x1, y1, x2, y2 = self.crop_box
            # Guardar el offset desde la esquina superior izquierda
            self._crop_start = (orig_x - x1, orig_y - y1)
        else:
            # Crear nuevo rectángulo - limitar a bounds visibles
            self._active_handle = None
            self._drag_mode = None
            orig_x, orig_y = self._preview_to_original_coords(x, y)
            # Limitar a los bounds visibles
            orig_x = max(vb[0], min(vb[2], orig_x))
            orig_y = max(vb[1], min(vb[3], orig_y))
            self._crop_start = (orig_x, orig_y)
            self.crop_box = None
    
    def _make_square(self, x1: int, y1: int, x2: int, y2: int, anchor_handle: Optional[int] = None) -> tuple[int, int, int, int]:
        """Fuerza el rectángulo a ser cuadrado manteniendo el anchor apropiado."""
        width = x2 - x1
        height = y2 - y1
        size = max(width, height, 1)  # Usar la dimensión mayor, mínimo 1 píxel
        
        # Obtener límites visibles
        vb = getattr(self, '_visible_bounds', (0, 0, self._original_size[0], self._original_size[1]))
        max_size_x = vb[2] - vb[0]
        max_size_y = vb[3] - vb[1]
        size = min(size, max_size_x, max_size_y)  # Limitar al tamaño máximo visible
        
        # Determinar qué esquina mantener fija según el handle o el centro
        if anchor_handle == 0:  # Esquina superior izquierda
            x2 = x1 + size
            y2 = y1 + size
        elif anchor_handle == 1:  # Esquina superior derecha
            x1 = x2 - size
            y2 = y1 + size
        elif anchor_handle == 2:  # Esquina inferior derecha
            x1 = x2 - size
            y1 = y2 - size
        elif anchor_handle == 3:  # Esquina inferior izquierda
            x2 = x1 + size
            y1 = y2 - size
        elif anchor_handle == 4:  # Borde superior - mantener centro horizontal
            center_x = (x1 + x2) // 2
            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y2 = y1 + size
        elif anchor_handle == 5:  # Borde derecho - mantener centro vertical
            center_y = (y1 + y2) // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2
            x1 = x2 - size
        elif anchor_handle == 6:  # Borde inferior - mantener centro horizontal
            center_x = (x1 + x2) // 2
            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y1 = y2 - size
        elif anchor_handle == 7:  # Borde izquierdo - mantener centro vertical
            center_y = (y1 + y2) // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2
            x2 = x1 + size
        else:
            # Por defecto, mantener la esquina superior izquierda
            x2 = x1 + size
            y2 = y1 + size
        
        # Limitar a los bounds visibles
        if x1 < vb[0]:
            x1, x2 = vb[0], vb[0] + size
        if x2 > vb[2]:
            x1, x2 = vb[2] - size, vb[2]
        if y1 < vb[1]:
            y1, y2 = vb[1], vb[1] + size
        if y2 > vb[3]:
            y1, y2 = vb[3] - size, vb[3]
        
        return (x1, y1, x2, y2)
    
    def _adjust_crop_box_by_handle(self, handle_idx: int, new_x: int, new_y: int) -> tuple[int, int, int, int]:
        """Ajusta el crop_box según el handle arrastrado. Limitado por los bounds visibles."""
        x1, y1, x2, y2 = self.crop_box
        
        # Obtener límites visibles
        vb = getattr(self, '_visible_bounds', (0, 0, self._original_size[0], self._original_size[1]))
        
        # Limitar las nuevas coordenadas a los límites visibles
        new_x = max(vb[0], min(vb[2], new_x))
        new_y = max(vb[1], min(vb[3], new_y))
        
        if handle_idx == 0:  # Esquina superior izquierda
            x1, y1 = new_x, new_y
        elif handle_idx == 1:  # Esquina superior derecha
            x2, y1 = new_x, new_y
        elif handle_idx == 2:  # Esquina inferior derecha
            x2, y2 = new_x, new_y
        elif handle_idx == 3:  # Esquina inferior izquierda
            x1, y2 = new_x, new_y
        elif handle_idx == 4:  # Borde superior
            y1 = new_y
        elif handle_idx == 5:  # Borde derecho
            x2 = new_x
        elif handle_idx == 6:  # Borde inferior
            y2 = new_y
        elif handle_idx == 7:  # Borde izquierdo
            x1 = new_x
        
        # Asegurar que el rectángulo tenga al menos 1x1 píxel
        if x2 <= x1:
            if handle_idx in [0, 7]:  # Ajustar desde la izquierda
                x1 = x2 - 1
            else:
                x2 = x1 + 1
        
        if y2 <= y1:
            if handle_idx in [0, 4]:  # Ajustar desde arriba
                y1 = y2 - 1
            else:
                y2 = y1 + 1
        
        # Si está activado "forzar cuadrado", ajustar
        if self._force_square_var.get():
            x1, y1, x2, y2 = self._make_square(x1, y1, x2, y2, anchor_handle=handle_idx)
        
        return (x1, y1, x2, y2)
    
    def _on_crop_drag(self, event) -> None:
        """Actualiza el rectángulo de selección mientras se arrastra."""
        if self._crop_start is None:
            return
        
        x, y = self._preview_to_original_coords(event.x, event.y)
        
        # Obtener límites visibles
        vb = getattr(self, '_visible_bounds', (0, 0, self._original_size[0], self._original_size[1]))
        
        # Limitar coordenadas a los límites visibles
        x = max(vb[0], min(vb[2], x))
        y = max(vb[1], min(vb[3], y))
        
        if self._drag_mode == "handle" and self._active_handle is not None and self.crop_box is not None:
            # Ajustar handle
            self.crop_box = self._adjust_crop_box_by_handle(self._active_handle, x, y)
        elif self._drag_mode == "move" and self.crop_box is not None:
            # Mover rectángulo completo
            offset_x, offset_y = self._crop_start
            x1, y1, x2, y2 = self.crop_box
            width = x2 - x1
            height = y2 - y1
            
            new_x1 = x - offset_x
            new_y1 = y - offset_y
            new_x2 = new_x1 + width
            new_y2 = new_y1 + height
            
            # Limitar el movimiento a los límites visibles
            if new_x1 < vb[0]:
                new_x1, new_x2 = vb[0], vb[0] + width
            if new_x2 > vb[2]:
                new_x1, new_x2 = vb[2] - width, vb[2]
            if new_y1 < vb[1]:
                new_y1, new_y2 = vb[1], vb[1] + height
            if new_y2 > vb[3]:
                new_y1, new_y2 = vb[3] - height, vb[3]
            
            self.crop_box = (new_x1, new_y1, new_x2, new_y2)
        else:
            # Crear nuevo rectángulo
            start_x, start_y = self._crop_start
            x1 = min(start_x, x)
            y1 = min(start_y, y)
            x2 = max(start_x, x)
            y2 = max(start_y, y)
            
            # Si está activado "forzar cuadrado", ajustar
            if self._force_square_var.get():
                # Determinar qué esquina es el punto de inicio
                if x >= start_x and y >= start_y:
                    # Arrastrando hacia abajo-derecha: anchor esquina superior izquierda (0)
                    anchor = 0
                elif x < start_x and y >= start_y:
                    # Arrastrando hacia abajo-izquierda: anchor esquina superior derecha (1)
                    anchor = 1
                elif x < start_x and y < start_y:
                    # Arrastrando hacia arriba-izquierda: anchor esquina inferior derecha (2)
                    anchor = 2
                else:  # x >= start_x and y < start_y
                    # Arrastrando hacia arriba-derecha: anchor esquina inferior izquierda (3)
                    anchor = 3
                x1, y1, x2, y2 = self._make_square(x1, y1, x2, y2, anchor_handle=anchor)
            
            self.crop_box = (x1, y1, x2, y2)
        
        self._update_preview()
    
    def _on_crop_end(self, event) -> None:
        """Finaliza la selección de área de recorte."""
        if self._crop_start is None:
            return
        
        if self._drag_mode is None:
            # Crear nuevo rectángulo
            x, y = self._preview_to_original_coords(event.x, event.y)
            start_x, start_y = self._crop_start
            
            x1 = min(start_x, x)
            y1 = min(start_y, y)
            x2 = max(start_x, x)
            y2 = max(start_y, y)
            
            # Si está activado "forzar cuadrado", ajustar
            if self._force_square_var.get():
                # Determinar qué esquina es el punto de inicio
                if x >= start_x and y >= start_y:
                    # Arrastrando hacia abajo-derecha: anchor esquina superior izquierda (0)
                    anchor = 0
                elif x < start_x and y >= start_y:
                    # Arrastrando hacia abajo-izquierda: anchor esquina superior derecha (1)
                    anchor = 1
                elif x < start_x and y < start_y:
                    # Arrastrando hacia arriba-izquierda: anchor esquina inferior derecha (2)
                    anchor = 2
                else:  # x >= start_x and y < start_y
                    # Arrastrando hacia arriba-derecha: anchor esquina inferior izquierda (3)
                    anchor = 3
                x1, y1, x2, y2 = self._make_square(x1, y1, x2, y2, anchor_handle=anchor)
            
            # Solo guardar si el área es válida (al menos 1x1 píxel)
            if abs(x2 - x1) > 0 and abs(y2 - y1) > 0:
                self.crop_box = (x1, y1, x2, y2)
            else:
                self.crop_box = None
        
        self._crop_start = None
        self._active_handle = None
        self._drag_mode = None
        # Actualizar el preview completo
        self._update_preview()
    
    def _draw_checkerboard_background(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Dibuja un fondo de cuadros (checkerboard) en el área especificada del canvas."""
        # Tamaño de cada cuadrado del patrón
        square_size = 10
        
        # Colores del patrón (gris claro y gris oscuro)
        color1 = "lightgray"
        color2 = "gray"
        
        # Calcular el rango de cuadrados
        start_i = (x1 - 10) // square_size
        end_i = (x2 - 10) // square_size + 1
        start_j = (y1 - 10) // square_size
        end_j = (y2 - 10) // square_size + 1
        
        for i in range(start_i, end_i):
            for j in range(start_j, end_j):
                # Determinar el color según la posición (patrón de tablero de ajedrez)
                if (i + j) % 2 == 0:
                    color = color1
                else:
                    color = color2
                
                # Calcular coordenadas del cuadrado
                sq_x1 = 10 + i * square_size
                sq_y1 = 10 + j * square_size
                sq_x2 = sq_x1 + square_size
                sq_y2 = sq_y1 + square_size
                
                # Solo dibujar si está dentro del área especificada
                if sq_x2 > x1 and sq_x1 < x2 and sq_y2 > y1 and sq_y1 < y2:
                    # Recortar a los límites del área
                    draw_x1 = max(sq_x1, x1)
                    draw_y1 = max(sq_y1, y1)
                    draw_x2 = min(sq_x2, x2)
                    draw_y2 = min(sq_y2, y2)
                    
                    self.preview_canvas.create_rectangle(
                        draw_x1, draw_y1, draw_x2, draw_y2,
                        fill=color, outline="",
                        tags="crop_rect"
                    )
    
    def _draw_crop_rect(self) -> None:
        """Dibuja el rectángulo de recorte, sombreado del área no seleccionada y los 8 handles en el canvas."""
        # Eliminar todos los elementos de recorte anteriores
        self.preview_canvas.delete("crop_rect")
        self._crop_handles = []
        
        if self.crop_box is None:
            self._crop_rect_id = None
            return
        
        x1, y1, x2, y2 = self.crop_box
        
        # Usar la escala de visualización actual
        scale = getattr(self, '_display_scale', self._preview_scale)
        # Usar el offset de centrado dinámico
        center_offset = getattr(self, '_center_offset', (10, 10))
        
        # Convertir coordenadas originales a coordenadas del canvas
        canvas_x1 = int(x1 * scale) + center_offset[0]
        canvas_y1 = int(y1 * scale) + center_offset[1]
        canvas_x2 = int(x2 * scale) + center_offset[0]
        canvas_y2 = int(y2 * scale) + center_offset[1]
        
        # Calcular las coordenadas de la imagen en el canvas
        img_canvas_x1 = center_offset[0]
        img_canvas_y1 = center_offset[1]
        disp_w, disp_h = getattr(self, "_current_display_size", self._original_size)
        img_canvas_x2 = img_canvas_x1 + int(disp_w * scale)
        img_canvas_y2 = img_canvas_y1 + int(disp_h * scale)
        
        # Dibujar fondo de cuadros en las áreas del crop_box que están fuera de la imagen
        # Área izquierda del crop_box fuera de la imagen
        if canvas_x1 < img_canvas_x1:
            self._draw_checkerboard_background(
                canvas_x1, canvas_y1, min(canvas_x2, img_canvas_x1), canvas_y2
            )
        # Área derecha del crop_box fuera de la imagen
        if canvas_x2 > img_canvas_x2:
            self._draw_checkerboard_background(
                max(canvas_x1, img_canvas_x2), canvas_y1, canvas_x2, canvas_y2
            )
        # Área superior del crop_box fuera de la imagen
        if canvas_y1 < img_canvas_y1:
            self._draw_checkerboard_background(
                max(canvas_x1, img_canvas_x1), canvas_y1,
                min(canvas_x2, img_canvas_x2), min(canvas_y2, img_canvas_y1)
            )
        # Área inferior del crop_box fuera de la imagen
        if canvas_y2 > img_canvas_y2:
            self._draw_checkerboard_background(
                max(canvas_x1, img_canvas_x1), max(canvas_y1, img_canvas_y2),
                min(canvas_x2, img_canvas_x2), canvas_y2
            )
        
        # Dibujar sombreado translúcido sobre el área NO seleccionada
        # Usar un color gris oscuro con stipple para efecto translúcido
        shadow_color = "gray40"
        
        # Calcular los límites del área de la imagen escalada
        content_x1 = img_canvas_x1
        content_y1 = img_canvas_y1
        content_x2 = img_canvas_x2
        content_y2 = img_canvas_y2
        
        # Área superior
        if canvas_y1 > content_y1:
            self.preview_canvas.create_rectangle(
                content_x1, content_y1, content_x2, canvas_y1,
                fill=shadow_color, stipple="gray25", outline="",
                tags="crop_rect"
            )
        
        # Área inferior
        if canvas_y2 < content_y2:
            self.preview_canvas.create_rectangle(
                content_x1, canvas_y2, content_x2, content_y2,
                fill=shadow_color, stipple="gray25", outline="",
                tags="crop_rect"
            )
        
        # Área izquierda
        if canvas_x1 > content_x1:
            self.preview_canvas.create_rectangle(
                content_x1, max(canvas_y1, content_y1), canvas_x1, min(canvas_y2, content_y2),
                fill=shadow_color, stipple="gray25", outline="",
                tags="crop_rect"
            )
        
        # Área derecha
        if canvas_x2 < content_x2:
            self.preview_canvas.create_rectangle(
                canvas_x2, max(canvas_y1, content_y1), content_x2, min(canvas_y2, content_y2),
                fill=shadow_color, stipple="gray25", outline="",
                tags="crop_rect"
            )
        
        # Dibujar rectángulo con borde y fondo semitransparente
        self._crop_rect_id = self.preview_canvas.create_rectangle(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            outline="red", width=2, fill="", stipple="gray50",
            tags="crop_rect"
        )
        
        # También dibujar un rectángulo interno más claro
        self.preview_canvas.create_rectangle(
            canvas_x1 + 1, canvas_y1 + 1, canvas_x2 - 1, canvas_y2 - 1,
            outline="yellow", width=1, fill="", stipple="gray25",
            tags="crop_rect"
        )
        
        # Dibujar los 8 handles
        handle_positions = self._get_handle_positions()
        half_size = self._handle_size // 2
        
        for i, (hx, hy) in enumerate(handle_positions):
            # Dibujar handle como un cuadrado con borde
            handle_id = self.preview_canvas.create_rectangle(
                hx - half_size, hy - half_size,
                hx + half_size, hy + half_size,
                outline="white", width=2, fill="blue",
                tags="crop_rect"
            )
            self._crop_handles.append(handle_id)
            
            # Dibujar un pequeño punto central para mejor visibilidad
            self.preview_canvas.create_oval(
                hx - 1, hy - 1, hx + 1, hy + 1,
                fill="white", outline="white",
                tags="crop_rect"
            )
    
    def _apply_crop(self, img: Image.Image) -> Image.Image:
        """Aplica el recorte a la imagen si hay un área seleccionada.
        Si el área se sale de los límites, completa con blanco."""
        if self.crop_box is None:
            return img
        
        x1, y1, x2, y2 = self.crop_box
        
        # Calcular el área de recorte (puede estar fuera de los límites)
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Crear una imagen en blanco del tamaño del recorte
        if img.mode == "1":
            # Para modo 1 (blanco y negro), usar blanco (1)
            result = Image.new("1", (crop_width, crop_height), 1)
        elif img.mode == "L":
            # Para escala de grises, usar blanco (255)
            result = Image.new("L", (crop_width, crop_height), 255)
        else:
            # Para RGB u otros modos, usar blanco (255, 255, 255)
            result = Image.new("RGB", (crop_width, crop_height), (255, 255, 255))
        
        # Calcular qué parte de la imagen original está dentro del recorte
        img_x1 = max(0, x1)
        img_y1 = max(0, y1)
        img_x2 = min(img.width, x2)
        img_y2 = min(img.height, y2)
        
        # Si hay parte de la imagen dentro del recorte, copiarla
        if img_x1 < img_x2 and img_y1 < img_y2:
            # Calcular offset en la imagen de resultado
            result_x1 = img_x1 - x1
            result_y1 = img_y1 - y1
            result_x2 = result_x1 + (img_x2 - img_x1)
            result_y2 = result_y1 + (img_y2 - img_y1)
            
            # Extraer la parte de la imagen original que está dentro
            cropped_part = img.crop((img_x1, img_y1, img_x2, img_y2))
            
            # Pegar en la imagen de resultado
            result.paste(cropped_part, (result_x1, result_y1))
        
        return result
    
    def _copy_to_clipboard(self) -> None:
        """Copia la imagen resultante al portapapeles."""
        try:
            current_img = self._get_current_image()
            # Aplicar recorte si hay uno seleccionado
            current_img = self._apply_crop(current_img)
            
            # Convertir a RGB para el portapapeles
            if current_img.mode != "RGB":
                clipboard_img = current_img.convert("RGB")
            else:
                clipboard_img = current_img
            
            # Copiar al portapapeles usando PIL
            output = io.BytesIO()
            clipboard_img.save(output, format='PNG')
            output.seek(0)
            
            # En Windows, usar win32clipboard
            try:
                import win32clipboard
                from PIL import ImageWin
                
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                
                # Convertir a DIB para Windows clipboard
                output_dib = io.BytesIO()
                clipboard_img.save(output_dib, 'BMP')
                data = output_dib.getvalue()[14:]  # Saltar el header BMP
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                
                messagebox.showinfo("Copiado", "¡Imagen copiada al portapapeles exitosamente!")
            except ImportError:
                # Fallback: usar tkinter (menos confiable en Windows)
                self.clipboard_clear()
                # Intentar copiar la imagen directamente
                # Nota: esto puede no funcionar bien en todos los sistemas
                messagebox.showinfo(
                    "Limitación",
                    "Para copiar al portapapeles necesitas instalar pywin32:\n\npip install pywin32\n\n"
                    "Por ahora, usa 'Guardar como...' para guardar la imagen."
                )
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo copiar al portapapeles:\n{str(exc)}")
    
    def _save_as(self) -> None:
        """Permite guardar la imagen con un diálogo de archivo."""
        try:
            file_path = filedialog.asksaveasfilename(
                title="Guardar imagen como",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("BMP", "*.bmp"),
                    ("TIFF", "*.tif *.tiff"),
                    ("Todos los archivos", "*.*")
                ],
                initialfile=f"Engrosado_{self._get_thicken_percentage():.0f}%_portapapeles.png"
            )
            
            if not file_path:
                return
            
            output_path = Path(file_path)
            
            # Determinar formato por extensión
            ext = output_path.suffix.lower()
            to_save = self._get_current_image()
            # Aplicar recorte si hay uno seleccionado
            to_save = self._apply_crop(to_save)
            
            # Para JPEG, convertir a RGB
            if ext in {".jpg", ".jpeg"}:
                to_save = to_save.convert("RGB")
                to_save.save(output_path, format="JPEG", quality=95, optimize=True)
            else:
                to_save.save(output_path)
            
            messagebox.showinfo("Guardado", f"Imagen guardada exitosamente como:\n{output_path.name}")
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar la imagen:\n{str(exc)}")


class EngrosarLineasApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Engrosar líneas de imagen")
        self.geometry("640x240")
        self.resizable(False, False)
        
        # Estado
        self.selected_image = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Listo.")
        self.clipboard_image: Optional[Image.Image] = None  # Imagen temporal del portapapeles
        
        self._build_ui()
    
    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 8}
        
        # Selección de imagen
        frame_image = ttk.LabelFrame(self, text="Imagen")
        frame_image.pack(fill="x", **padding)
        
        btn_browse = ttk.Button(frame_image, text="Elegir imagen...", command=self._on_browse)
        btn_browse.pack(side="left", padx=(10, 8), pady=10)
        
        btn_paste = ttk.Button(frame_image, text="Pegar desde portapapeles", command=self._on_paste_clipboard)
        btn_paste.pack(side="left", padx=(0, 8), pady=10)
        
        self.lbl_image = ttk.Label(frame_image, text="(ninguna)", width=55)
        self.lbl_image.pack(side="left", padx=(0, 10))
        
        # Acciones
        frame_actions = ttk.Frame(self)
        frame_actions.pack(fill="x", **padding)
        
        self.btn_process = ttk.Button(
            frame_actions,
            text="Procesar imagen",
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
        self.clipboard_image = None  # Limpiar imagen del portapapeles si había
        self.lbl_image.configure(text=self._ellipsize_path(selected, max_chars=55))
        self.status_var.set("Imagen seleccionada. Pulsa Procesar para continuar.")
    
    def _on_paste_clipboard(self) -> None:
        """Intenta obtener una imagen desde el portapapeles."""
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showwarning(
                    "Sin imagen", 
                    "No hay imagen en el portapapeles.\n\nCopia una imagen (Ctrl+C) e intenta de nuevo."
                )
                return
            
            # Verificar que sea una imagen
            if not isinstance(img, Image.Image):
                messagebox.showwarning(
                    "No es una imagen",
                    "El contenido del portapapeles no es una imagen válida."
                )
                return
            
            # Guardar imagen en memoria
            self.clipboard_image = img
            self.selected_image.set("")  # Limpiar ruta de archivo
            self.lbl_image.configure(text=f"(portapapeles: {img.width}×{img.height}px)")
            self.status_var.set("Imagen del portapapeles cargada. Pulsa Procesar para continuar.")
            
        except Exception as exc:
            messagebox.showerror(
                "Error",
                f"No se pudo obtener la imagen del portapapeles:\n{str(exc)}"
            )
    
    def _on_process(self) -> None:
        image_path = self.selected_image.get().strip()
        
        # Verificar si hay una imagen del portapapeles o un archivo seleccionado
        if not image_path and self.clipboard_image is None:
            messagebox.showwarning("Falta imagen", "Selecciona primero una imagen o pega una desde el portapapeles.")
            return
        
        # Si hay imagen del portapapeles, usarla
        if self.clipboard_image is not None:
            # Usar valor por defecto (150%) - se ajustará en el diálogo de previsualización
            default_percentage = 150.0
            
            # Preparar UI
            self.progress.start()
            self._set_ui_enabled(False)
            self.status_var.set("Procesando imagen del portapapeles...")
            
            # Procesar en hilo separado
            thread = threading.Thread(
                target=self._process_clipboard_in_background,
                args=(self.clipboard_image.copy(), default_percentage),
                daemon=True,
            )
            thread.start()
            return
        
        # Flujo original para archivos
        if not os.path.exists(image_path):
            messagebox.showerror("Error", "La imagen seleccionada no existe.")
            return
        
        # Usar valor por defecto (150%) para archivos también
        default_percentage = 150.0
        
        input_path = Path(image_path)
        if not is_image_file(input_path):
            messagebox.showwarning("Archivo inválido", "El archivo seleccionado no es una imagen válida.")
            return
        
        # Generar nombre de salida
        output_name = f"Engrosado_{default_percentage:.0f}%_{input_path.name}"
        output_path = input_path.parent / output_name
        
        # Preparar UI
        self.progress.start()
        self._set_ui_enabled(False)
        self.status_var.set("Procesando imagen...")
        
        # Procesar en hilo separado
        thread = threading.Thread(
            target=self._process_in_background,
            args=(input_path, default_percentage, output_path),
            daemon=True,
        )
        thread.start()
    
    def _process_in_background(
        self,
        input_path: Path,
        percentage: float,
        output_path: Path
    ) -> None:
        try:
            thicken_lines(input_path, percentage, output_path)
            
            def _finish() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                self.status_var.set(f"¡Completado! Imagen guardada como: {output_path.name}")
                messagebox.showinfo("Listo", f"Imagen procesada exitosamente.\n\nGuardada como:\n{output_path.name}")
            
            self.after(0, _finish)
        except Exception as exc:
            def _error() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                error_msg = f"Error al procesar la imagen: {str(exc)}"
                self.status_var.set(error_msg)
                messagebox.showerror("Error", error_msg)
            
            self.after(0, _error)
    
    def _process_clipboard_in_background(
        self,
        img: Image.Image,
        percentage: float
    ) -> None:
        """Procesa una imagen del portapapeles y muestra el resultado en un diálogo."""
        try:
            def _finish() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                self.status_var.set("¡Completado! Mostrando resultado...")
                
                # Mostrar diálogo con la imagen original (el engrosamiento se aplica en tiempo real)
                ResultDialog(self, img, percentage)
            
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
    app = EngrosarLineasApp()
    app.mainloop()


if __name__ == "__main__":
    main()

