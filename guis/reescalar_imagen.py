import os
import io
import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from PIL import Image, ImageFilter, ImageGrab, ImageTk


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp",
}


RESAMPLE_METHODS = {
    "LANCZOS (mejor calidad fotos)": Image.Resampling.LANCZOS,
    "BICUBIC (suave)": Image.Resampling.BICUBIC,
    "HAMMING (reducción)": Image.Resampling.HAMMING,
    "BOX (reducción rápida)": Image.Resampling.BOX,
    "BILINEAR": Image.Resampling.BILINEAR,
    "NEAREST (pixel art)": Image.Resampling.NEAREST,
}


def is_image_file(file_path: Path) -> bool:
    return file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def ensure_unique_output_path(base_path: Path) -> Path:
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


def rescale_image(
    img: Image.Image,
    new_size: tuple[int, int],
    method: Image.Resampling = Image.Resampling.LANCZOS,
    sharpen_amount: float = 0.0,
    multi_step: bool = True,
) -> Image.Image:
    """Reescala imagen preservando calidad.

    Args:
        img: imagen PIL.
        new_size: (ancho, alto) destino en px.
        method: algoritmo de remuestreo.
        sharpen_amount: 0..200 (porcentaje UnsharpMask). 0 desactiva.
        multi_step: para upscaling grande (>2x), aplicar pasos intermedios
            (mejora calidad percibida con BICUBIC/LANCZOS).
    """
    src_w, src_h = img.size
    dst_w, dst_h = new_size
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError("Tamaño destino inválido")

    work = img
    if work.mode in ("P", "1"):
        work = work.convert("RGBA" if "A" in work.mode else "RGB")

    if multi_step and method in (Image.Resampling.LANCZOS, Image.Resampling.BICUBIC):
        scale_x = dst_w / src_w
        scale_y = dst_h / src_h
        if scale_x > 2.0 or scale_y > 2.0:
            cur_w, cur_h = src_w, src_h
            while cur_w * 2 < dst_w and cur_h * 2 < dst_h:
                cur_w *= 2
                cur_h *= 2
                work = work.resize((cur_w, cur_h), method)

    resized = work.resize((dst_w, dst_h), method)

    if sharpen_amount > 0:
        if resized.mode == "RGBA":
            r, g, b, a = resized.split()
            rgb = Image.merge("RGB", (r, g, b)).filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=int(sharpen_amount), threshold=2)
            )
            r2, g2, b2 = rgb.split()
            resized = Image.merge("RGBA", (r2, g2, b2, a))
        else:
            if resized.mode != "RGB" and resized.mode != "L":
                resized = resized.convert("RGB")
            resized = resized.filter(
                ImageFilter.UnsharpMask(radius=1.5, percent=int(sharpen_amount), threshold=2)
            )

    return resized


class ResultDialog(tk.Toplevel):
    """Diálogo para previsualizar y guardar el reescalado."""

    def __init__(self, parent: tk.Tk, original_image: Image.Image) -> None:
        super().__init__(parent)
        self.title("Reescalar imagen - Resultado")
        self.original_image = original_image
        self._orig_w, self._orig_h = original_image.size

        self.scale_var = tk.DoubleVar(value=200.0)
        self.scale_label_var = tk.StringVar(value="200%")
        self.width_var = tk.StringVar(value=str(self._orig_w * 2))
        self.height_var = tk.StringVar(value=str(self._orig_h * 2))
        self.lock_aspect_var = tk.BooleanVar(value=True)
        self.method_var = tk.StringVar(value="LANCZOS (mejor calidad fotos)")
        self.multi_step_var = tk.BooleanVar(value=True)
        self.sharpen_var = tk.DoubleVar(value=0.0)
        self.sharpen_label_var = tk.StringVar(value="0%")
        self.info_var = tk.StringVar()

        self._cached_key: Optional[tuple] = None
        self._cached_image: Optional[Image.Image] = None
        self._suspend_sync = False
        self._update_after_id: Optional[str] = None

        self._build_ui()
        self._update_preview()

        self.update_idletasks()
        w = max(720, self.winfo_reqwidth())
        h = max(560, self.winfo_reqheight())
        self.geometry(f"{w}x{h}")
        self.minsize(640, 500)
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.transient(parent)
        self.grab_set()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        info_frame = ttk.Frame(outer)
        info_frame.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Label(info_frame, textvariable=self.info_var, font=("TkDefaultFont", 9, "bold")).pack()

        preview_frame = ttk.LabelFrame(outer, text="Vista previa")
        preview_frame.pack(fill="both", expand=True, padx=10, pady=4)
        self.preview_label = ttk.Label(preview_frame, anchor="center", background="white")
        self.preview_label.pack(fill="both", expand=True, padx=8, pady=8)

        scale_frame = ttk.LabelFrame(outer, text="Escala")
        scale_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(scale_frame, text="Factor:").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        ttk.Scale(
            scale_frame, from_=10.0, to=800.0, orient="horizontal",
            variable=self.scale_var, command=self._on_scale_change,
        ).grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=6)
        ttk.Label(scale_frame, textvariable=self.scale_label_var, width=8, anchor="e").grid(
            row=0, column=2, sticky="e", padx=(0, 10), pady=6
        )
        scale_frame.columnconfigure(1, weight=1)

        ttk.Label(scale_frame, text="Ancho (px):").grid(row=1, column=0, sticky="w", padx=10, pady=4)
        w_entry = ttk.Entry(scale_frame, textvariable=self.width_var, width=10)
        w_entry.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=4)
        w_entry.bind("<KeyRelease>", lambda _e: self._on_width_change())
        w_entry.bind("<FocusOut>", lambda _e: self._on_width_change())

        ttk.Label(scale_frame, text="Alto (px):").grid(row=2, column=0, sticky="w", padx=10, pady=4)
        h_entry = ttk.Entry(scale_frame, textvariable=self.height_var, width=10)
        h_entry.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=4)
        h_entry.bind("<KeyRelease>", lambda _e: self._on_height_change())
        h_entry.bind("<FocusOut>", lambda _e: self._on_height_change())

        ttk.Checkbutton(
            scale_frame, text="Mantener proporción", variable=self.lock_aspect_var
        ).grid(row=1, column=2, rowspan=2, sticky="w", padx=10, pady=4)

        algo_frame = ttk.LabelFrame(outer, text="Algoritmo y nitidez")
        algo_frame.pack(fill="x", padx=10, pady=4)

        ttk.Label(algo_frame, text="Método:").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        method_cb = ttk.Combobox(
            algo_frame, textvariable=self.method_var,
            values=list(RESAMPLE_METHODS.keys()), state="readonly", width=32,
        )
        method_cb.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=6)
        method_cb.bind("<<ComboboxSelected>>", lambda _e: self._schedule_update())

        ttk.Checkbutton(
            algo_frame, text="Upscaling progresivo (mejor calidad >2x)",
            variable=self.multi_step_var, command=self._schedule_update,
        ).grid(row=0, column=2, sticky="w", padx=10, pady=6)

        ttk.Label(algo_frame, text="Nitidez:").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        ttk.Scale(
            algo_frame, from_=0.0, to=200.0, orient="horizontal",
            variable=self.sharpen_var, command=self._on_sharpen_change,
        ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 10), pady=6)
        ttk.Label(algo_frame, textvariable=self.sharpen_label_var, width=8, anchor="e").grid(
            row=1, column=3, sticky="e", padx=(0, 10), pady=6
        )
        algo_frame.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(outer)
        btn_frame.pack(fill="x", padx=10, pady=(4, 10))
        ttk.Button(btn_frame, text="Vista previa", command=self._update_preview).pack(side="left", padx=(0, 5))
        ttk.Button(btn_frame, text="Copiar al portapapeles", command=self._copy_to_clipboard).pack(
            side="left", padx=5, expand=True, fill="x"
        )
        ttk.Button(btn_frame, text="Guardar como...", command=self._save_as).pack(
            side="left", padx=5, expand=True, fill="x"
        )
        ttk.Button(btn_frame, text="Cerrar", command=self.destroy).pack(side="left", padx=(5, 0))

    def _schedule_update(self, delay_ms: int = 250) -> None:
        if self._update_after_id is not None:
            try:
                self.after_cancel(self._update_after_id)
            except Exception:
                pass
        self._update_after_id = self.after(delay_ms, self._update_preview)

    def _on_scale_change(self, _value: str) -> None:
        try:
            pct = float(self.scale_var.get())
        except (TypeError, ValueError):
            return
        self.scale_label_var.set(f"{pct:.0f}%")
        new_w = max(1, int(round(self._orig_w * pct / 100.0)))
        new_h = max(1, int(round(self._orig_h * pct / 100.0)))
        self._suspend_sync = True
        self.width_var.set(str(new_w))
        self.height_var.set(str(new_h))
        self._suspend_sync = False
        self._schedule_update()

    def _on_sharpen_change(self, _value: str) -> None:
        try:
            v = float(self.sharpen_var.get())
        except (TypeError, ValueError):
            return
        self.sharpen_label_var.set(f"{v:.0f}%")
        self._schedule_update()

    def _on_width_change(self) -> None:
        if self._suspend_sync:
            return
        try:
            w = int(self.width_var.get())
        except (TypeError, ValueError):
            return
        if w <= 0:
            return
        self._suspend_sync = True
        if self.lock_aspect_var.get():
            h = max(1, int(round(w * self._orig_h / self._orig_w)))
            self.height_var.set(str(h))
        pct = w / self._orig_w * 100.0
        self.scale_var.set(pct)
        self.scale_label_var.set(f"{pct:.0f}%")
        self._suspend_sync = False
        self._schedule_update()

    def _on_height_change(self) -> None:
        if self._suspend_sync:
            return
        try:
            h = int(self.height_var.get())
        except (TypeError, ValueError):
            return
        if h <= 0:
            return
        self._suspend_sync = True
        if self.lock_aspect_var.get():
            w = max(1, int(round(h * self._orig_w / self._orig_h)))
            self.width_var.set(str(w))
            pct = w / self._orig_w * 100.0
        else:
            pct = h / self._orig_h * 100.0
        self.scale_var.set(pct)
        self.scale_label_var.set(f"{pct:.0f}%")
        self._suspend_sync = False
        self._schedule_update()

    def _get_target_size(self) -> tuple[int, int]:
        try:
            w = max(1, int(self.width_var.get()))
            h = max(1, int(self.height_var.get()))
        except (TypeError, ValueError):
            w, h = self._orig_w, self._orig_h
        return w, h

    def _get_method(self) -> Image.Resampling:
        return RESAMPLE_METHODS.get(self.method_var.get(), Image.Resampling.LANCZOS)

    def _get_current_image(self) -> Image.Image:
        target = self._get_target_size()
        method = self._get_method()
        sharpen = float(self.sharpen_var.get())
        multi = bool(self.multi_step_var.get())
        key = (target, method, round(sharpen, 1), multi)
        if self._cached_image is not None and self._cached_key == key:
            return self._cached_image
        result = rescale_image(self.original_image, target, method, sharpen, multi)
        self._cached_image = result
        self._cached_key = key
        return result

    def _update_preview(self) -> None:
        self._update_after_id = None
        try:
            current_img = self._get_current_image()
        except Exception as exc:
            self.info_var.set(f"Error: {exc}")
            return

        tw, th = current_img.size
        self.info_var.set(
            f"Original: {self._orig_w}×{self._orig_h}px  →  Destino: {tw}×{th}px  "
            f"({tw / self._orig_w * 100:.0f}% × {th / self._orig_h * 100:.0f}%)"
        )

        max_preview = 520
        scale = min(max_preview / tw, max_preview / th, 1.0)
        pw = max(1, int(tw * scale))
        ph = max(1, int(th * scale))
        preview_img = current_img if scale >= 1.0 else current_img.resize(
            (pw, ph), Image.Resampling.LANCZOS
        )
        if preview_img.mode not in ("RGB", "RGBA"):
            preview_img = preview_img.convert("RGB")
        self._photo = ImageTk.PhotoImage(preview_img)
        self.preview_label.configure(image=self._photo, text="")

    def _copy_to_clipboard(self) -> None:
        try:
            current_img = self._get_current_image()
            clipboard_img = current_img.convert("RGB") if current_img.mode != "RGB" else current_img
            try:
                import win32clipboard
                output_dib = io.BytesIO()
                clipboard_img.save(output_dib, "BMP")
                data = output_dib.getvalue()[14:]
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
                win32clipboard.CloseClipboard()
                messagebox.showinfo("Copiado", "Imagen copiada al portapapeles.")
            except ImportError:
                messagebox.showinfo(
                    "Limitación",
                    "Para copiar al portapapeles instala pywin32:\n\npip install pywin32\n\n"
                    "Mientras tanto usa 'Guardar como...'"
                )
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo copiar:\n{exc}")

    def _save_as(self) -> None:
        try:
            tw, th = self._get_target_size()
            file_path = filedialog.asksaveasfilename(
                title="Guardar imagen reescalada",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("BMP", "*.bmp"),
                    ("TIFF", "*.tif *.tiff"),
                    ("WEBP", "*.webp"),
                    ("Todos los archivos", "*.*"),
                ],
                initialfile=f"Reescalado_{tw}x{th}.png",
            )
            if not file_path:
                return
            output_path = Path(file_path)
            ext = output_path.suffix.lower()
            to_save = self._get_current_image()
            if ext in {".jpg", ".jpeg"}:
                if to_save.mode != "RGB":
                    to_save = to_save.convert("RGB")
                to_save.save(output_path, format="JPEG", quality=95, subsampling=0, optimize=True)
            elif ext == ".webp":
                to_save.save(output_path, format="WEBP", quality=95, method=6)
            else:
                to_save.save(output_path)
            messagebox.showinfo("Guardado", f"Imagen guardada como:\n{output_path.name}")
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo guardar:\n{exc}")


class ReescalarImagenApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Reescalar imagen sin perder calidad")
        self.geometry("640x240")
        self.resizable(False, False)

        self.selected_image = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Listo.")
        self.clipboard_image: Optional[Image.Image] = None

        self._build_ui()

    def _build_ui(self) -> None:
        padding = {"padx": 10, "pady": 8}

        frame_image = ttk.LabelFrame(self, text="Imagen")
        frame_image.pack(fill="x", **padding)

        ttk.Button(frame_image, text="Elegir imagen...", command=self._on_browse).pack(
            side="left", padx=(10, 8), pady=10
        )
        ttk.Button(frame_image, text="Pegar desde portapapeles", command=self._on_paste_clipboard).pack(
            side="left", padx=(0, 8), pady=10
        )
        self.lbl_image = ttk.Label(frame_image, text="(ninguna)", width=55)
        self.lbl_image.pack(side="left", padx=(0, 10))

        frame_actions = ttk.Frame(self)
        frame_actions.pack(fill="x", **padding)
        self.btn_process = ttk.Button(frame_actions, text="Abrir reescalador", command=self._on_process)
        self.btn_process.pack(side="left", padx=(10, 8))
        self.progress = ttk.Progressbar(frame_actions, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill="x", expand=True, padx=(0, 10))

        frame_status = ttk.Frame(self)
        frame_status.pack(fill="x", **padding)
        ttk.Label(frame_status, textvariable=self.status_var, anchor="w").pack(fill="x", padx=10)

    def _on_browse(self) -> None:
        selected = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff *.webp"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not selected:
            return
        self.selected_image.set(selected)
        self.clipboard_image = None
        self.lbl_image.configure(text=self._ellipsize_path(selected, max_chars=55))
        self.status_var.set("Imagen seleccionada. Pulsa 'Abrir reescalador'.")

    def _on_paste_clipboard(self) -> None:
        try:
            img = ImageGrab.grabclipboard()
            if img is None or not isinstance(img, Image.Image):
                messagebox.showwarning("Sin imagen", "No hay imagen válida en el portapapeles.")
                return
            self.clipboard_image = img
            self.selected_image.set("")
            self.lbl_image.configure(text=f"(portapapeles: {img.width}×{img.height}px)")
            self.status_var.set("Imagen del portapapeles cargada. Pulsa 'Abrir reescalador'.")
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo obtener la imagen:\n{exc}")

    def _on_process(self) -> None:
        image_path = self.selected_image.get().strip()
        if not image_path and self.clipboard_image is None:
            messagebox.showwarning("Falta imagen", "Selecciona o pega una imagen primero.")
            return

        if self.clipboard_image is not None:
            self._open_dialog(self.clipboard_image.copy())
            return

        if not os.path.exists(image_path):
            messagebox.showerror("Error", "La imagen seleccionada no existe.")
            return
        input_path = Path(image_path)
        if not is_image_file(input_path):
            messagebox.showwarning("Archivo inválido", "No es una imagen válida.")
            return

        self.progress.start()
        self._set_ui_enabled(False)
        self.status_var.set("Cargando imagen...")
        thread = threading.Thread(target=self._load_in_background, args=(input_path,), daemon=True)
        thread.start()

    def _load_in_background(self, input_path: Path) -> None:
        try:
            with Image.open(input_path) as img:
                img.load()
                loaded = img.copy()

            def _finish() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                self.status_var.set("Listo.")
                self._open_dialog(loaded)

            self.after(0, _finish)
        except Exception as exc:
            def _error() -> None:
                self.progress.stop()
                self._set_ui_enabled(True)
                self.status_var.set(f"Error: {exc}")
                messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{exc}")
            self.after(0, _error)

    def _open_dialog(self, img: Image.Image) -> None:
        ResultDialog(self, img)

    def _set_ui_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_process.configure(state=state)

    @staticmethod
    def _ellipsize_path(path_str: str, max_chars: int = 70) -> str:
        if len(path_str) <= max_chars:
            return path_str
        head = path_str[: max_chars // 2 - 2]
        tail = path_str[-(max_chars // 2 - 3):]
        return f"{head}...{tail}"


def main() -> None:
    app = ReescalarImagenApp()
    app.mainloop()


if __name__ == "__main__":
    main()
