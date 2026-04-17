import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Permite importar desde Src/lib al ejecutar este archivo directamente.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from lib import procesamiento as proc


class EditorImagenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Imagenes - Procesamiento Vectorial")
        self.root.geometry("1300x800")

        self.original_img = None
        self.current_img = None
        self.mix_img = None

        self._build_ui()

    def _build_ui(self):
        controls_container = ttk.Frame(self.root, padding=10)
        controls_container.pack(side=tk.LEFT, fill=tk.Y)

        controls_canvas = tk.Canvas(controls_container, highlightthickness=0, width=380)
        controls_scroll = ttk.Scrollbar(controls_container, orient=tk.VERTICAL, command=controls_canvas.yview)
        controls_canvas.configure(yscrollcommand=controls_scroll.set)

        controls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        def _on_controls_configure(_event):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def _on_canvas_configure(event):
            controls_canvas.itemconfigure(controls_window, width=event.width)

        controls.bind("<Configure>", _on_controls_configure)
        controls_canvas.bind("<Configure>", _on_canvas_configure)

        viewer = ttk.Frame(self.root, padding=10)
        viewer.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=viewer)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        top = ttk.LabelFrame(controls, text="Carga", padding=8)
        top.pack(fill=tk.X, pady=4)

        ttk.Button(top, text="Cargar imagen", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(top, text="Cargar imagen mezcla", command=self.load_mix_image).pack(fill=tk.X, pady=2)
        ttk.Button(top, text="Exportar imagen", command=self.export_current_image).pack(fill=tk.X, pady=2)
        ttk.Button(top, text="Guardar imagen", command=self.save_current_as_original).pack(fill=tk.X, pady=2)
        ttk.Button(top, text="Restaurar original", command=self.reset_image).pack(fill=tk.X, pady=2)

        capas = ttk.LabelFrame(controls, text="Capas y Basicos", padding=8)
        capas.pack(fill=tk.X, pady=4)
        ttk.Button(capas, text="Capa Roja", command=lambda: self.apply_rgb(proc.capaRoja)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Capa Verde", command=lambda: self.apply_rgb(proc.capaVerde)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Capa Azul", command=lambda: self.apply_rgb(proc.capaAzul)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Capa Cyan", command=lambda: self.apply_rgb(proc.capaCyan)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Capa Magenta", command=lambda: self.apply_rgb(proc.capaMagenta)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Capa Amarillo", command=lambda: self.apply_rgb(proc.capaAmarillo)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Negativa", command=lambda: self.apply_rgb(proc.capaNegativa)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Gris Average", command=lambda: self.apply_gray(proc.grisAverage)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Mid Gray", command=lambda: self.apply_gray(proc.midGray)).pack(fill=tk.X, pady=2)
        ttk.Button(capas, text="Unir Canales (R+G+B)", command=self.apply_unir_canales).pack(fill=tk.X, pady=2)

        blend = ttk.LabelFrame(controls, text="Suma de Imagenes", padding=8)
        blend.pack(fill=tk.X, pady=4)
        self.blend_factor = tk.DoubleVar(value=0.5)
        ttk.Scale(blend, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.blend_factor).pack(fill=tk.X, pady=2)
        ttk.Button(blend, text="Aplicar sumaImagenes", command=self.apply_suma_imagenes).pack(fill=tk.X, pady=2)

        ajuste = ttk.LabelFrame(controls, text="Ajustes", padding=8)
        ajuste.pack(fill=tk.X, pady=4)

        self.brightness = tk.DoubleVar(value=0.0)
        ttk.Label(ajuste, text="Brillo [-1,1]").pack(anchor="w")
        ttk.Scale(ajuste, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, variable=self.brightness).pack(fill=tk.X, pady=2)
        ttk.Button(ajuste, text="Aplicar ajusteBrillo", command=self.apply_ajuste_brillo).pack(fill=tk.X, pady=2)

        self.channel_delta = tk.DoubleVar(value=0.0)
        self.channel = tk.IntVar(value=0)
        ttk.Label(ajuste, text="Ajuste Canal [-1,1]").pack(anchor="w")
        ttk.Scale(ajuste, from_=-1.0, to=1.0, orient=tk.HORIZONTAL, variable=self.channel_delta).pack(fill=tk.X, pady=2)
        canal_frame = ttk.Frame(ajuste)
        canal_frame.pack(fill=tk.X)
        ttk.Radiobutton(canal_frame, text="R", value=0, variable=self.channel).pack(side=tk.LEFT)
        ttk.Radiobutton(canal_frame, text="G", value=1, variable=self.channel).pack(side=tk.LEFT)
        ttk.Radiobutton(canal_frame, text="B", value=2, variable=self.channel).pack(side=tk.LEFT)
        ttk.Button(ajuste, text="Aplicar ajusteCanal", command=self.apply_ajuste_canal).pack(fill=tk.X, pady=2)

        self.k_contrast = tk.DoubleVar(value=1.0)
        self.tipo_contrast = tk.IntVar(value=1)
        ttk.Label(ajuste, text="Contraste k [0.1, 3]").pack(anchor="w")
        ttk.Scale(ajuste, from_=0.1, to=3.0, orient=tk.HORIZONTAL, variable=self.k_contrast).pack(fill=tk.X, pady=2)
        tipo_frame = ttk.Frame(ajuste)
        tipo_frame.pack(fill=tk.X)
        ttk.Radiobutton(tipo_frame, text="Log (tipo=1)", value=1, variable=self.tipo_contrast).pack(side=tk.LEFT)
        ttk.Radiobutton(tipo_frame, text="Potencia", value=2, variable=self.tipo_contrast).pack(side=tk.LEFT)
        ttk.Button(ajuste, text="Aplicar ajusteContraste", command=self.apply_ajuste_contraste).pack(fill=tk.X, pady=2)

        umbral_frame = ttk.LabelFrame(controls, text="Binarizacion", padding=8)
        umbral_frame.pack(fill=tk.X, pady=4)
        self.umbral = tk.DoubleVar(value=0.5)
        ttk.Scale(umbral_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, variable=self.umbral).pack(fill=tk.X, pady=2)
        ttk.Button(umbral_frame, text="Aplicar binarizar", command=self.apply_binarizar).pack(fill=tk.X, pady=2)

        geo = ttk.LabelFrame(controls, text="Geometricas", padding=8)
        geo.pack(fill=tk.X, pady=4)

        self.dx = tk.IntVar(value=0)
        self.dy = tk.IntVar(value=0)
        ttk.Label(geo, text="Trasladar dx [-300,300]").pack(anchor="w")
        ttk.Scale(geo, from_=-300, to=300, orient=tk.HORIZONTAL, variable=self.dx).pack(fill=tk.X, pady=2)
        ttk.Label(geo, text="Trasladar dy [-300,300]").pack(anchor="w")
        ttk.Scale(geo, from_=-300, to=300, orient=tk.HORIZONTAL, variable=self.dy).pack(fill=tk.X, pady=2)
        ttk.Button(geo, text="Aplicar trasladar", command=self.apply_trasladar).pack(fill=tk.X, pady=2)

        recorte_frame = ttk.LabelFrame(geo, text="Recorte", padding=6)
        recorte_frame.pack(fill=tk.X, pady=2)

        self.crop_x = tk.IntVar(value=0)
        self.crop_y = tk.IntVar(value=0)
        self.crop_w = tk.IntVar(value=740)
        self.crop_h = tk.IntVar(value=400)
        self.crop_summary = tk.StringVar(value="Desde (0, 0) con tamano 740 x 400")

        self.crop_x_scale = tk.Scale(
            recorte_frame,
            from_=0,
            to=739,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Inicio X",
            variable=self.crop_x,
            command=self._on_crop_control_change,
        )
        self.crop_x_scale.pack(fill=tk.X)

        self.crop_y_scale = tk.Scale(
            recorte_frame,
            from_=0,
            to=399,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Inicio Y",
            variable=self.crop_y,
            command=self._on_crop_control_change,
        )
        self.crop_y_scale.pack(fill=tk.X)

        self.crop_w_scale = tk.Scale(
            recorte_frame,
            from_=1,
            to=740,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Ancho del recorte",
            variable=self.crop_w,
            command=self._on_crop_control_change,
        )
        self.crop_w_scale.pack(fill=tk.X)

        self.crop_h_scale = tk.Scale(
            recorte_frame,
            from_=1,
            to=400,
            orient=tk.HORIZONTAL,
            resolution=1,
            label="Alto del recorte",
            variable=self.crop_h,
            command=self._on_crop_control_change,
        )
        self.crop_h_scale.pack(fill=tk.X)

        ttk.Label(
            recorte_frame,
            text="Selecciona el punto inicial (X, Y) y luego el ancho y alto del area a recortar.",
            wraplength=320,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=2)
        ttk.Label(recorte_frame, textvariable=self.crop_summary, wraplength=320, justify=tk.LEFT).pack(fill=tk.X, pady=2)

        self._update_crop_ranges()
        ttk.Button(geo, text="Aplicar recorte", command=self.apply_recorte).pack(fill=tk.X, pady=2)

        self.angle = tk.DoubleVar(value=0.0)
        ttk.Label(geo, text="Rotar angulo [-180,180]").pack(anchor="w")
        ttk.Scale(geo, from_=-180, to=180, orient=tk.HORIZONTAL, variable=self.angle).pack(fill=tk.X, pady=2)
        ttk.Button(geo, text="Aplicar rotar", command=self.apply_rotar).pack(fill=tk.X, pady=2)

        self.reduc_factor = tk.DoubleVar(value=0.5)
        ttk.Label(geo, text="Reduccion factor (0,1]").pack(anchor="w")
        ttk.Scale(geo, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.reduc_factor).pack(fill=tk.X, pady=2)
        ttk.Button(geo, text="Aplicar reduccion", command=self.apply_reduccion).pack(fill=tk.X, pady=2)

        self.zoom_factor = tk.DoubleVar(value=1.5)
        ttk.Label(geo, text="Zoom factor > 0").pack(anchor="w")
        ttk.Scale(geo, from_=0.1, to=3.0, orient=tk.HORIZONTAL, variable=self.zoom_factor).pack(fill=tk.X, pady=2)
        ttk.Button(geo, text="Aplicar zoom", command=self.apply_zoom).pack(fill=tk.X, pady=2)

    def normalize_image(self, img):
        arr = np.array(img, dtype=np.float64)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)

    def to_rgb(self, img):
        if img.ndim == 2:
            return np.stack([img, img, img], axis=2)
        return img

    def resize_to_shape(self, img, target_h, target_w):
        h, w = img.shape[:2]
        y_idx = np.linspace(0, h - 1, target_h).astype(int)
        x_idx = np.linspace(0, w - 1, target_w).astype(int)
        return img[y_idx[:, None], x_idx]

    def _on_crop_control_change(self, _value=None):
        self._update_crop_ranges()

    def _update_crop_ranges(self):
        """Ajusta rangos de sliders de recorte para evitar valores invalidos."""
        if self.original_img is None:
            max_w = 740
            max_h = 400
            max_x = max_w - 1
            max_y = max_h - 1
        else:
            base = self.get_base_rgb()
            max_h, max_w = base.shape[:2]
            max_x = max_w - 1
            max_y = max_h - 1

        self.crop_x_scale.configure(to=max_x)
        self.crop_y_scale.configure(to=max_y)

        if self.crop_x.get() > max_x:
            self.crop_x.set(max_x)
        if self.crop_y.get() > max_y:
            self.crop_y.set(max_y)

        w_max_dynamic = max(1, max_w - self.crop_x.get())
        h_max_dynamic = max(1, max_h - self.crop_y.get())
        self.crop_w_scale.configure(to=w_max_dynamic)
        self.crop_h_scale.configure(to=h_max_dynamic)

        if self.crop_w.get() > w_max_dynamic:
            self.crop_w.set(w_max_dynamic)
        if self.crop_h.get() > h_max_dynamic:
            self.crop_h.set(h_max_dynamic)

        self.crop_summary.set(
            f"Desde ({self.crop_x.get()}, {self.crop_y.get()}) con tamano {self.crop_w.get()} x {self.crop_h.get()}"
        )

    def reset_controls(self):
        """Restaura sliders y controles a su estado inicial."""
        self.blend_factor.set(0.5)

        self.brightness.set(0.0)
        self.channel_delta.set(0.0)
        self.channel.set(0)

        self.k_contrast.set(1.0)
        self.tipo_contrast.set(1)
        self.umbral.set(0.5)

        self.dx.set(0)
        self.dy.set(0)

        self.angle.set(0.0)
        self.reduc_factor.set(0.5)
        self.zoom_factor.set(1.5)

        self._update_crop_ranges()
        self.crop_x.set(0)
        self.crop_y.set(0)
        self._update_crop_ranges()

        if self.original_img is None:
            self.crop_w.set(740)
            self.crop_h.set(400)
        else:
            base = self.get_base_rgb()
            h, w = base.shape[:2]
            self.crop_w.set(w)
            self.crop_h.set(h)

    def has_image(self):
        if self.original_img is None:
            messagebox.showwarning("Falta imagen", "Primero carga una imagen.")
            return False
        return True

    def get_base_rgb(self):
        """Devuelve la imagen original actual como RGB para procesar."""
        return self.to_rgb(self.original_img)

    def update_view(self):
        if self.current_img is None:
            return
        self.ax.clear()
        self.ax.axis("off")
        self.ax.imshow(np.clip(self.current_img, 0.0, 1.0))
        self.canvas.draw_idle()

    def load_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
        )
        if not path:
            return
        try:
            img = plt.imread(path)
            img = self.normalize_image(img)
            self.original_img = np.copy(img)
            self.current_img = np.copy(img)
            self._update_crop_ranges()
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")

    def load_mix_image(self):
        path = filedialog.askopenfilename(
            title="Selecciona la segunda imagen",
            filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")],
        )
        if not path:
            return
        try:
            img = plt.imread(path)
            self.mix_img = self.normalize_image(img)
            messagebox.showinfo("Ok", "Imagen de mezcla cargada.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar la segunda imagen:\n{e}")

    def reset_image(self):
        if self.original_img is not None:
            self.current_img = np.copy(self.original_img)
            self.reset_controls()
            self.update_view()

    def save_current_as_original(self):
        """Convierte la imagen actualmente visualizada en la nueva imagen original."""
        if self.current_img is None:
            messagebox.showwarning("Falta imagen", "No hay imagen visualizada para guardar.")
            return
        self.original_img = np.copy(self.current_img)
        self._update_crop_ranges()
        messagebox.showinfo("Imagen guardada", "La imagen visualizada ahora es la nueva original.")

    def export_current_image(self):
        """Exporta la imagen visualizada a la ruta elegida por el usuario."""
        if self.current_img is None:
            messagebox.showwarning("Falta imagen", "No hay imagen visualizada para exportar.")
            return

        path = filedialog.asksaveasfilename(
            title="Exportar imagen",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif;*.tiff"),
            ],
        )
        if not path:
            return

        try:
            plt.imsave(path, np.clip(self.current_img, 0.0, 1.0))
            messagebox.showinfo("Imagen exportada", f"La imagen se guardo en:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar la imagen:\n{e}")

    def apply_rgb(self, fn):
        if not self.has_image():
            return
        try:
            img = self.get_base_rgb()
            self.current_img = self.normalize_image(fn(img))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_gray(self, fn):
        if not self.has_image():
            return
        try:
            img = self.get_base_rgb()
            gray = fn(img)
            self.current_img = self.to_rgb(self.normalize_image(gray))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_unir_canales(self):
        if not self.has_image():
            return
        try:
            img = self.get_base_rgb()
            r = np.zeros_like(img)
            g = np.zeros_like(img)
            b = np.zeros_like(img)
            r[:, :, 0] = img[:, :, 0]
            g[:, :, 1] = img[:, :, 1]
            b[:, :, 2] = img[:, :, 2]
            self.current_img = self.normalize_image(proc.unirCanales(r, g, b))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_suma_imagenes(self):
        if not self.has_image():
            return
        if self.mix_img is None:
            messagebox.showwarning("Falta imagen", "Carga la segunda imagen para sumaImagenes.")
            return
        try:
            base = self.get_base_rgb()
            mix = self.to_rgb(self.mix_img)
            if mix.shape != base.shape:
                mix = self.resize_to_shape(mix, base.shape[0], base.shape[1])
            # Solo muestra la combinacion; no cambia la imagen original.
            self.current_img = self.normalize_image(proc.sumaImagenes(base, mix, float(self.blend_factor.get())))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_ajuste_brillo(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(proc.ajusteBrillo(self.get_base_rgb(), float(self.brightness.get())))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_ajuste_canal(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(
                proc.ajusteCanal(self.get_base_rgb(), float(self.channel_delta.get()), int(self.channel.get()))
            )
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_ajuste_contraste(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(
                proc.ajusteContraste(self.get_base_rgb(), float(self.k_contrast.get()), int(self.tipo_contrast.get()))
            )
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_binarizar(self):
        if not self.has_image():
            return
        try:
            out = proc.binarizar(self.get_base_rgb(), float(self.umbral.get()))
            self.current_img = self.to_rgb(self.normalize_image(out))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_trasladar(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(
                proc.trasladar(self.get_base_rgb(), int(self.dx.get()), int(self.dy.get()))
            )
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_recorte(self):
        if not self.has_image():
            return
        try:
            x = int(self.crop_x.get())
            y = int(self.crop_y.get())
            w = int(self.crop_w.get())
            h = int(self.crop_h.get())
            out = proc.recorte(self.get_base_rgb(), x, y, w, h)
            if out.size == 0:
                messagebox.showwarning("Recorte", "El recorte resulto vacio. Revisa parametros.")
                return
            self.current_img = self.normalize_image(out)
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_rotar(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(proc.rotar(self.get_base_rgb(), float(self.angle.get())))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_reduccion(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(proc.reduccion(self.get_base_rgb(), float(self.reduc_factor.get())))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_zoom(self):
        if not self.has_image():
            return
        try:
            self.current_img = self.normalize_image(proc.zoom(self.get_base_rgb(), float(self.zoom_factor.get())))
            self.update_view()
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    app = EditorImagenApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
