import numpy as np
import matplotlib.pyplot as plt
import math


def rangoMax(img):
    """Las imagenes del proyecto se manejan normalizadas en [0,1]."""
    return 1.0


def clipLike(img_ref, arr):
    """Recorta al rango valido [0,1]."""
    return np.clip(arr, 0.0, 1.0)

def capaRoja(img):#slising
    """Conserva solo el canal rojo y anula verde y azul."""
    newImg = np.copy(img)
    newImg[:,:,1] = 0
    newImg[:,:,2] = 0
    return newImg

def capaVerde(img):
    """Conserva solo el canal verde y anula rojo y azul."""
    newImg = np.copy(img)
    newImg[:,:,0] = 0
    newImg[:,:,2] = 0
    return newImg

def capaAzul(img):
    """Conserva solo el canal azul y anula rojo y verde."""
    newImg = np.copy(img)
    newImg[:,:,0] = 0
    newImg[:,:,1] = 0
    return newImg

def capaAmarillo(img):
    """Genera capa amarilla (R+G), anulando el canal azul."""
    newImg = np.copy(img)
    newImg[:,:,2] = 0
    return newImg

def capaCyan(img):
    """Genera capa cyan (G+B), anulando el canal rojo."""
    newImg = np.copy(img)
    newImg[:,:,0] = 0
    return newImg

def capaMagenta(img):
    """Genera capa magenta (R+B), anulando el canal verde."""
    newImg = np.copy(img)
    newImg[:,:,1] = 0
    return newImg


def capaNegativa(img):
    """Calcula el negativo de una imagen normalizada en [0,1]."""
    newImg = 1.0 - np.copy(img)
    return newImg


def sumaImagenes(img1, img2, factor):
    """
    Combina dos imagenes: img1*factor + (1-factor)*img2.
    factor debe estar entre 0 y 1.
    """
    if img1.shape != img2.shape:
        raise ValueError("Las dos imagenes deben tener la misma forma")
    if factor < 0 or factor > 1:
        raise ValueError("El factor debe estar entre 0 y 1")

    mezcla = img1.astype(np.float64) * factor + (1.0 - factor) * img2.astype(np.float64)
    return clipLike(img1, mezcla)

def unirCanales(R,G,B):
    """Suma tres matrices/canales para recomponer una imagen."""
    return R+G+B

def grisAverage(img):
    """Convierte a gris promediando los canales RGB."""
    newImg = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0
    return newImg

def midGray(img):
    """Convierte a gris como promedio entre maximo y minimo por pixel."""
    max_c = np.max(img, axis=2)
    min_c = np.min(img, axis=2)
    newImg = (max_c + min_c) / 2.0
    return newImg

def ajusteBrillo(img,brillo):
    """Ajusta brillo sumando un desplazamiento y recortando a [0,1]."""
    newImg = np.copy(img).astype(np.float64)
    return clipLike(img, newImg + brillo)

def ajusteCanal(img,brillo,canal):
    """Ajusta brillo de un canal especifico (0=R, 1=G, 2=B)."""
    if canal not in (0, 1, 2):
        raise ValueError("canal debe ser 0 (R), 1 (G) o 2 (B)")
    newImg = np.copy(img).astype(np.float64)
    newImg[:,:,canal] = newImg[:,:,canal] + brillo
    return clipLike(img, newImg)

def ajusteContraste(img, k, tipo):
    """
    tipo=1: ajuste logaritmico, tipo!=1: ajuste exponencial (potencia).
    """
    img_norm = np.copy(img).astype(np.float64)

    if tipo == 1:
        # Normaliza por log(2) para mantener salida en [0, 1]
        new_norm = k * (np.log1p(img_norm) / np.log(2.0))
    else:
        # k>1 oscurece tonos medios, 0<k<1 los aclara
        new_norm = np.power(img_norm, k)

    return clipLike(img, new_norm)

def binarizar(img,umbral):
    """Binariza imagen en gris con umbral, devolviendo valores 0.0 o 1.0."""
    gray = grisAverage(img)
    return (gray > umbral).astype(np.float64)


def trasladar(img, dx, dy):
    """
    Traslada la imagen en pixeles:
    dx > 0 mueve a la derecha, dx < 0 a la izquierda.
    dy > 0 mueve hacia abajo, dy < 0 hacia arriba.
    El fondo nuevo se rellena con negro.
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    x_src_ini = max(0, -dx)
    x_src_fin = min(w, w - dx)
    y_src_ini = max(0, -dy)
    y_src_fin = min(h, h - dy)

    x_dst_ini = max(0, dx)
    x_dst_fin = min(w, w + dx)
    y_dst_ini = max(0, dy)
    y_dst_fin = min(h, h + dy)

    if x_src_ini < x_src_fin and y_src_ini < y_src_fin:
        out[y_dst_ini:y_dst_fin, x_dst_ini:x_dst_fin] = img[y_src_ini:y_src_fin, x_src_ini:x_src_fin]

    return out


def recorte(img, x, y, ancho, alto):
    """Devuelve el recorte empezando en (x, y) con tamano (ancho, alto)."""
    h, w = img.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(w, x0 + int(ancho))
    y1 = min(h, y0 + int(alto))
    return np.copy(img[y0:y1, x0:x1])


def rotar(img, angulo_grados):
    """
    Rota alrededor del centro usando vecino mas cercano y fondo negro.
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img)

    ang = math.radians(angulo_grados)
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)

    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    yy, xx = np.indices((h, w))
    x_rel = xx - cx
    y_rel = yy - cy

    # Mapeo inverso: destino -> origen
    x_src = cos_a * x_rel + sin_a * y_rel + cx
    y_src = -sin_a * x_rel + cos_a * y_rel + cy

    x_src_i = np.rint(x_src).astype(int)
    y_src_i = np.rint(y_src).astype(int)

    valid = (x_src_i >= 0) & (x_src_i < w) & (y_src_i >= 0) & (y_src_i < h)
    out[yy[valid], xx[valid]] = img[y_src_i[valid], x_src_i[valid]]
    return out


def reduccion(img, factor):
    """
    Reduce resolucion espacial.
    factor en (0, 1]: 0.5 deja la imagen a la mitad en ancho y alto.
    """
    if factor <= 0 or factor > 1:
        raise ValueError("factor debe estar en el rango (0, 1]")

    h, w = img.shape[:2]
    new_h = max(1, int(round(h * factor)))
    new_w = max(1, int(round(w * factor)))

    y_idx = np.linspace(0, h - 1, new_h).astype(int)
    x_idx = np.linspace(0, w - 1, new_w).astype(int)
    return img[y_idx[:, None], x_idx]


def zoom(img, factor):
    """
    Zoom centrado respecto al centro de la imagen.
    factor > 1 acerca, 0 < factor < 1 aleja.
    La salida conserva el mismo tamano de la imagen original.
    """
    if factor <= 0:
        raise ValueError("factor debe ser mayor que 0")

    h, w = img.shape[:2]

    yy, xx = np.indices((h, w))
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Mapeo inverso: cada pixel destino toma valor desde origen escalado al centro.
    x_src = (xx - cx) / factor + cx
    y_src = (yy - cy) / factor + cy

    x_src_i = np.rint(x_src).astype(int)
    y_src_i = np.rint(y_src).astype(int)

    out = np.zeros_like(img)
    valid = (x_src_i >= 0) & (x_src_i < w) & (y_src_i >= 0) & (y_src_i < h)

    if img.ndim == 2:
        out[valid] = img[y_src_i[valid], x_src_i[valid]]
    else:
        out[yy[valid], xx[valid]] = img[y_src_i[valid], x_src_i[valid]]

    return out

