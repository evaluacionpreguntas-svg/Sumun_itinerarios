
# -*- coding: utf-8 -*-
import os
import io
import json
import re

# Visualización y Datos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as mpl_Polygon, Circle as mpl_Circle, FancyArrowPatch as mpl_FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib_venn import venn2, venn3
import networkx as nx
import graphviz
from shapely.geometry import LineString, Polygon

# IA (LangChain)
from langchain.prompts import PromptTemplate
import vertexai
from langchain_google_vertexai import ChatVertexAI

PLUGIN_REGISTRY = {}
PLUGIN_ALIASES = {}

def _escape_braces(text: str) -> str:
    """Duplica llaves para que LangChain no las trate como variables en el prompt."""
    return text.replace("{", "{{").replace("}", "}}")


def _init_vertex() -> None:
    """Inicializa Vertex AI con project y location desde env vars."""
    project = os.environ.get("GCP_PROJECT_ID")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project:
        raise RuntimeError("Falta GCP_PROJECT_ID en variables de entorno.")
    vertexai.init(project=project, location=location)

def _get_llm() -> ChatVertexAI:
    """Retorna un ChatVertexAI (Gemini en Vertex) listo para usar."""
    _init_vertex()
    # Puedes ajustar el modelo si lo necesitas
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash") 
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2"))
    return ChatVertexAI(
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=8192,
    )

def register_chart(name, *aliases):
    """Decorador para registrar una función como un plugin para crear gráficos."""
    def _inner(func):
        key = str(name).lower().strip()
        PLUGIN_REGISTRY[key] = func
        for a in aliases:
            PLUGIN_ALIASES[str(a).lower().strip()] = key
        return func
    return _inner

def _resolve_plugin_key(tipo: str) -> str:
    """Resuelve un nombre o alias al nombre oficial del plugin."""
    t = str(tipo).lower().strip()
    return PLUGIN_ALIASES.get(t, t)

def ensure_fig_ax(ax=None, **kwargs):
    """Asegura que tengamos una figura y ejes de Matplotlib para dibujar."""
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
        return fig, ax
    return ax.get_figure(), ax


# ==============================================================================
# 2. DEFINICIÓN DE PLUGINS PARA CADA TIPO DE GRÁFICO
# ==============================================================================


# 1) grafico_barras_verticales
@register_chart("grafico_barras_verticales", "barras_verticales", "bar")
def plugin_grafico_barras_verticales(datos: dict, configuracion: dict, ax=None, **kwargs):
    """
    Barras verticales tolerante:
    - Acepta:
      1) dict de listas: {"Categorias":[...], "Valores":[...]}
      2) dict mapeo: {"A":12,"B":9,"C":15}
      3) sinónimos: x/labels/categorias/etiquetas vs y/values/valores/data
      4) "series": [{"x":[...],"y":[...]}]
      5) CSV: "A,B,C" y "12,9,15"
    """
    import matplotlib.pyplot as plt

    # --- Normalizar fig/ax sin suponer tipo del 3er argumento ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    def _split_csv(x):
        if isinstance(x, str) and "," in x:
            return [s.strip() for s in x.split(",")]
        return x

    def _as_list(x):
        x = _split_csv(x)
        if isinstance(x, list):
            return x
        if x is None:
            return None
        try:
            if not isinstance(x, (str, bytes, dict)):
                return list(x)
        except Exception:
            pass
        return [x]

    def _is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def _to_num_list(lst):
        out = []
        for it in lst:
            if _is_num(it):
                out.append(float(it))
            elif isinstance(it, str):
                s = it.strip().replace(",", ".")
                try:
                    out.append(float(s))
                except Exception:
                    raise ValueError(f"Valor no numérico: {it!r}")
            else:
                raise ValueError(f"Valor no numérico: {it!r}")
        return out

    categorias, valores = None, None

    # 0) dict mapeo {"A":12,"B":9}
    if datos and isinstance(datos, dict) and len(datos) > 0 and all(not isinstance(v, list) for v in datos.values()):
        try:
            categorias = list(datos.keys())
            valores = _to_num_list(list(datos.values()))
        except Exception:
            categorias, valores = None, None

    # 1) series
    if (categorias is None or valores is None) and isinstance(datos.get("series"), list) and datos["series"]:
        s0 = datos["series"][0] or {}
        for ck in ("x", "labels", "categorias", "categorías", "etiquetas"):
            if ck in s0:
                categorias = _as_list(s0[ck]); break
        for vk in ("y", "values", "valores", "data"):
            if vk in s0:
                valores = _as_list(s0[vk]); break

    # 2) sinónimos en raíz
    if categorias is None:
        for ck in ("x", "labels", "categorias", "categorías", "etiquetas"):
            if ck in datos:
                categorias = _as_list(datos[ck]); break
    if valores is None:
        for vk in ("y", "values", "valores", "data"):
            if vk in datos:
                valores = _as_list(datos[vk]); break

    # 3) autodetección
    if (categorias is None or valores is None) and isinstance(datos, dict):
        list_keys = [k for k, v in datos.items() if isinstance(_as_list(v), list)]
        str_keys, num_keys = [], []
        for k in list_keys:
            lst = _as_list(datos[k]) or []
            if len(lst) == 0:
                continue
            if all(isinstance(x, str) for x in lst):
                str_keys.append(k)
            else:
                # si todos (o casi) son numéricos (o strings numéricos), la tratamos como numérica
                try:
                    _to_num_list(lst)
                    num_keys.append(k)
                except Exception:
                    pass

        if categorias is None and str_keys:
            categorias = _as_list(datos[str_keys[0]])
        if valores is None and num_keys:
            valores = _as_list(datos[num_keys[0]])

        # Si no hubo texto, aceptar (num, num): primera como categorías numéricas
        if (categorias is None or valores is None) and len(num_keys) >= 2:
            if categorias is None:
                categorias = _as_list(datos[num_keys[0]])
            if valores is None:
                valores = _as_list(datos[num_keys[1]])

        # Último recurso: dos primeras listas cualesquiera
        if (categorias is None or valores is None) and len(list_keys) >= 2:
            if categorias is None:
                categorias = _as_list(datos[list_keys[0]])
            if valores is None:
                valores = _as_list(datos[list_keys[1]])

    if categorias is None or valores is None:
        raise ValueError("Se requiere 1 lista de categorías y 1 lista numérica (o dict categoría→valor).")

    categorias = _as_list(categorias)
    valores = _to_num_list(_as_list(valores))

    n = min(len(categorias), len(valores))
    if n == 0:
        raise ValueError("Listas vacías de categorías/valores.")
    categorias, valores = categorias[:n], valores[:n]

    ax.bar(
        categorias, valores,
        color=(configuracion or {}).get("color", "blue"),
        **((configuracion or {}).get("plot_config", {}) or {})
    )
    ax.set_title((configuracion or {}).get("titulo", "Gráfico de Barras"), pad=10)
    ax.set_xlabel((configuracion or {}).get("xlabel", "Categoría"))
    ax.set_ylabel((configuracion or {}).get("ylabel", "Valor"))
    plt.xticks(
        rotation=(configuracion or {}).get("xticks_rotation", 0),
        ha=(configuracion or {}).get("xticks_ha", "center")
    )
    return fig, ax




# 2) grafico_circular
@register_chart("grafico_circular", "pie")
def plugin_circular(datos, configuracion, debug=False):
    text_keys = [k for k, v in datos.items() if isinstance(v, list) and all(isinstance(x, str) for x in v)]
    num_keys  = [k for k, v in datos.items() if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v)]
    if len(text_keys) != 1 or len(num_keys) != 1:
        raise ValueError("Se requiere 1 lista de etiquetas (texto) y 1 lista numérica.")
    labels_key, values_key = text_keys[0], num_keys[0]
    labels, sizes = datos[labels_key], datos[values_key]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct=configuracion.get("autopct", "%1.1f%%"),
        startangle=configuracion.get("startangle", 90),
        colors=configuracion.get("colors", None),
        wedgeprops={"edgecolor": "w"},
        **configuracion.get("plot_config", {})
    )
    ax.axis("equal")
    ax.set_title(configuracion.get("titulo", "Diagrama Circular"), pad=10)
    return fig, ax


# 3) tabla
@register_chart("tabla", "table")
def plugin_tabla(datos: dict, configuracion: dict, *maybe_ax, **kwargs):
    """
    Tabla con ajuste automático de texto por celda (multilínea real) y clip.
    Parche rápido: ignora cualquier tercer argumento posicional que NO sea un Axes.
    Retorna (fig, ax).
    """
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.axes import Axes

    # ------------------ Parche rápido: aceptar 'ax' solo si es Axes ------------------
    ax = None
    if maybe_ax:
        cand = maybe_ax[0]
        if isinstance(cand, Axes):
            ax = cand  # usar el Axes si realmente lo pasaron
        else:
            ax = None  # ignorar cosas como True/False, etc.

    # ------------------ utilidades internas ------------------
    def _ensure_fig_ax(ax=None, figsize=(8, 5)):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        return fig, ax

    def _parse_figsize(fs):
        """Admite (w,h), [w,h], '11x6.5', '11×6.5', '11,6.5', '[11,6.5]'."""
        if fs is None:
            return None
        if isinstance(fs, (list, tuple)) and len(fs) == 2:
            w, h = fs
            if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                return float(w), float(h)
            return None
        if isinstance(fs, str):
            s = fs.strip().lower().replace(" ", "").replace("×", "x")
            m = re.match(r"^(\d+(\.\d+)?)[x,](\d+(\.\d+)?)$", s)
            if m:
                return float(m.group(1)), float(m.group(3))
            m = re.match(r"^\[(\d+(\.\d+)?),(\d+(\.\d+)?)\]$", s)
            if m:
                return float(m.group(1)), float(m.group(3))
        return None

    def _parse_col_widths(cw, ncols):
        """Devuelve lista de ncols con fracciones (≈suman 1). Admite lista o string."""
        if isinstance(cw, (list, tuple)) and len(cw) == ncols:
            vals = [float(x) for x in cw]
        elif isinstance(cw, str):
            s = cw.strip().replace("[", "").replace("]", "")
            parts = [p for p in re.split(r"[,\s]+", s) if p]
            if len(parts) == ncols:
                vals = [float(p) for p in parts]
            else:
                return None
        else:
            return None
        s = sum(vals) or 1.0
        return [v / s for v in vals]

    def _measure_px(renderer, fp, text: str) -> float:
        w, _, _ = renderer.get_text_width_height_descent(text, fp, ismath=False)
        return w

    def _wrap_to_px(renderer, fp, text: str, max_px: float) -> str:
        """Inserta '\n' para no exceder max_px; parte palabras largas si es necesario."""
        words = str(text).replace("\r", " ").replace("\n", " ").split()
        if not words:
            return " "
        lines, line = [], ""
        for w in words:
            cand = (line + " " + w).strip() if line else w
            if _measure_px(renderer, fp, cand) <= max_px:
                line = cand
            else:
                if line:
                    lines.append(line); line = ""
                chunk = ""
                for ch in w:
                    if _measure_px(renderer, fp, chunk + ch) <= max_px or not chunk:
                        chunk += ch
                    else:
                        lines.append(chunk); chunk = ch
                line = chunk
        if line:
            lines.append(line)
        return "\n".join(lines)

    # ------------------ validar datos ------------------
    matrix = datos.get("matrix")
    if not (isinstance(matrix, list) and matrix and all(isinstance(r, (list, tuple)) for r in matrix)):
        raise ValueError("Datos inválidos para 'tabla': 'matrix' debe ser lista de listas.")

    headers = ["" if h is None else str(h) for h in matrix[0]]
    body    = [list(r) for r in matrix[1:]]
    ncols   = len(headers)
    nrows   = len(body)

    # rectangularizar filas del cuerpo y convertir a str
    rows = []
    for r in body:
        rr = list(r[:ncols]) + [""] * max(0, ncols - len(r))
        rows.append([ "" if c is None else str(c) for c in rr ])

    # ------------------ configuración ------------------
    cfg = configuracion or {}
    title         = cfg.get("titulo", "Tabla")
    fontsize      = int(cfg.get("fontsize", 9))
    cellLoc       = cfg.get("cellLoc", "left")
    fill_axes     = bool(cfg.get("fill_axes", True))
    header_weight = float(cfg.get("header_weight", 1.25))
    row_extra     = float(cfg.get("row_extra", 0.0))
    min_col_frac  = float(cfg.get("min_col_frac", 0.08))
    max_col_frac  = float(cfg.get("max_col_frac", 0.72))

    fs = _parse_figsize(cfg.get("figsize"))
    if fs is None:
        fig_w = max(7.5, min(16.0, 2.2 * ncols))
        fig_h = max(4.5, min(14.0, 0.55 * (nrows + 1)))
        fs = (fig_w, fig_h)

    fig, ax = _ensure_fig_ax(ax, figsize=fs)
    try:
        plt.rcParams["text.usetex"] = False
    except Exception:
        pass
    ax.clear(); ax.axis("off")

    # ------------------ construir la grilla (sin texto) ------------------
    empty = [[""] * ncols for _ in range(nrows)]
    table = ax.table(
        cellText=empty,
        colLabels=headers,
        cellLoc=cellLoc,
        loc="center",
        bbox=[0, 0, 1, 1] if fill_axes else None,
        **(cfg.get("plot_config", {}) or {})
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.8 if r == 0 else 0.5)

    # ------------------ anchos de columna ------------------
    col_widths = _parse_col_widths(cfg.get("col_widths"), ncols)
    if col_widths is None:
        # heurística por nº de caracteres
        max_chars = []
        for c in range(ncols):
            col_vals = [headers[c]] + [rows[i][c] for i in range(nrows)]
            max_chars.append(max(len(str(x)) for x in col_vals) if col_vals else 1)
        total = float(sum(max_chars)) or 1.0
        norm = [max(min_col_frac, min(max_col_frac, m / total)) for m in max_chars]
        s = sum(norm) or 1.0
        norm = [w / s * 0.98 for w in norm]
    else:
        norm = [max(min_col_frac, min(max_col_frac, float(w))) for w in col_widths]
        s = sum(norm) or 1.0
        norm = [w / s * 0.98 for w in norm]
    for (r, c), cell in table.get_celld().items():
        cell.set_width(norm[c])

    # ------------------ 1er draw: medidas y wrapping ------------------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fp = FontProperties(size=fontsize)

    axes_bb   = ax.get_window_extent(renderer=renderer)
    axes_px_w = axes_bb.width
    pad_px    = 10
    col_px    = [max(12, axes_px_w * float(norm[c]) - 2 * pad_px) for c in range(ncols)]

    # cabecera
    wrapped_header = []
    max_header_lines = 1
    for c in range(ncols):
        w = _wrap_to_px(renderer, fp, headers[c], col_px[c])
        wrapped_header.append(w)
        max_header_lines = max(max_header_lines, w.count("\n") + 1)

    # cuerpo
    wrapped_rows = []
    line_cnt = [max_header_lines]
    for r in range(nrows):
        row_wrapped = []
        max_lines = 1
        for c in range(ncols):
            w = _wrap_to_px(renderer, fp, rows[r][c], col_px[c])
            row_wrapped.append(w)
            max_lines = max(max_lines, w.count("\n") + 1)
        wrapped_rows.append(row_wrapped)
        line_cnt.append(max(1, max_lines))

    # alturas de fila proporcional al nº de líneas
    weights = [max(1, line_cnt[0]) * header_weight] + [n + row_extra for n in line_cnt[1:]]
    tot = float(sum(weights)) or 1.0
    for r in range(nrows + 1):  # incluye cabecera (r=0)
        h = 0.98 * (weights[r] / tot)
        for c in range(ncols):
            table[r, c].set_height(h)

    # ------------------ 2º draw y overlay multilínea con clip ------------------
    fig.canvas.draw()

    def _place_text(r, c, text, bold=False):
        cell = table[r, c]
        x0, y0 = cell.get_x(), cell.get_y()
        cw, ch = cell.get_width(), cell.get_height()

        # clip en coords del axes
        clip_rect = plt.Rectangle((x0, y0), cw, ch, transform=ax.transAxes,
                                  facecolor="none", edgecolor="none")
        ax.add_patch(clip_rect); clip_rect.set_visible(False)

        padx = 0.02 * cw
        if cellLoc == "left":
            xx, ha = x0 + padx, "left"
        elif cellLoc == "right":
            xx, ha = x0 + cw - padx, "right"
        else:
            xx, ha = x0 + cw / 2.0, "center"
        yy = y0 + ch / 2.0

        ax.text(
            xx, yy, text if text else " ",
            fontsize=fontsize, weight="bold" if bold else "normal",
            ha=ha, va="center", wrap=True,
            transform=ax.transAxes,
            clip_path=clip_rect, clip_on=True, zorder=6
        )

    # ocultar texto nativo del table (solo usamos grilla)
    for (r, c), cell in table.get_celld().items():
        if cell.get_text() is not None:
            cell.get_text().set_visible(False)

    # pintar cabecera y cuerpo
    for c in range(ncols):
        _place_text(0, c, wrapped_header[c], bold=True)
    for r in range(nrows):
        for c in range(ncols):
            _place_text(r + 1, c, wrapped_rows[r][c], bold=False)

    ax.set_title(title, pad=int(cfg.get("title_pad", 14)))
    fig.canvas.draw()
    plt.tight_layout()
    return fig, ax

# 4) construccion_geometrica
@register_chart("construccion_geometrica")
def plugin_construccion(datos, configuracion, debug=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(configuracion.get("titulo", "Construcción Geométrica"), pad=10)
    min_x, max_x = float('inf'), float('-inf'); min_y, max_y = float('inf'), float('-inf')
    if 'elements' in datos and isinstance(datos['elements'], list):
        for element in datos['elements']:
            elem_type = str(element.get('type', '')).lower()
            coords = element.get('coords')
            cfg = element.get('config', {}) or {}
            try:
                if elem_type == 'point' and isinstance(coords, list) and len(coords) == 2:
                    ax.plot(coords[0], coords[1], **{k: v for k, v in cfg.items() if k != 'label'})
                    if 'label' in cfg:
                        ax.text(coords[0], coords[1], cfg['label'], fontsize=cfg.get('fontsize', 10),
                                ha=cfg.get('ha', 'right'), va=cfg.get('va', 'bottom'))
                    min_x, max_x = min(min_x, coords[0]), max(max_x, coords[0])
                    min_y, max_y = min(min_y, coords[1]), max(max_y, coords[1])

                elif elem_type == 'line' and isinstance(coords, list) and len(coords) >= 2:
                    line = LineString(coords); x, y = line.xy
                    ax.plot(list(x), list(y), **cfg)
                    min_x, max_x = min(min_x, min(x)), max(max_x, max(x))
                    min_y, max_y = min(min_y, min(y)), max(max_y, max(y))

                elif elem_type == 'polygon' and isinstance(coords, list) and len(coords) >= 3:
                    patch = mpl_Polygon(coords, closed=True, **cfg); ax.add_patch(patch)
                    poly = Polygon(coords)
                    min_x, max_x = min(min_x, poly.bounds[0]), max(max_x, poly.bounds[2])
                    min_y, max_y = min(min_y, poly.bounds[1]), max(max_y, poly.bounds[3])

                elif elem_type == 'circle':
                    center = cfg.get('center'); radius = cfg.get('radius'); patch_cfg = dict(cfg.get('patch_config', {}))
                    if not (isinstance(center, (list, tuple)) and len(center) == 2 and isinstance(radius, (int, float))):
                        raise ValueError("Círculo requiere 'center' [x,y] y 'radius'.")
                    for k, v in cfg.items():
                        if k not in ('center', 'radius', 'start', 'end', 'patch_config'):
                            patch_cfg[k] = v
                    circ = mpl_Circle(center, radius, **patch_cfg); ax.add_patch(circ)
                    min_x, max_x = min(min_x, center[0] - radius), max(max_x, center[0] + radius)
                    min_y, max_y = min(min_y, center[1] - radius), max(max_y, center[1] + radius)

                elif elem_type == 'arrow':
                    start = cfg.get('start'); end = cfg.get('end'); patch_cfg = dict(cfg.get('patch_config', {}))
                    if not (isinstance(start, (list, tuple)) and isinstance(end, (list, tuple)) and len(start) == 2 and len(end) == 2):
                        raise ValueError("Flecha requiere 'start' [x1,y1] y 'end' [x2,y2].")
                    for k, v in cfg.items():
                        if k not in ('center', 'radius', 'start', 'end', 'patch_config'):
                            patch_cfg[k] = v
                    arr = mpl_FancyArrowPatch(start, end, **patch_cfg); ax.add_patch(arr)
                    min_x, max_x = min(min_x, start[0], end[0]), max(max_x, start[0], end[0])
                    min_y, max_y = min(min_y, start[1], end[1]), max(max_y, start[1], end[1])

            except Exception as e:
                print(f"❌ Error geométrico '{elem_type}': {e}")
    else:
        print("⚠️ Falta 'elements' lista.")

    x_margin = (max_x - min_x) * 0.1 if max_x > min_x else 1
    y_margin = (max_y - min_y) * 0.1 if max_y > min_y else 1
    ax.set_xlim(configuracion.get('xlim', (min_x - x_margin, max_x + x_margin)))
    ax.set_ylim(configuracion.get('ylim', (min_y - y_margin, max_y + y_margin)))
    if configuracion.get('axis_off'):
        ax.axis('off')
    return fig, ax


# 5) diagrama_arbol (DOT opcional)
@register_chart("diagrama_arbol")
def plugin_arbol(datos, configuracion, debug=False):
    if 'dot_source' in datos and isinstance(datos.get('dot_source'), str):
        try:
            dot = graphviz.Source(datos['dot_source'])
            return io.BytesIO(dot.pipe(format='png'))
        except graphviz.backend.execute.ExecutableNotFound:
            print("❌ Graphviz no disponible, uso networkx.")
        except Exception as e:
            print(f"❌ DOT error: {e}; uso networkx.")

    # networkx fallback
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(configuracion.get("titulo", "Diagrama de Árbol"), pad=10); ax.axis('off')
    G = nx.DiGraph() if configuracion.get('directed', False) else nx.Graph()
    if 'nodes' in datos: G.add_nodes_from(datos['nodes'])
    if 'edges' in datos: G.add_edges_from(datos['edges'])
    layout = str(configuracion.get('layout', 'spring')).lower()
    try:
        if layout == 'sugiyama':
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        elif layout == 'planar':
            pos = nx.planar_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
    except Exception:
        pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=configuracion.get('with_labels', True), labels=datos.get('labels', {}),
        node_size=configuracion.get('node_size', 700), node_color=configuracion.get('node_color', 'skyblue'),
        font_size=configuracion.get('font_size', 10), edge_color=configuracion.get('edge_color', 'black'),
        ax=ax, **configuracion.get("plot_config", {})
    )
    # Edge labels "A-B"
    parsed = {}
    for k, v in (configuracion.get('edge_labels', {}) or {}).items():
        if '-' in k:
            a, b = k.split('-', 1); a, b = a.strip(), b.strip()
            if (a, b) in G.edges() or (not G.is_directed() and (b, a) in G.edges()):
                parsed[(a, b)] = v
    if parsed:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=parsed, ax=ax,
                                     font_size=configuracion.get('font_size', 10),
                                     **configuracion.get("edge_label_config", {}))
    return fig, ax


# 6) flujograma (DOT puro)
@register_chart("flujograma")
def plugin_flujograma(datos, configuracion, debug=False):
    dot_src = datos.get("dot_source")
    if not isinstance(dot_src, str):
        raise ValueError("Para flujograma se requiere 'dot_source' (str).")
    try:
        dot = graphviz.Source(dot_src)
        return io.BytesIO(dot.pipe(format='png'))
    except graphviz.backend.execute.ExecutableNotFound:
        raise RuntimeError("Graphviz no está instalado en el sistema.")
    except Exception as e:
        raise RuntimeError(f"Error Graphviz: {e}")


# 7) pictograma (Waffle; fallback)
@register_chart("pictograma", "waffle")
def plugin_pictograma(datos, configuracion, debug=False):
    values = datos.get('values'); colors = datos.get('colors'); rows = int(configuracion.get('rows', 10))
    show_legend = configuracion.get('show_legend', True)
    legend_loc = configuracion.get('legend_loc', 'lower left'); legend_bbox = configuracion.get('legend_bbox', (0, -0.1))
    if not isinstance(values, dict) or not values:
        raise ValueError("'values' debe ser dict no vacío.")
    try:
        from pywaffle import Waffle
        fig = plt.figure(FigureClass=Waffle, rows=rows, values=values, colors=colors,
                         legend={'loc': legend_loc, 'bbox_to_anchor': legend_bbox} if show_legend else None)
        ax = fig.axes[0]; ax.set_title(configuracion.get("titulo", "Pictograma (Waffle)"), pad=10)
        return fig, ax
    except Exception as e:
        # Fallback manual
        cols = int(configuracion.get('cols', 10)); total = sum(values.values()); total_tiles = rows * cols
        tile_counts = {}; acc = 0
        for k, v in values.items():
            c = round(v / total * total_tiles); tile_counts[k] = c; acc += c
        diff = total_tiles - acc
        keys_sorted = sorted(values.keys(), key=lambda k: values[k], reverse=True)
        i = 0
        while diff != 0 and keys_sorted:
            k0 = keys_sorted[i % len(keys_sorted)]; tile_counts[k0] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1; i += 1
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(configuracion.get("titulo", "Pictograma"), pad=10)
        ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect('equal'); ax.axis('off')
        if not colors or len(colors) < len(values):
            cmap = plt.get_cmap('tab20'); colors = [cmap(i) for i in range(len(values))]
        cat_list = list(values.keys()); color_map = {k: colors[i % len(colors)] for i, k in enumerate(cat_list)}
        x, y = 0, rows - 1
        for k in cat_list:
            for _ in range(max(0, tile_counts[k])):
                ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color_map[k], edgecolor='white'))
                x += 1
                if x >= cols: x = 0; y -= 1
        if show_legend:
            handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[k]) for k in cat_list]
            ax.legend(handles, cat_list, loc=legend_loc, bbox_to_anchor=legend_bbox, ncol=2)
        return fig, ax


# 8) scatter_plot
@register_chart("scatter_plot")
def plugin_scatter(datos, configuracion, debug=False):
    x, y = datos.get('x'), datos.get('y')
    if not (isinstance(x, list) and isinstance(y, list) and len(x) == len(y)):
        raise ValueError("'x' y 'y' deben ser listas de igual longitud.")
    fig, ax = plt.subplots(figsize=(6, 4))
    if configuracion.get('use_seaborn', False):
        try:
            import seaborn as sns
            sns.scatterplot(x=x, y=y, ax=ax, color=configuracion.get('color', None),
                            marker=configuracion.get('marker', 'o'))
        except Exception:
            ax.scatter(x, y, c=configuracion.get('color', None), marker=configuracion.get('marker', 'o'))
    else:
        ax.scatter(x, y, c=configuracion.get('color', None), marker=configuracion.get('marker', 'o'))
    ax.set_title(configuracion.get('titulo', 'Gráfico de Dispersión'), pad=10)
    ax.set_xlabel(configuracion.get('xlabel', 'Eje X')); ax.set_ylabel(configuracion.get('ylabel', 'Eje Y'))
    return fig, ax


# 9) line_plot
@register_chart("line_plot")
def plugin_line(datos, configuracion, debug=False):
    x, y = datos.get('x'), datos.get('y')
    if not (isinstance(x, list) and isinstance(y, list) and len(x) == len(y)):
        raise ValueError("'x' y 'y' deben ser listas de igual longitud.")
    fig, ax = plt.subplots(figsize=(6, 4))
    if configuracion.get('use_seaborn', False):
        try:
            import seaborn as sns
            sns.lineplot(x=x, y=y, ax=ax, color=configuracion.get('color', None),
                         linestyle=configuracion.get('linestyle', '-'), marker=configuracion.get('marker', None))
        except Exception:
            ax.plot(x, y, color=configuracion.get('color', None), linestyle=configuracion.get('linestyle', '-'),
                    marker=configuracion.get('marker', None))
    else:
        ax.plot(x, y, color=configuracion.get('color', None), linestyle=configuracion.get('linestyle', '-'),
                marker=configuracion.get('marker', None))
    ax.set_title(configuracion.get('titulo', 'Gráfico de Línea'), pad=10)
    ax.set_xlabel(configuracion.get('xlabel', 'Eje X')); ax.set_ylabel(configuracion.get('ylabel', 'Eje Y'))
    return fig, ax


# 10) histogram
@register_chart("histogram")
def plugin_hist(datos, configuracion, debug=False):
    values = datos.get('values')
    if not (isinstance(values, list) and values):
        raise ValueError("'values' debe ser lista no vacía.")
    fig, ax = plt.subplots(figsize=(6, 4))
    if configuracion.get('use_seaborn', False):
        try:
            import seaborn as sns
            sns.histplot(values, bins=int(configuracion.get('bins', 10)), ax=ax)
        except Exception:
            ax.hist(values, bins=int(configuracion.get('bins', 10)))
    else:
        ax.hist(values, bins=int(configuracion.get('bins', 10)))
    ax.set_title(configuracion.get('titulo', 'Histograma'), pad=10)
    ax.set_xlabel(configuracion.get('xlabel', 'Valores')); ax.set_ylabel(configuracion.get('ylabel', 'Frecuencia'))
    return fig, ax


# 11) box_plot
@register_chart("box_plot")
def plugin_box(datos, configuracion, debug=False):
    data = datos.get('data'); labels = configuracion.get('labels')
    if data is None and any(isinstance(v, list) for v in datos.values()):
        keys = [k for k, v in datos.items() if isinstance(v, list)]
        data = [datos[k] for k in keys]
        if labels is None: labels = keys
    if not (isinstance(data, list) and all(isinstance(lst, list) for lst in data)):
        raise ValueError("Use 'data' como lista de listas o pase dict de listas (se normaliza).")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(data, labels=labels, patch_artist=True)
    ax.set_title(configuracion.get('titulo', 'Diagrama de Caja'), pad=10)
    ax.set_ylabel(configuracion.get('ylabel', 'Valores'))
    return fig, ax


# 12) violin_plot
@register_chart("violin_plot")
def plugin_violin(datos, configuracion, debug=False):
    data = datos.get('data'); labels = configuracion.get('labels')
    x_list, y_list = datos.get('x'), datos.get('y')
    fig, ax = plt.subplots(figsize=(6, 4))
    if configuracion.get('use_seaborn', False) and x_list is not None and y_list is not None:
        try:
            import pandas as pd, seaborn as sns
            df = pd.DataFrame({'x': x_list, 'y': y_list})
            sns.violinplot(data=df, x='x', y='y', ax=ax)
            ax.set_title(configuracion.get('titulo', 'Diagrama de Violín'), pad=10)
            ax.set_ylabel(configuracion.get('ylabel', 'Valores'))
            return fig, ax
        except Exception as e:
            print(f"⚠️ seaborn/pandas no disponible: {e}. Uso Matplotlib.")
    if not (isinstance(data, list) and all(isinstance(lst, list) for lst in data)):
        raise ValueError("Para Matplotlib, 'data' debe ser lista de listas.")
    ax.violinplot(dataset=data, showmeans=True, showextrema=True, showmedians=True)
    if labels:
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set_title(configuracion.get('titulo', 'Diagrama de Violín'), pad=10)
    ax.set_ylabel(configuracion.get('ylabel', 'Valores'))
    return fig, ax


# 13) heatmap
@register_chart("heatmap")
def plugin_heatmap(datos, configuracion, debug=False):
    matrix = datos.get('matrix')
    if not (isinstance(matrix, list) and matrix and all(isinstance(r, list) for r in matrix)):
        raise ValueError("'matrix' debe ser lista de listas.")
    fig, ax = plt.subplots(figsize=(6, 5))
    annot = bool(configuracion.get('annot', False)); cmap = configuracion.get('cmap', 'viridis')
    if configuracion.get('use_seaborn', False):
        try:
            import seaborn as sns
            sns.heatmap(matrix, annot=annot, cmap=cmap, ax=ax)
        except Exception:
            im = ax.imshow(matrix, cmap=cmap, aspect='auto'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect='auto'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if annot:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    ax.text(j, i, str(matrix[i][j]), ha='center', va='center', fontsize=8)
    ax.set_title(configuracion.get('titulo', 'Mapa de Calor'), pad=10)
    return fig, ax


# 14) contour_plot
@register_chart("contour_plot")
def plugin_contour(datos, configuracion, debug=False):
    x = datos.get('x'); y = datos.get('y'); z = datos.get('z')
    if not (isinstance(x, list) and isinstance(y, list) and isinstance(z, list)):
        raise ValueError("'x','y' listas y 'z' matriz (lista de listas).")
    X, Y = np.meshgrid(x, y); Z = np.array(z)
    if Z.shape != X.shape:
        raise ValueError(f"Z{Z.shape} debe coincidir con grid {X.shape}.")
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X, Y, Z, levels=int(configuracion.get("levels", 10)), cmap=configuracion.get('cmap', 'viridis'))
    if configuracion.get("colorbar", True): plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(configuracion.get('titulo', 'Gráfico de Contorno'), pad=10)
    return fig, ax


# 15) 3d_plot
@register_chart("3d_plot")
def plugin_3d(datos, configuracion, debug=False):
    plot_type = str(configuracion.get('plot_type', 'scatter')).lower()
    fig = plt.figure(figsize=(6, 5)); ax = fig.add_subplot(111, projection='3d')
    if plot_type in ('scatter', 'line'):
        x, y, z = datos.get('x'), datos.get('y'), datos.get('z')
        if not (isinstance(x, list) and isinstance(y, list) and isinstance(z, list) and len(x) == len(y) == len(z)):
            raise ValueError("Para scatter/line: 'x','y','z' listas de igual longitud.")
        if plot_type == 'scatter': ax.scatter(np.array(x), np.array(y), np.array(z))
        else: ax.plot(np.array(x), np.array(y), np.array(z))
    else:
        X, Y, Z = np.array(datos.get('X')), np.array(datos.get('Y')), np.array(datos.get('Z'))
        if not (X.ndim == Y.ndim == Z.ndim == 2 and X.shape == Y.shape == Z.shape):
            raise ValueError("Para surface/wireframe: 'X','Y','Z' matrices 2D iguales.")
        if plot_type == 'surface': ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        elif plot_type == 'wireframe': ax.plot_wireframe(X, Y, Z)
        else: raise ValueError("plot_type no soportado: use scatter, line, surface o wireframe.")
    ax.set_title(configuracion.get('titulo', 'Gráfico 3D'), pad=10)
    ax.set_xlabel(configuracion.get('xlabel', 'Eje X')); ax.set_ylabel(configuracion.get('ylabel', 'Eje Y')); ax.set_zlabel(configuracion.get('zlabel', 'Eje Z'))
    return fig, ax


# 16) network_diagram
@register_chart("network_diagram")
def plugin_network(datos, configuracion, debug=False):
    nodes, edges = datos.get('nodes', []), datos.get('edges', [])
    node_labels = datos.get('labels', {}); directed = configuracion.get('directed', False)
    if not (isinstance(nodes, list) and isinstance(edges, list)):
        raise ValueError("'nodes' y 'edges' deben ser listas.")
    fig, ax = plt.subplots(figsize=(8, 6)); ax.set_title(configuracion.get("titulo", "Diagrama de Red"), pad=10); ax.axis('off')
    G = nx.DiGraph() if directed else nx.Graph(); G.add_nodes_from(nodes); G.add_edges_from(edges)
    layout = str(configuracion.get('layout', 'spring')).lower()
    try:
        if layout == 'sugiyama': pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        elif layout == 'planar': pos = nx.planar_layout(G)
        elif layout == 'kamada_kawai': pos = nx.kamada_kawai_layout(G)
        else: pos = nx.spring_layout(G)
    except Exception: pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=configuracion.get('with_labels', True), labels=node_labels,
        node_size=configuracion.get('node_size', 700), node_color=configuracion.get('node_color', 'skyblue'),
        font_size=configuracion.get('font_size', 10), edge_color=configuracion.get('edge_color', 'black'),
        ax=ax, **configuracion.get("plot_config", {})
    )
    parsed = {}
    for k, v in (configuracion.get('edge_labels', {}) or {}).items():
        if '-' in k:
            a, b = k.split('-', 1); a, b = a.strip(), b.strip()
            if (a, b) in G.edges() or (not directed and (b, a) in G.edges()):
                parsed[(a, b)] = v
    if parsed:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=parsed, ax=ax,
                                     font_size=configuracion.get('font_size', 10),
                                     **configuracion.get("edge_label_config", {}))
    return fig, ax


# 17) area_plot
@register_chart("area_plot")
def plugin_area(datos, configuracion, debug=False):
    if 'y' not in datos or not isinstance(datos['y'], list):
        raise ValueError("Falta 'y' como lista (serie única o lista de listas).")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(configuracion.get("titulo", "Gráfico de Área"), pad=10)
    ax.set_xlabel(configuracion.get("xlabel", "Eje X")); ax.set_ylabel(configuracion.get("ylabel", "Eje Y"))
    x_data = datos.get('x'); y_data = datos['y']
    if all(isinstance(v, (int, float)) for v in y_data):
        if x_data is not None and isinstance(x_data, list) and len(x_data) == len(y_data):
            ax.fill_between(x_data, y_data, **configuracion.get("plot_config", {}))
        else:
            ax.fill_between(range(len(y_data)), y_data, **configuracion.get("plot_config", {}))
            if configuracion.get("xlabel", "Eje X") == "Eje X": ax.set_xlabel("Índice")
    elif all(isinstance(lst, list) for lst in y_data):
        if not all(all(isinstance(v, (int, float)) for v in lst) for lst in y_data): raise ValueError("Series apiladas deben ser numéricas.")
        if not all(len(lst) == len(y_data[0]) for lst in y_data): raise ValueError("Series apiladas deben tener igual longitud.")
        y_stack = np.array(y_data).T
        if x_data is not None and isinstance(x_data, list) and len(x_data) == y_stack.shape[0]:
            ax.stackplot(x_data, y_stack, **configuracion.get("plot_config", {}))
        else:
            ax.stackplot(range(y_stack.shape[0]), y_stack, **configuracion.get("plot_config", {}))
            if configuracion.get("xlabel", "Eje X") == "Eje X": ax.set_xlabel("Índice")
    else:
        raise ValueError("'y' debe ser lista numérica o lista de listas.")
    return fig, ax


# 18) radar_chart
@register_chart("radar_chart")
def plugin_radar(datos, configuracion, debug=False):
    labels = datos.get('labels'); values_list = datos.get('values')
    if not (isinstance(labels, list) and isinstance(values_list, list) and labels):
        raise ValueError("'labels' lista y 'values' lista (o lista de listas) requeridas.")
    num_vars = len(labels)
    if all(isinstance(v, (int, float)) for v in values_list):
        values_list = [values_list]
    if not all(len(lst) == num_vars and all(isinstance(v, (int, float)) for v in lst) for lst in values_list):
        raise ValueError("Cada serie debe ser numérica y del mismo largo que 'labels'.")
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_title(configuracion.get("titulo", "Gráfico de Radar"), va="bottom", pad=10)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    for i, vals in enumerate(values_list):
        vals = np.concatenate((vals, [vals[0]]))
        ax.plot(angles, vals, label=configuracion.get(f"series_label_{i}", f"Series {i+1}"),
                **configuracion.get("plot_config", {}))
        ax.fill(angles, vals, alpha=configuracion.get("fill_alpha", 0.25), **configuracion.get("fill_config", {}))
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    if configuracion.get("show_legend", True) and len(values_list) > 1:
        ax.legend(loc=configuracion.get("legend_loc", "upper right"), bbox_to_anchor=configuracion.get("legend_bbox", (1.1, 1.1)))
    return fig, ax


# 19) venn_diagram
@register_chart("venn_diagram")
def plugin_venn(datos, configuracion, debug=False):
    subsets = datos.get('subsets')
    if not isinstance(subsets, (tuple, list)):
        raise ValueError("'subsets' debe ser tuple/list.")
    fig, ax = plt.subplots(figsize=(6, 4)); ax.set_title(configuracion.get("titulo", "Diagrama de Venn"), pad=10)
    if len(subsets) == 3:
        set_labels = configuracion.get('set_labels', None)
        if set_labels is not None and len(set_labels) != 2: set_labels = None
        venn2(subsets=subsets, set_labels=set_labels, ax=ax, **configuracion.get('plot_config', {}))
    elif len(subsets) == 7:
        set_labels = configuracion.get('set_labels', None)
        if set_labels is not None and len(set_labels) != 3: set_labels = None
        venn3(subsets=subsets, set_labels=set_labels, ax=ax, **configuracion.get('plot_config', {}))
    else:
        raise ValueError("Para Venn: 3 elementos (2 sets) o 7 (3 sets).")
    return fig, ax


# 20) fractal (mandelbrot/julia)
@register_chart("fractal")
def plugin_fractal(datos, configuracion, debug=False):
    ftype = str(datos.get('type', 'mandelbrot')).lower().strip()
    cfg = datos.get('config', {}) or {}
    if ftype == 'mandelbrot':
        width, height = int(cfg.get('width', 400)), int(cfg.get('height', 400))
        xmin, xmax = cfg.get('xmin', -2.0), cfg.get('xmax', 1.0)
        yres = cfg.get('yres', (xmax - xmin) * height / width)
        ymin, ymax = cfg.get('ymin', -yres/2), cfg.get('ymax', yres/2)
        max_iter = int(cfg.get('max_iter', 100)); cmap = configuracion.get('cmap', 'hot')
        img = np.zeros((height, width), dtype=np.uint16)
        x_vals = np.linspace(xmin, xmax, width); y_vals = np.linspace(ymin, ymax, height)
        for i in range(height):
            for j in range(width):
                c = complex(x_vals[j], y_vals[i]); z = 0
                for k in range(max_iter):
                    z = z*z + c
                    if abs(z) > 2: img[i, j] = k; break
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(img, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=cmap, **configuracion.get("plot_config", {}))
        ax.set_title(configuracion.get("titulo", "Conjunto de Mandelbrot"), pad=10)
        ax.set_xlabel(configuracion.get("xlabel", "Re(c)")); ax.set_ylabel(configuracion.get("ylabel", "Im(c)"))
        return fig, ax
    elif ftype == 'julia':
        width, height = int(cfg.get('width', 400)), int(cfg.get('height', 400))
        xmin, xmax = cfg.get('xmin', -1.5), cfg.get('xmax', 1.5)
        yres = cfg.get('yres', (xmax - xmin) * height / width)
        ymin, ymax = cfg.get('ymin', -yres/2), cfg.get('ymax', yres/2)
        max_iter = int(cfg.get('max_iter', 100))
        c_const = complex(cfg.get('c_real', -0.7), cfg.get('c_imag', 0.27015))
        cmap = configuracion.get('cmap', 'hot')
        img = np.zeros((height, width), dtype=np.uint8)
        x_vals = np.linspace(xmin, xmax, width); y_vals = np.linspace(ymin, ymax, height)
        for i in range(height):
            for j in range(width):
                z = complex(x_vals[j], y_vals[i])
                for k in range(max_iter):
                    z = z*z + c_const
                    if abs(z) > 2: img[i, j] = k; break
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(img, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap=cmap, **configuracion.get("plot_config", {}))
        ax.set_title(configuracion.get("titulo", "Conjunto de Julia"), pad=10)
        ax.set_xlabel(configuracion.get("xlabel", "Re(z)")); ax.set_ylabel(configuracion.get("ylabel", "Im(z)"))
        return fig, ax
    else:
        raise ValueError(f"Tipo de fractal no soportado: '{ftype}'")

# Catálogo/alias para robustecer reconocimiento del LLM
CHART_ALIASES = {
    "grafico_barras_verticales": "grafico_barras_verticales",
    "barras_verticales": "grafico_barras_verticales",
    "bar": "grafico_barras_verticales",
    "pie": "grafico_circular",
    "grafico_circular": "grafico_circular",
    "tabla": "tabla",
    "construccion_geometrica": "construccion_geometrica",
    "diagrama_arbol": "diagrama_arbol",
    "flujograma": "flujograma",
    "pictograma": "pictograma",
    "waffle": "pictograma",
    "scatter_plot": "scatter_plot",
    "line_plot": "line_plot",
    "histogram": "histogram",
    "box_plot": "box_plot",
    "violin_plot": "violin_plot",
    "heatmap": "heatmap",
    "contour_plot": "contour_plot",
    "3d_plot": "3d_plot",
    "network_diagram": "network_diagram",
    "area_plot": "area_plot",
    "radar_chart": "radar_chart",
    "venn_diagram": "venn_diagram",
    "fractal": "fractal",
}

# ==============================================================================
# 3. MOTOR PRINCIPAL DE GENERACIÓN DE GRÁFICOS
# ==============================================================================

def crear_grafico(tipo_grafico, datos, configuracion):
    """Motor que usa el sistema de plugins para generar un gráfico."""
    plugin_key = _resolve_plugin_key(tipo_grafico)

    if plugin_key not in PLUGIN_REGISTRY:
        raise ValueError(f"El tipo de gráfico '{tipo_grafico}' no está soportado.")

    plugin_function = PLUGIN_REGISTRY[plugin_key]

    fig, ax = None, None
    try:
        resultado = plugin_function(datos, configuracion)

        if isinstance(resultado, io.BytesIO):
            return resultado
        elif isinstance(resultado, tuple) and len(resultado) == 2:
            fig, ax = resultado
        else:
            raise TypeError(f"El plugin '{plugin_key}' devolvió un tipo de resultado inesperado.")

        if fig:
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return buf
        else:
            raise ValueError(f"El plugin '{plugin_key}' no generó una figura de Matplotlib válida.")

    except Exception as e:
        print(f"❌ Error al ejecutar el plugin '{plugin_key}': {e}")
        if fig:
            plt.close(fig)
        return None

# ==============================================================================
# 4. ORQUESTADOR CON INTELIGENCIA ARTIFICIAL
# ==============================================================================

VISUAL_SPEC_TEMPLATE = """
Eres un generador que devuelve EXCLUSIVAMENTE un objeto JSON válido con la especificación de UN (1) elemento visual.

Campos requeridos:
- "tipo_elemento": uno de: grafico_barras_verticales, grafico_circular, tabla, construccion_geometrica, diagrama_arbol,
  flujograma, pictograma, scatter_plot, line_plot, histogram, box_plot, violin_plot, heatmap, contour_plot, 3d_plot,
  network_diagram, area_plot, radar_chart, venn_diagram, fractal
- "datos": objeto con los datos que requiere ese tipo.
- "configuracion": objeto con opciones (por ejemplo "titulo").
- "ubicacion": siempre "enunciado".

Notas importantes:
- Devuelve SOLO un JSON (sin comentarios ni texto extra).
- Si el tipo es "flujograma", debes construir "dot_source" con lenguaje DOT, p.ej:
  "dot_source": "digraph G {{ A->B; B->C; }}"
- Para "tabla": "matrix" es [[enc1, enc2], [f1c1, f1c2], ...]
- Para "venn_diagram": "subsets" debe ser longitud 3 (2 conjuntos) o 7 (3 conjuntos).
- Para "fractal": "type" es "mandelbrot" o "julia" y "config" con parámetros.

LINEAMIENTOS RÁPIDOS:
- grafico_barras_verticales: {{ "X":[...], "Y":[...] }} (claves arbitrarias; 1 lista texto/números, 1 lista numérica)
- grafico_circular: {{ "Etiquetas":[...], "Valores":[...] }}
- tabla: {{ "matrix":[[enc1, enc2, ...], [fila1c1, fila1c2, ...], ...],
          "configuracion": {{ "titulo":"...", "cellLoc":"left",
                             "figsize":[11, 6.5],
                             "col_widths":[0.14, 0.56, 0.30],
                             "fontsize":9, "fill_axes":true }} }}
- construccion_geometrica: {{ "elements":[
    {{ "type":"point","coords":[x,y],"config":{{"label":"A"}} }},
    {{ "type":"line","coords":[[x1,y1],[x2,y2]],"config":{{}} }},
    {{ "type":"polygon","coords":[[x,y],...],"config":{{"facecolor":"#...","alpha":0.3}} }},
    {{ "type":"circle","config":{{"center":[cx,cy],"radius":r,"patch_config":{{"edgecolor":"k"}}}} }},
    {{ "type":"arrow","config":{{"start":[x1,y1],"end":[x2,y2],"patch_config":{{"arrowstyle":"->"}}}} }}
  ]}}
- diagrama_arbol/network_diagram: {{ "nodes":[...], "edges":[["A","B"],...], "labels":{{"A":"Raíz"}} }}
- flujograma: {{ "dot_source": "digraph G {{ A->B; B->C; }}" }}
- pictograma: {{ "values":{{"CatA":10,"CatB":5}}, "colors":["#...","#..."] }}
- scatter_plot/line_plot: {{ "x":[...], "y":[...] }}
- histogram: {{ "values":[...] }}
- box_plot: {{ "data":[[...],[...]] }} o dict de listas
- violin_plot: {{ "data":[[...],[...]] }} (o x/y si prefieres seaborn)
- heatmap: {{ "matrix":[[...], [...], ...] }}
- contour_plot: {{ "x":[...], "y":[...], "z":[[...], ...] }}
- 3d_plot: {{ "x":[...], "y":[...], "z":[...] }} (scatter/line) o {{ "X":[[..]], "Y":[[..]], "Z":[[..]] }} (surface/wireframe)
- area_plot: {{ "y":[...] }} o {{ "x":[...], "y":[[...],[...]] }}
- radar_chart: {{ "labels":[...], "values":[...] }} (o lista de listas)
- venn_diagram: {{ "subsets":(a,b,ab) }} (2 sets) o 7-tuple (3 sets)
- fractal: {{ "type":"mandelbrot","config":{{"width":400,"height":400,"max_iter":100}} }}

REGLAS:
- Devuelve **solo** un objeto JSON, sin texto adicional.
- Usa nombres de claves y estructura correctos para el tipo elegido.
- Incluye "configuracion": {{"titulo":"..."}} coherente con la descripción.

DESCRIPCION:
{descripcion}
"""

def build_visual_json_with_llm(descripcion: str) -> dict:
    """Llama al LLM para convertir una descripción textual en una especificación JSON."""
    prompt = PromptTemplate(input_variables=["descripcion"], template=VISUAL_SPEC_TEMPLATE)

    # Evita que llaves en la descripción rompan el PromptTemplate
    descripcion_safe = _escape_braces(descripcion)

    # Construye la cadena prompt -> LLM (faltaba esto)
    llm = _get_llm()
    chain = prompt | llm

    # Invoca y extrae texto
    response = chain.invoke({"descripcion": descripcion_safe})
    content = getattr(response, "content", str(response))

    # Intenta parsear el primer objeto JSON de la respuesta
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No se encontró un objeto JSON en la respuesta del LLM.", content, 0)
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        print("❌ Error: La respuesta del LLM no fue un JSON válido.")
        print("Respuesta recibida:", content)
        raise

def generar_grafico_desde_texto(descripcion: str, ruta_png=None,
                                mostrar: bool=False, abrir_archivo: bool=False):
    """Función principal y orquestadora."""
    print(f"🧠 Generando especificación JSON para: '{descripcion}'...")
    try:
        spec = build_visual_json_with_llm(descripcion)
        print("✅ JSON generado con éxito.")
        print(json.dumps(spec, indent=2))
    except Exception as e:
        print(f"❌ Falló la generación de JSON: {e}")
        return None, None

    print("\n🎨 Renderizando el gráfico...")
    try:
        buffer = crear_grafico(
            spec.get("tipo_elemento"),
            spec.get("datos", {}),
            spec.get("configuracion", {})
        )
        if not buffer:
            print("❌ El motor de gráficos no pudo generar una imagen.")
            return spec, None

        print("✅ Gráfico renderizado con éxito.")

        # Guardar PNG si se solicitó
        if ruta_png:
            with open(ruta_png, "wb") as f:
                f.write(buffer.getvalue())
            print(f"💾 Gráfico guardado en: {ruta_png}")

            # Abrir con el visor por defecto del SO
            if abrir_archivo:
                import sys, subprocess, os
                try:
                    if os.name == "nt":
                        os.startfile(ruta_png)                       # Windows
                    elif sys.platform == "darwin":
                        subprocess.run(["open", ruta_png], check=False)  # macOS
                    else:
                        subprocess.run(["xdg-open", ruta_png], check=False)  # Linux
                except Exception as e:
                    print(f"⚠️ No se pudo abrir el archivo automáticamente: {e}")

        # Mostrar en pantalla con Matplotlib (sin depender de archivo)
        if mostrar:
            try:
                from PIL import Image
                import io as _io
                img = Image.open(_io.BytesIO(buffer.getvalue()))
                plt.figure()
                plt.imshow(img)
                plt.axis("off")
                plt.show()
            except Exception as e:
                print(f"⚠️ No se pudo mostrar el gráfico en ventana: {e}")

        return spec, buffer

    except Exception as e:
        print(f"❌ Falló el renderizado del gráfico: {e}")
        return spec, None

# ==============================================================================
# 5. EJEMPLO DE USO INTERACTIVO
# ==============================================================================
if __name__ == '__main__':
    # Bucle infinito para mantener el programa corriendo
    while True:
        # 1. Pedir la descripción al usuario
        descripcion_usuario = input("✍️ Describe el gráfico que quieres (o escribe 'salir' para terminar): ")

        # 2. Condición para salir del bucle
        if descripcion_usuario.lower() in ['salir', 'exit', 'quit']:
            print("👋 ¡Hasta luego!")
            break

        # 3. Verificar que no esté vacío
        if not descripcion_usuario.strip():
            print("❌ Por favor, introduce una descripción válida.")
            continue

        # 4. Llamar a la función principal con la descripción del usuario
        generar_grafico_desde_texto(
            descripcion_usuario,
            ruta_png="grafico_generado.png", # Se guardará siempre con este nombre
            mostrar=True, # Para que siempre se muestre la ventana con el gráfico
            abrir_archivo=False # Opcional, puedes ponerlo en True si quieres que también se abra el archivo
        )

        print("\n" + "="*50 + "\n")
