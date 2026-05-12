# Vision AI — Reconocimiento de Imágenes Local

Aplicación web de reconocimiento de imágenes que corre completamente en local, sin APIs de pago.

**Modelos utilizados**
- **Swin-Large-384** — Clasificación top-5 con TTA (87% top-1 en ImageNet)
- **BLIP-large** — Descripción en lenguaje natural traducida al español

---

## Requisitos

- Python 3.10 o superior
- GPU NVIDIA recomendada (funciona en CPU también, más lento)
- ~3 GB de espacio libre para los modelos (se descargan automáticamente)

---

## Instalación

```bash
git clone https://github.com/Politi23/Modelo-de-IA-para-Reconocimiento-de-Imagenes-Local.git
cd Modelo-de-IA-para-Reconocimiento-de-Imagenes-Local

python setup.py
```

El script detecta tu GPU automáticamente e instala la versión correcta de PyTorch (CUDA 12.8 para RTX 5000, CUDA 12.6 para RTX 4000, etc.).

---

## Uso

```bash
python app.py
```

Abre el navegador en **http://127.0.0.1:5000**

La primera vez descarga los modelos (~1.7 GB en total). Quedan en caché para usos futuros.

---

## Solución de problemas

**"CUDA not compatible"** — Tu GPU es muy nueva para el PyTorch instalado. Ejecuta:
```bash
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

**Predicciones en inglés** — Borra `labels_es.json` y reinicia el servidor para regenerar las traducciones.

**Puerto en uso** — Cambia el puerto en la última línea de `app.py`:
```python
app.run(debug=False, port=5001)
```
