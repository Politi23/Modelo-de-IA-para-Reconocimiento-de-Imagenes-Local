"""
Setup script para Vision AI — detecta GPU y instala dependencias correctas.
Uso: python setup.py
"""

import subprocess
import sys
import platform


def run(cmd, check=True):
    print(f"  > {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check)
    return result.returncode == 0


def pip(*args):
    return run([sys.executable, "-m", "pip", "install", "--upgrade", *args])


def check_python():
    major, minor = sys.version_info[:2]
    print(f"Python {major}.{minor} detectado.")
    if major < 3 or minor < 10:
        print("ERROR: Se requiere Python 3.10 o superior.")
        print("Descarga desde https://www.python.org/downloads/")
        sys.exit(1)


def detect_cuda_capability():
    """Retorna la compute capability de la GPU NVIDIA, o None si no hay."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip().split("\n")[0].strip()
        major, minor = out.split(".")
        return int(major), int(minor)
    except Exception:
        return None


def get_torch_index(cap):
    """Elige el wheel index de PyTorch según la compute capability."""
    if cap is None:
        return None, "cpu"
    major, _ = cap
    # sm_120+ = Blackwell (RTX 5000)  → cu128
    # sm_89-sm_90 = Ada/Hopper        → cu124 o cu126
    # sm_80-sm_86 = Ampere             → cu124
    # < sm_80                          → cu121
    if major >= 12:
        return "https://download.pytorch.org/whl/cu128", "cu128"
    elif major >= 9:
        return "https://download.pytorch.org/whl/cu126", "cu126"
    elif major >= 8:
        return "https://download.pytorch.org/whl/cu124", "cu124"
    else:
        return "https://download.pytorch.org/whl/cu121", "cu121"


def install_torch(index_url, label):
    pkgs = ["torch", "torchvision"]
    if index_url:
        cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall",
               "--index-url", index_url, *pkgs]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *pkgs]
    print(f"  > Instalando PyTorch ({label})...")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print("  WARN: falló la instalación de PyTorch, intenta manualmente:")
        print(f"        pip install torch torchvision --index-url {index_url}")
    return result.returncode == 0


def verify_torch():
    try:
        code = (
            "import torch; "
            "print('PyTorch', torch.__version__); "
            "cuda = torch.cuda.is_available(); "
            "print('CUDA disponible:', cuda); "
            "print('GPU:', torch.cuda.get_device_name(0) if cuda else 'N/A')"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
        return True
    except Exception:
        return False


def main():
    print("=" * 55)
    print("  Vision AI — Instalación automática")
    print("=" * 55)
    print()

    # 1. Python
    print("[1/5] Verificando Python...")
    check_python()
    print()

    # 2. pip
    print("[2/5] Actualizando pip...")
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=False)
    print()

    # 3. PyTorch con CUDA correcto
    print("[3/5] Detectando GPU...")
    cap = detect_cuda_capability()
    if cap:
        print(f"  GPU NVIDIA detectada — compute capability sm_{cap[0]}{cap[1]}")
        index_url, label = get_torch_index(cap)
    else:
        print("  No se detectó GPU NVIDIA — instalando PyTorch CPU")
        index_url, label = None, "cpu"

    print(f"  Usando PyTorch {label} (esto puede tardar varios minutos)...")
    install_torch(index_url, label)
    print()

    # 4. Resto de dependencias
    print("[4/5] Instalando dependencias del proyecto...")
    deps = [
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "transformers>=4.40.0",
        "Pillow>=10.0.0",
        "deep-translator>=1.11.0",
        "accelerate",
    ]
    for dep in deps:
        pip(dep)
    print()

    # 5. Verificación final
    print("[5/5] Verificando instalación...")
    ok = verify_torch()
    print()

    print("=" * 55)
    if ok:
        print("  Instalacion completada.")
        print()
        print("  Para iniciar el servidor:")
        print("    python app.py")
        print()
        print("  Luego abre en el navegador:")
        print("    http://127.0.0.1:5000")
        print()
        print("  NOTA: La primera vez descarga los modelos de IA")
        print("  (~1.7 GB). Quedan en cache para usos futuros.")
    else:
        print("  Algo falló en la verificación.")
        print("  Revisa los mensajes anteriores e intenta:")
        print("    pip install torch torchvision --force-reinstall")
    print("=" * 55)


if __name__ == "__main__":
    main()
