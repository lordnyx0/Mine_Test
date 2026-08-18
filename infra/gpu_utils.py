# coding=utf-8
"""
Utilidades para não derrubar a máquina durante o treino.

O que resolve, em ordem de impacto:

  1. TRAVA DE GPU — só um processo de treino por vez. Dois modelos completos
     num 3060 de 12 GB estouram a VRAM e travam o Windows inteiro.
  2. BACKBONE EM bf16 — Qwen e SigLIP estão congelados; manter os pesos deles
     em fp32 é desperdício. bf16 corta a VRAM pela metade e acelera. Só os
     módulos treináveis ficam em fp32, que é onde a precisão importa.
  3. LIMITE DE CPU/PRIORIDADE — mantém o PC usável enquanto treina.
"""
import os
import sys
import time
import atexit

LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gpu_em_uso.lock")


def limitar_vram(fracao=0.65):
    """
    Teto de VRAM para o processo. O Minecraft tambem usa a GPU; deixar o treino
    encostar nos 12GB fez o Windows inteiro travar. Estourar o teto vira um
    OOM do PyTorch (falha alta e recuperavel) em vez de congelar a maquina.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.set_per_process_memory_fraction(fracao, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        return f"{fracao*100:.0f}% de {total:.1f}GB = {fracao*total:.1f}GB"
    except Exception:
        return None


def limitar_recursos(threads=2, prioridade_baixa=True):
    """Deixa a máquina respirável: poucas threads e prioridade abaixo do normal."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    torch.set_num_threads(threads)
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if prioridade_baixa:
        try:
            import psutil
            p = psutil.Process()
            if sys.platform == "win32":
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                p.nice(10)
        except Exception:
            pass


def travar_gpu(espera_max=0, obrigatorio=True):
    """
    Garante um único processo de GPU. Se outro já estiver rodando, aborta
    (ou espera, se espera_max > 0).
    """
    t0 = time.time()
    while True:
        if not os.path.exists(LOCK):
            break
        try:
            with open(LOCK) as f:
                dono = f.read().strip()
        except OSError:
            dono = "?"
        # Lock órfão de um processo morto
        try:
            import psutil
            pid = int(dono.split("|")[0])
            if not psutil.pid_exists(pid):
                os.remove(LOCK)
                continue
        except Exception:
            pass

        if time.time() - t0 >= espera_max:
            if obrigatorio:
                print(f"[gpu] JA EXISTE um treino usando a GPU ({dono}).")
                print(f"[gpu] Encerre-o ou apague '{LOCK}'. Abortando para nao travar o PC.")
                sys.exit(1)
            return False
        time.sleep(5)

    with open(LOCK, "w") as f:
        f.write(f"{os.getpid()}|{time.strftime('%H:%M:%S')}|{' '.join(sys.argv[:2])}")
    atexit.register(destravar_gpu)
    return True


def destravar_gpu():
    try:
        os.remove(LOCK)
    except OSError:
        pass


def compactar_backbone(vla, dtype=None):
    """
    Converte os módulos CONGELADOS para bf16 e mantém os treináveis em fp32.
    Retorna o dtype usado (None se não aplicou).
    """
    import torch
    if not torch.cuda.is_available():
        return None
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    congelados = 0
    for modulo in (vla.qwen_model, vla.vision_encoder):
        if all(not p.requires_grad for p in modulo.parameters()):
            modulo.to(dtype)
            congelados += 1

    # Treináveis permanecem em fp32. Lista à mão ja divergiu 1x (goal_encoder
    # ficou de fora e voltava aleatorio, ver docs/metodo.md §12) — mantida
    # explicita aqui, mas o save/load usa named_parameters(requires_grad) como
    # fonte de verdade, entao um modulo esquecido aqui nao perde o treino, so
    # fica em fp32 por default do torch em vez de compactado.
    for m in (vla.resampler, vla.projector, vla.state_encoder, vla.action_heads,
             vla.goal_encoder):
        m.to(torch.float32)

    return dtype if congelados else None


def memoria_gpu():
    import torch
    if not torch.cuda.is_available():
        return "sem cuda"
    usado = torch.cuda.memory_allocated() / 2**30
    reservado = torch.cuda.memory_reserved() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"{usado:.2f}GB usados / {reservado:.2f}GB reservados / {total:.1f}GB"
