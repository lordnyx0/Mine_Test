# coding=utf-8
"""
modal_treino.py — Pipeline de Treinamento Cloud no Modal para Qwen3Loop VLA Minecraft.

Aproveita GPUs Cloud (A10G / L4 / A100) com cobrança por segundo no Modal:
- Sobe o ambiente Node.js do servidor offline e o simulador de 16 a 32 bots.
- Roda o PPO de Raciocínio Lógico (Fase 4) em velocidade ultra-rápida.
- Persiste checkpoints no modal.Volume('minecraft-vla-checkpoints').

Uso:
    modal run infra/modal_treino.py --fase 4 --iteracoes 80 --gpu A10G
    modal volume get minecraft-vla-checkpoints vla_fase4_logica.pt ./checkpoints_vla/
"""
import os
import modal

# 1. Definição do App no Modal
app = modal.App("minecraft-qwen3loop-vla")

# 2. Volume Persistente para Checkpoints e Mundo Minecraft
volume_checkpoints = modal.Volume.from_name("minecraft-vla-checkpoints", create_if_missing=True)
volume_mundo = modal.Volume.from_name("minecraft-world-save", create_if_missing=True)

# 3. Imagem do Container: Debian + Python 3.11 + Node.js 20 + Canvas + PyTorch CUDA
imagem_treino = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "libcairo2-dev",
        "libpango1.0-dev",
        "libjpeg-dev",
        "libgif-dev",
        "librsvg2-dev"
    )
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
        "node -v",
        "npm -v"
    )
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.43.0",
        "pillow",
        "requests",
        "numpy",
        "tqdm",
        "pyyaml"
    )
)


@app.function(
    image=imagem_treino,
    gpu="A10G",  # Opções: "L4", "A10G", "A100", "H100"
    timeout=7200,  # 2 horas max
    volumes={
        "/root/checkpoints_vla": volume_checkpoints,
        "/root/world": volume_mundo
    },
    mounts=[modal.Mount.from_local_dir(".", remote_path="/root/minecraft_adapter")]
)
def executar_treino_fase4(iteracoes: int = 80, epocas: int = 2, lr: float = 2.0e-5, n_envs: int = 16):
    """Executa o treinamento da Fase 4 dentro do container GPU no Modal."""
    import subprocess
    import sys
    import time
    import urllib.request
    import os

    os.chdir("/root/minecraft_adapter")

    print("=" * 70)
    print(f"🚀 INICIANDO TREINAMENTO NO MODAL CLOUD ({iteracoes} iterações, {n_envs} bots)")
    print("=" * 70)

    # 1. Instala dependências do Node.js se necessário
    print("[1/4] Preparando servidor offline Node.js...")
    subprocess.run(["npm", "install", "--omit=optional"], cwd="mineflayer_server", check=True)

    # 2. Inicia o servidor offline em background na porta 3002
    env_node = os.environ.copy()
    env_node["PORTA_OFFLINE"] = "3002"
    env_node["N_ENVS"] = str(n_envs)
    if os.path.exists("/root/world/region"):
        env_node["SAVE_PATH"] = "/root/world"

    proc_servidor = subprocess.Popen(
        ["node", "servidor_offline.js"],
        cwd="mineflayer_server",
        env=env_node
    )

    # Aguarda o servidor subir
    print("[2/4] Aguardando servidor offline inicializar...")
    pronto = False
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:3002/lote/info", timeout=2) as resp:
                if resp.status == 200:
                    pronto = True
                    break
        except Exception:
            time.sleep(1.0)

    if not pronto:
        proc_servidor.kill()
        raise RuntimeError("Falha ao inicializar servidor offline no Modal.")

    print(f"[OK] Servidor offline ativo com {n_envs} ambientes em memória!")

    # 3. Executa o treinamento da Fase 4
    print(f"[3/4] Disparando loop de treino com LoRA e Raciocínio Multi-Loop...")
    cmd_treino = [
        sys.executable, "treino/treinar_fase4_logica.py",
        "--iteracoes", str(iteracoes),
        "--epocas", str(epocas),
        "--lr", str(lr),
        "--loops", "3",
        "--ckpt-entrada", "checkpoints_vla/vla_fase3_merged.pt"
    ]

    p_treino = subprocess.run(cmd_treino, cwd="/root/minecraft_adapter")

    # 4. Salva no volume persistente
    print("[4/4] Salvando checkpoints no Volume Cloud persistente...")
    if os.path.exists("checkpoints_vla/vla_fase4_logica.pt"):
        import shutil
        shutil.copy("checkpoints_vla/vla_fase4_logica.pt", "/root/checkpoints_vla/vla_fase4_logica.pt")
        volume_checkpoints.commit()
        print("✅ Checkpoint salvo com sucesso no Modal Volume!")

    proc_servidor.kill()
    return p_treino.returncode == 0


@app.local_entrypoint()
def main(iteracoes: int = 80, gpu: str = "A10G", n_envs: int = 16):
    print(f"Disparando treino no Modal com GPU={gpu}, {iteracoes} iterações...")
    sucesso = executar_treino_fase4.remote(iteracoes=iteracoes, n_envs=n_envs)
    if sucesso:
        print("\n🎉 Treinamento concluído com sucesso no Modal!")
        print("Para baixar o checkpoint treinado para seu PC, execute:")
        print("modal volume get minecraft-vla-checkpoints vla_fase4_logica.pt ./checkpoints_vla/")
    else:
        print("\n⚠️ O treino encontrou um erro no container.")
