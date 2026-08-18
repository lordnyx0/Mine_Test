# coding=utf-8
"""
Mineflayer VLA Agent — Arquitetura Nova (sem DirectInput, sem ScreenCapture)

Loop autônomo completo:
  1. BotVisionCapture.capture()  → GET /frame   → PIL Image (1ª pessoa do bot)
  2. Qwen3LoopVLA.forward()      → action_dict  (WASD + câmera)
  3. send_action()               → POST /action → bot.setControlState() no jogo
  4. /delta                      → PositionRewardEvaluator → reward (+1/-1)
  5. TrajectoryLogger            → grava em dataset/live_trajectories.jsonl
  6. A cada 200 passos com ≥50 positivos → retrain_self_imitation.py
"""
import os
import sys
import time
import json
import io
import base64
import subprocess
import random
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

TESTES_DIR = r"C:\Users\Nyx\Desktop\Testes"
if TESTES_DIR not in sys.path and os.path.exists(TESTES_DIR):
    sys.path.append(TESTES_DIR)

from qwen3loop.modeling_qwen3loop  import Qwen3LoopModel
from qwen3loop.configuration_qwen3loop import Qwen3LoopConfig
from modelo.vla_model               import Qwen3LoopVLA
from infra.bot_vision_capture       import BotVisionCapture, send_action
from infra.trajectory_logger        import TrajectoryLogger
from infra.position_reward_evaluator import PositionRewardEvaluator

LIVE_DATASET_PATH   = "dataset/live_trajectories.jsonl"
CHECKPOINT_PATH     = "checkpoints_vla/vla_locomotion.pt"
# Backbone base pré-treinado Qwen3Loop
_LOCAL_BB = os.path.join(_ROOT, "checkpoints_vla", "backbone_base")
BACKBONE_DIR = _LOCAL_BB if os.path.exists(_LOCAL_BB) else "C:/Users/Nyx/Desktop/Testes/checkpoints/qwen3loop_v2/final_model"
RETRAIN_EVERY_STEPS = 200
MIN_POSITIVE_SAMPLES = 50


def load_vla_agent(caminho=None):
    """caminho=None usa CHECKPOINT_PATH (a base pre-treinada, imutavel)."""
    caminho = caminho or CHECKPOINT_PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== CARREGANDO QWEN3LOOP VLA ({device.upper()}) ===", flush=True)

    caminho_bb = os.environ.get("QWEN3LOOP_DIR", BACKBONE_DIR)

    # A config vem do PROPRIO diretorio do modelo, nao codificada aqui. A versao
    # codificada omitia `intermediate_size` e pegava o padrao 22016 contra os
    # 3072 reais — MLPs 7x maiores que os do modelo treinado. Config duplicada a
    # mao e a mesma familia de bug que ja custou caro neste projeto.
    if os.path.exists(os.path.join(caminho_bb, "config.json")):
        config = Qwen3LoopConfig.from_pretrained(caminho_bb)
    else:
        config = Qwen3LoopConfig(
            vocab_size=151936, hidden_size=1024, num_hidden_layers=28,
            num_attention_heads=16, num_key_value_heads=8,
            num_loops=2, enable_double_loop_split=True
        )
    qwen_model = Qwen3LoopModel(config).to(device)

    # ── O backbone PRECISA ser carregado ──────────────────────────────────────
    # Ate 2026-08-13 esta linha nao existia: `Qwen3LoopModel(config)` nasce com
    # pesos ALEATORIOS e nunca eram substituidos. Todo o VLA rodou sobre ruido.
    #
    # O sintoma que denunciou: treino reportava 100% e qualquer script de
    # avaliacao dava 8%. `treinar_fase1.py` chama torch.manual_seed(0), entao
    # sorteava sempre o MESMO backbone aleatorio e os adaptadores aprendiam
    # contra ele; scripts sem semente sorteavam outro e os adaptadores viravam
    # lixo. Nao era bug de laco, de pooling nem de arena.
    arq = os.path.join(caminho_bb, "model.safetensors")
    if os.path.exists(arq):
        from safetensors.torch import load_file
        sd = load_file(arq)
        # O save e no formato ForCausalLM, com tudo sob `model.`; o
        # Qwen3LoopModel espera `layers.*`, `embed_tokens.*`, `norm.*`.
        sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        r = qwen_model.load_state_dict(sd, strict=False)
        faltando_bb = [k for k in r.missing_keys if "rotary" not in k]
        print(f"[VLA] backbone carregado de {caminho_bb} "
              f"({len(sd)} tensores, {len(faltando_bb)} ausentes, "
              f"{len(r.unexpected_keys)} inesperados)", flush=True)
        if faltando_bb:
            print(f"[AVISO] pesos do backbone AUSENTES: {faltando_bb[:4]}", flush=True)
    else:
        print(f"[ALERTA] backbone treinado nao encontrado em {caminho_bb}. "
              f"O modelo esta rodando com pesos ALEATORIOS.", flush=True)
    vla_model  = Qwen3LoopVLA(
        qwen_model=qwen_model,
        vision_model_name="google/siglip-base-patch16-224",
        num_visual_tokens=32,
        freeze_vision=True,
        freeze_qwen=True
    ).to(device)

    if os.path.exists(caminho):
        print(f"[VLA] Carregando checkpoint '{caminho}'...", flush=True)
        ckpt = torch.load(caminho, map_location=device)
        # As chaves por submodulo sao LEGADO: checkpoints novos guardam tudo em
        # `treinaveis`, por nome, e nao as tem. Carregar so o que existir.
        if "resampler" in ckpt:
            vla_model.resampler.load_state_dict(ckpt["resampler"])
        if "projector" in ckpt:
            vla_model.projector.load_state_dict(ckpt["projector"])
        # O state_encoder cresceu de 16 para 32 entradas (historico temporal).
        # Copia as 16 colunas antigas e zera as novas: o comportamento aprendido
        # e preservado exatamente, e as dims novas comecam neutras.
        se_novo = vla_model.state_encoder.state_dict()
        se_velho = ckpt.get("state_encoder") or {}
        k = "mlp.0.weight"
        if k in se_velho and se_velho[k].shape != se_novo[k].shape:
            w = se_novo[k].clone()
            w.zero_()
            n_ant = se_velho[k].shape[1]
            w[:, :n_ant] = se_velho[k]
            se_velho = dict(se_velho)
            se_velho[k] = w
            print(f"[VLA] state_encoder migrado de {n_ant} para {w.shape[1]} entradas.", flush=True)
        if se_velho:
            vla_model.state_encoder.load_state_dict(se_velho)
        # strict=False: cabecas novas (yaw/pitch discretos, previsao de rota)
        # nao existem em checkpoints antigos e devem manter a inicializacao.
        faltando = vla_model.action_heads.load_state_dict(
            ckpt["action_heads"], strict=False).missing_keys if "action_heads" in ckpt else []
        if faltando:
            novas = sorted({k.split(".")[0] for k in faltando})
            print(f"[VLA] cabecas novas inicializadas do zero: {', '.join(novas)}", flush=True)
        # goal_encoder so existe nos checkpoints condicionados a objetivo. Sem
        # ele o modelo roda igual ao de antes (o forward ignora goal_vec=None),
        # entao checkpoint antigo continua valido.
        if "goal_encoder" in ckpt:
            vla_model.goal_encoder.load_state_dict(ckpt["goal_encoder"])
            print("[VLA] goal_encoder carregado (politica condicionada a objetivo).", flush=True)
        # `treinaveis` cobre TODO parametro com requires_grad, por nome. Existe
        # porque a lista de submodulos acima deixava `frame_time_embed` de fora,
        # e um parametro treinado que volta aleatorio e um bug silencioso: o
        # treino reporta um numero e o checkpoint recarregado entrega outro.
        if "treinaveis" in ckpt:
            faltou = vla_model.load_state_dict(ckpt["treinaveis"], strict=False)
            n = len(ckpt["treinaveis"])
            print(f"[VLA] {n} tensores treinaveis restaurados por nome.", flush=True)
        print("[VLA] Pesos carregados com sucesso.", flush=True)
    else:
        print(f"[AVISO] Checkpoint nao encontrado. Usando pesos iniciais.", flush=True)

    vla_model.eval()
    return vla_model, device


def img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def trigger_self_imitation():
    script = "retrain_self_imitation.py"
    if os.path.exists(script):
        print("\n[Self-Imitation] Iniciando fine-tuning...", flush=True)
        flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
        subprocess.Popen([sys.executable, script], cwd=os.getcwd(), creationflags=flags)


# ── Exploração ────────────────────────────────────────────────────────────────
# Sem isso a política determinística nunca descobre a recompensa de andar.
EXPLORE_EPS       = float(os.environ.get("EXPLORE_EPS", "0.15"))
STOCHASTIC_POLICY = os.environ.get("STOCHASTIC_POLICY", "1") == "1"
CAMERA_NOISE      = float(os.environ.get("CAMERA_NOISE", "3.0"))


def run_autonomous_loop(duration_seconds: int = 0):
    vla_model, device = load_vla_agent()
    processor   = vla_model.vision_processor

    # Nova arquitetura: sem ScreenCapturer, sem DirectInput
    capturer    = BotVisionCapture()          # frames do renderer voxel via HTTP
    logger      = TrajectoryLogger(dataset_path=LIVE_DATASET_PATH)
    evaluator   = PositionRewardEvaluator()   # recompensa via /delta do bot

    print("\n" + "=" * 68)
    print(" LOOP AUTONOMO — BOT MINEFLAYER VLA")
    print(" Visao: GET /frame  -> raycaster voxel (bot 1a pessoa, ~10ms)")
    print(" Acoes: POST /action → bot.setControlState()")
    print(" Reward: GET /delta  → deslocamento real no jogo")
    print(" Ctrl+C para parar.")
    print("=" * 68 + "\n", flush=True)

    t_start        = time.perf_counter()
    step_count     = 0
    positive_total = 0
    negative_total = 0

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Captura frame do bot via HTTP (sem captura de tela)
            img = capturer.capture()

            # 2. Pre-processa para o VLA
            pixel_inputs = processor(images=img, return_tensors="pt")["pixel_values"].to(device)
            state_vec    = torch.zeros(1, 16, device=device)

            # 3. Forward pass do modelo
            with torch.no_grad():
                actions = vla_model(pixel_values=pixel_inputs, state_vec=state_vec)

            b_probs = actions["buttons"][0].cpu().numpy()
            camera  = actions["camera"][0].cpu().numpy()

            # 4. Decodifica ações
            #
            # Amostragem estocástica em vez de limiar fixo em 0.5. Com o limiar,
            # a política é determinística: se o modelo aprender a não apertar
            # nada, ele NUNCA mais experimenta andar e não há como descobrir a
            # recompensa. Bernoulli(p) dá exploração proporcional à incerteza.
            names = ["W", "S", "A", "D", "SPACE", "LCLICK", "RCLICK", "SHIFT"]
            usable = [0, 1, 2, 3, 4, 7]   # W S A D SPACE SHIFT

            if random.random() < EXPLORE_EPS:
                # Passo totalmente aleatório: garante cobertura do espaço
                hold_keys = [names[i] for i in usable if random.random() < 0.25]
                dx = random.randint(-25, 25)
                dy = random.randint(-8, 8)
            elif STOCHASTIC_POLICY:
                hold_keys = [names[i] for i in usable if random.random() < float(b_probs[i])]
                dx = int(round(float(camera[0]) + random.gauss(0, CAMERA_NOISE)))
                dy = int(round(float(camera[1]) + random.gauss(0, CAMERA_NOISE)))
            else:
                hold_keys = [names[i] for i in usable if b_probs[i] > 0.5]
                dx = int(round(float(camera[0])))
                dy = int(round(float(camera[1])))

            action_dict = {
                "hold":        hold_keys,
                "mouse":       [dx, dy],
                "duration_ms": 100
            }

            # 5. Envia ação ao bot via HTTP POST (sem DirectInput!)
            send_action(action_dict)

            # 6. Calcula recompensa via posição real do bot (/delta)
            reward, reward_src = evaluator.compute_reward(img, action_dict)

            src_tag = "POS" if reward_src == "position" else "VIS"
            if reward > 0.0:
                positive_total += 1
                reward_tag = f"+{reward:.1f} [OK/{src_tag}]"
            elif reward < 0.0:
                negative_total += 1
                reward_tag = f"{reward:.1f} [BLOQ/{src_tag}]"
            else:
                reward_tag = f" 0.0 [---/{src_tag}]"

            # 7. Grava trajetória
            logger.log_trajectory(
                user_command="autonomous_locomotion",
                thought=(f"Step {step_count}: keys={hold_keys}, "
                         f"mouse=[{dx},{dy}], reward={reward:.1f}"),
                action=action_dict,
                reward=reward,
                curriculum_level="mineflayer_autonomous",
                image_b64=img_to_b64(img)
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            keys_str   = "+".join(hold_keys) if hold_keys else "PARADO"
            print(
                f"[{step_count:04d}] {elapsed_ms:5.1f}ms | "
                f"{keys_str:16s} | Mouse[{dx:3d},{dy:3d}] | "
                f"Reward: {reward_tag} | Bal: +{positive_total}/-{negative_total}",
                flush=True
            )

            step_count += 1

            # 8. Gatilho de self-imitation a cada N passos
            if step_count % RETRAIN_EVERY_STEPS == 0 and positive_total >= MIN_POSITIVE_SAMPLES:
                trigger_self_imitation()

            if duration_seconds > 0 and (time.perf_counter() - t_start) >= duration_seconds:
                print(f"\n[Loop] {duration_seconds}s concluidos. Encerrando.", flush=True)
                break

            time.sleep(0.03)  # ~30 fps máximo

    except KeyboardInterrupt:
        print("\n[Loop Interrompido]", flush=True)

    print(f"\n=== RESUMO: {step_count} passos | +{positive_total} OK | -{negative_total} BLOQ ===")
    print(f"    Trajetorias: {LIVE_DATASET_PATH}", flush=True)


if __name__ == "__main__":
    run_autonomous_loop(duration_seconds=0)
