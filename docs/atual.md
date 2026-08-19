# Estado atual — 2026-08-19 (Progresso Registrado Pré-Queda de Energia)

Ponto de retomada imediata. O estado dos checkpoints e métricas foram registrados com segurança.

---

## 📊 Progresso Registrado do Treinamento (Fase 5.5)

- **Iteração Atual Alcançada:** **66 / 100** (66% concluído)
- **Estágio Curricular:** **ETAPA A** (Pilar Único)
- **Checkpoints Salvos em Disco (`checkpoints_vla/`):**
  - `vla_fase5_ppo_bc.pt` (Snapshot mais recente — Iteração 66, timestamp 08:40)
  - `vla_fase5_ppo_bc_melhor.pt` (Melhor modelo histórico — Iteração 12 com chegada $12.5\%$, $\text{Rec}=-2.05$)
  - `vla_fase5_ppo_bc_it60.pt` (Snapshot de segurança — Iteração 60)
  - `vla_fase5_ppo_bc_it40.pt` (Snapshot de segurança — Iteração 40)
  - `vla_fase5_ppo_bc_it20.pt` (Snapshot de segurança — Iteração 20)

### Métricas da Iteração 66:
- **Value Loss (Critic MSE):** $0.0081$ (Critic altamente calibrado, $<0.01$)
- **PPO Loss:** $+0.0085$ | **Clip Fraction:** $17.0\%$ (Estável na faixa $10-20\%$)
- **BC Loss (Ancoragem Fatorada):** $0.5033$ (Decaimento curricular $\lambda_{bc} = 0.33$)
- **Entropia Total:** $0.66$ (Modo: $0.10$, Yaw: $0.56$)
- **Adv[HighH] vs Adv[LowH]:** $\text{mean}=-0.177$ vs $-0.064$ ($|adv|=0.182$ vs $0.175$)

---

## ⚡ Como Retomar Imediatamente Após o Retorno da Energia

Assim que o computador ligar:

1. **Subir o Simulador Offline (Porta 3002):**
   ```bash
   node mineflayer_server/servidor_offline.js
   ```

2. **Retomar o Treinamento do Checkpoint Mais Recente (`vla_fase5_ppo_bc.pt`):**
   ```bash
   python -u fase5/treinar_ppo_bc_hibrido.py \
       --dataset fase5/dados/dataset_wasd_tatico_36_v2.pt \
       --base checkpoints_vla/vla_fase5_ppo_bc.pt \
       --saida checkpoints_vla/vla_fase5_ppo_bc.pt \
       --iteracoes 100 --passos 100 --lr 3e-5 \
       --lambda-shaping 0.10 \
       --curriculo-estagio auto \
       --criterio-a 0.35 --criterio-b 0.20 \
       --salvar-cada 20
   ```

3. **(Opcional) Rodar Avaliação TopView 2D do Checkpoint Atual:**
   ```bash
   python fase5/avaliar_fase5_topview.py \
       --ckpt checkpoints_vla/vla_fase5_ppo_bc.pt \
       --lotes 3 --passos 100 --raio 1.5
   ```
