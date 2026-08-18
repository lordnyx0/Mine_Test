# coding=utf-8
"""
Gravador e Exportador de Dataset Embodied Multimodal.

Salva trajetórias de sucesso do simulador (frames, prompts em texto, ações motoras,
estados e recompensas) em arquivos compactados para reutilização em:
  1. Pré-treino / Co-treino multimodal do Qwen3Loop v2.
  2. Fusão de conhecimento e destilação de políticas.
  3. Transferência de habilidades para outros motores 3D.
"""
import os
import time
import torch
import numpy as np


class GravadorDatasetEmbodied:
    """Acumula episódios de sucesso e exporta em lotes compactados."""
    def __init__(self, pasta_saida="dataset_embodied", shard_size=100):
        self.pasta_saida = pasta_saida
        self.shard_size = shard_size
        self.amostras = []
        self.total_salvo = 0
        os.makedirs(self.pasta_saida, exist_ok=True)

    def adicionar_episodio(self, frames_u8, prompts, acoes_idx, estados, recompensas, sucesso=True):
        """Adiciona um episódio completo ao buffer de gravação."""
        self.amostras.append({
            "frames": np.array(frames_u8, dtype=np.uint8),
            "prompts": list(prompts),
            "acoes": np.array(acoes_idx, dtype=np.int64),
            "estados": np.array(estados, dtype=np.float32),
            "recompensas": np.array(recompensas, dtype=np.float32),
            "sucesso": bool(sucesso),
            "timestamp": time.time()
        })

        if len(self.amostras) >= self.shard_size:
            self.despejar_shard()

    def despejar_shard(self):
        if not self.amostras:
            return
        shard_id = self.total_salvo // self.shard_size
        caminho = os.path.join(self.pasta_saida, f"embodied_shard_{shard_id:04d}.pt")
        torch.save(self.amostras, caminho)
        print(f"[Dataset] Shard salvo com {len(self.amostras)} episódios -> {caminho}")
        self.total_salvo += len(self.amostras)
        self.amostras = []

    def finalizar(self):
        if self.amostras:
            self.despejar_shard()
