## 基于 V-JEPA2 的 Human World Model（Gazelle 版）

![Human World Model 主图](assets/figure1.png)

本项目在 Meta 官方的 `vjepa2` 仓库基础上，引入 **Gazelle 场景 token** 作为条件信号，对 V‑JEPA2 的 predictor 进行增强，得到一个更贴近人类注意/意图的 **Human World Model**。整体流程分为两步：

- **阶段一**：在 Ego4D/Ego-Exo4D 视频上，用 Gazelle 提取 per-frame 场景 token，在冻结的 V‑JEPA2 encoder/target encoder 上训练一个 **Gazelle‑conditioned predictor**。
- **阶段二**：将该 Human World Model 作为冻结 backbone，在 `Ego4D KeyStep` 任务上只训练一个轻量的 **anticipation head**，评估其在动作预判上的增益。

关于原始 V‑JEPA2 的安装、基础训练与通用评测流程，请参考上游仓库文档（如 `vjepa2/README.md` 与 `vjepa2/evals/action_anticipation_frozen/EGO4D_KEYSTEP_TRAINING.md`）。本说明仅补充与 Human World Model 相关的关键步骤与命令。

---

### 环境与数据准备（简要）

- **依赖安装**：按 `vjepa2/README.md` 与 `V-jepa2-heatmap/README.md` 配置：
  - PyTorch + CUDA
  - `decord`、`wandb` 等基础依赖
- **Gazelle 安装与配置**：
  - clone Gazelle 仓库，并设置 `GAZELLE_REPO` 或在 yaml 里通过 `data.gazelle.python_path` 指向 Gazelle 根目录。
  - 准备 Gazelle checkpoint，例如  
    `gazelle/checkpoints/gazelle_dinov2_vitb14_inout.pt`。
- **数据**：
  - Ego4D / Ego-Exo4D 视频（供 Gazelle 预训练用），路径在 `V-jepa2-heatmap/configs/train/*.yaml` 中的 `data.datasets` 指定。
  - Ego4D KeyStep 标注和视频（用于下游评估），路径在 `vjepa2/configs/eval/vitl/ego4d_keystep.yaml` 中配置。

---

### 阶段一：用 Gazelle 预训练 Human World Model（训练 predictor）

这一阶段在 `V-jepa2-heatmap` 仓库中进行，通过 `app/main.py` + 训练配置 yaml 启动。示例配置为：

- `V-jepa2-heatmap/configs/train/vitg16/heatmap-224px-32f.yaml`  
 （其中 `data.condition_mode: gazelle` 且 `data.gazelle` 已配置好 Gazelle 参数）

#### 训练命令示例

```bash
cd /data3/lg2/human_wm/V-jepa2-heatmap

python app/main.py \
  --fname configs/train/vitg16/heatmap-224px-32f.yaml
```

该命令会：

- 加载预训练的 V‑JEPA2 encoder/target encoder（例如 `features/vitl.pt`）并冻结参数；
- 使用 `GazelleSceneTokenDataset` 从 Ego4D 视频中采样 clip，得到：
  - 给 encoder 的增强视频张量；
  - 给 Gazelle 的 raw uint8 帧；
- 通过 `GazelleSceneTokenExtractor` 在线提取每帧的 **scene tokens**，重采样到 ViT-L/16 的 patch grid，并线性映射为 latent gaze token；
- 在 token 空间上训练 Gazelle‑conditioned predictor，使预测的 future tokens 逼近 target encoder 输出的目标 tokens。

训练完成后，会在配置文件中 `folder` 字段指定的目录（例如：

```text
/data3/lg2/human_wm/V-jepa2-heatmap/logs/ego4d-heatmap-online
```

）下生成：

- `latest.pt`：包含 **Human World Model**（冻结 encoder/target encoder + 训练好的 Gazelle‑conditioned predictor），以及优化器/调度器状态；
- 以及按 `save_every_freq` 保存的中间 checkpoint（如 `e{epoch}.pt`）。

---

### 阶段二：在 Ego4D KeyStep 上评估 Human World Model

下游评估在原始 `vjepa2` 仓库中进行，通过 `evals/main.py` 与 `configs/eval/vitl/ego4d_keystep.yaml` 完成。

在该阶段，**Human World Model 完全冻结**，只在其输出的 future tokens 上训练一个轻量的 anticipation head（`AttentivePooler + Linear`）。

#### 评估命令示例

```bash
cd /data3/lg2/human_wm/vjepa2

python evals/main.py \
  --fname configs/eval/vitl/ego4d_keystep.yaml \
  --checkpoint /data3/lg2/human_wm/V-jepa2-heatmap/logs/ego4d-heatmap-online/latest.pt
```

该命令会：

- 在 `configs/eval/vitl/ego4d_keystep.yaml` 的基础上，使用 `--checkpoint` 指定的 `latest.pt` 覆盖 `model_kwargs.checkpoint`，将 Human World Model 作为 backbone；
- 通过 `ego4d_keystep.py`：
  - 使用 KeyStep JSON 标注构建 train/val 片段；
  - 按指定的 anticipation 时间采样 clip，解码为 `(video, action, anticipation_time)`；
  - 将 video clip 送入 **冻结的 Human World Model**（encoder + Gazelle‑conditioned predictor），得到对应未来帧的时空 token 表征；
  - 通过 `AttentivePooler` 将 token 序列聚合为 clip-level 表征，并经线性分类头预测 key-step 类别；
- 仅更新下游 head 的参数，评估 Human World Model 在 Ego4D KeyStep action anticipation 上的性能（Top‑1、mAP、mean class recall 等）。

---

### 关键代码位置（参考）

- **预训练阶段（Human World Model）**
  - `V-jepa2-heatmap/app/vjepa_droid/train.py`  
    Human World Model 训练主循环，`condition_mode="gazelle"` 分支、Gazelle 条件的 `compute_losses` 等。
  - `V-jepa2-heatmap/app/vjepa_droid/droid.py`  
    `init_data(..., condition_type="gazelle")` 与 `GazelleSceneTokenDataset`。
  - `V-jepa2-heatmap/app/vjepa_droid/gazelle_tokens.py`  
    `GazelleSceneTokenExtractor` 封装及在线场景 token 提取。
  - `V-jepa2-heatmap/src/models/gaze_ac_predictor.py`  
    Gazelle‑conditioned predictor：对齐 Gazelle scene tokens 到 ViT patch grid，并作为残差条件注入 predictor。

- **下游 Ego4D KeyStep 评估**
  - `vjepa2/evals/action_anticipation_frozen/ego4d_keystep.py`  
    KeyStep 数据切片、decode_videos_to_clips、以及使用 Human World Model 作为 backbone 的评估逻辑。
  - `vjepa2/configs/eval/vitl/ego4d_keystep.yaml`  
    Ego4D KeyStep 评估配置文件，包含数据路径、模型 checkpoint、训练超参等。

通过上述两个命令，即可从原始 V‑JEPA2 预训练权重出发，训练出带 Gazelle 条件的 Human World Model，并在 Ego4D KeyStep 动作预判任务上进行端到端评估。