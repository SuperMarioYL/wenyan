<div align="right">[English](./README.en.md) | <strong>简体中文</strong></div>

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.svg">
  <img alt="wenyan 文言 hero" src="assets/hero-light.svg" width="880">
</picture>

<p><sub>面向 DeepSeek / Qwen / GLM 编码 Agent 的 per-tokenizer 中文 prompt-compression Skill —— 同一中文 prompt 在三个国产模型上 token 数差 &gt;5%，按模型测量并挑选最优压缩策略。</sub></p>

<p>
  <a href="./LICENSE"><img alt="license" src="https://img.shields.io/github/license/SuperMarioYL/wenyan?color=7c3aed"></a>
  <img alt="release" src="https://img.shields.io/github/v/release/SuperMarioYL/wenyan?color=14b8a6">
  <img alt="ci" src="https://img.shields.io/github/actions/workflow/status/SuperMarioYL/wenyan/ci.yml?branch=main&label=ci&logo=github">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Skill" src="https://img.shields.io/badge/Skill-caveman_wave-7c3aed">
  <img alt="Agent" src="https://img.shields.io/badge/Agent-国产模型-14b8a6">
</p>

</div>

**caveman 在英文上靠 function-word 截断省了 ~65% token；中文没有空格、也没有可截的 function-word —— wenyan 接的是 caveman 结构上做不到的表面。**

## <img src="https://api.iconify.design/tabler/topology-star-3.svg?color=%237c3aed&width=24" alt="" /> 架构

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/atlas-dark.svg">
  <img alt="wenyan architecture" src="assets/atlas-light.svg" width="880">
</picture>

单一 Python 进程 + 一份 markdown Skill：真实 `transformers` 分词器测量每个国产模型的中文 subword 切分，挑出该模型下净 token 最省的策略，再由 Claude Code Skill 按检测到的模型套用。

## 目录

- [为什么做](#为什么做)
- [安装与快速开始](#安装与快速开始)
- [用法](#用法)
- [Demo](#demo)
- [配置](#配置)
- [路线图](#路线图)
- [License](#license)

## 为什么做

caveman（[@JuliusBrussee](https://github.com/JuliusBrussee) 的 94K-star 英文 token 压缩 Skill）证明了这个原语——但它的机制（空格分词 + function-word 截断）在中文上无牙可咬。同一句中文 prompt 在 DeepSeek（32K BBPE，切得更碎）、Qwen、GLM（151K 词表，切得更密）上 token 数不同，一个全局改写省不出最优。wenyan 同属 Skill 原语，接的是国产模型分词器差异与 文言文 / 成语 / 助词 三种中文压缩原语——面向 DeepSeek/Qwen/GLM 编码 Agent，按模型测量并挑选最优策略。这是 caveman 的英文版结构上做不到的 owned layer，不是中文换皮。

## 安装与快速开始

```bash
git clone https://github.com/SuperMarioYL/wenyan && cd wenyan
uv pip install -e .
wenyan profile --suite          # 三模型 token 数 + 方差 % + 助词 net
```

<details><summary>样例输出</summary>

```
                        wenyan · per-tokenizer 中文 profile
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Model    ┃ Family              ┃ Baseline tokens ┃ 助词_strip tokens ┃ Net saved ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ deepseek │ DeepSeek BBPE (32K) │             303 │               293 │       +10 │
│ qwen     │ Qwen2 BBPE (151K)   │             250 │               241 │        +9 │
│ glm      │ GLM-4 BBPE (151K)   │             243 │               238 │        +5 │
└──────────┴─────────────────────┴─────────────────┴───────────────────┴───────────┘
╭───────────── m1 kill-gate · variance ─────────────╮
│ variance: mean 26.0%  ·  min 12.0%  (gate > 5%)   │
│ PASS ✅ per-tokenizer thesis holds — variance >5% │
╰───────────────────────────────────────────────────╯
╭──────────────── m1 kill-gate · net-token ────────────────╮
│ PASS ✅ 助词 prototype net-token positive on every model │
╰──────────────────────────────────────────────────────────╯
```

</details>

## <img src="https://api.iconify.design/tabler/terminal-2.svg?color=%237c3aed&width=24" alt="" /> 用法

```bash
# 用自带 10 条中文 coding-agent 语料做 per-tokenizer 方差画像
wenyan profile --suite

# 对自己的中文 prompt 做压缩，并打印每模型压缩前后 token
wenyan compress -p prompt.txt

# 净 token 回归（baseline − compressed − retry cost），m1 默认 retry cost=0
wenyan regress --suite --retry-cost 0

# m4：离线重跑 §8 net-savings 伪证（committed 真实分词器计数 fixture，无需联网）
wenyan harness

# 装进 Claude Code Skill，让编码 Agent 套用测得的赢家策略
cp wenyan/skill/SKILL.md ~/.claude/skills/wenyan/SKILL.md
```

编程式调用见 [`examples/profile.py`](examples/profile.py)；策略实现见 [`wenyan/strategies.py`](wenyan/strategies.py)；net-token 回归 harness 的可复现重跑与 fixture 重录见 [`docs/regress.md`](docs/regress.md)。

## <img src="https://api.iconify.design/tabler/photo.svg?color=%237c3aed&width=24" alt="" /> Demo

![demo](assets/demo.gif)

录制脚本 [`docs/demo.tape`](docs/demo.tape)（vhs 生成 `assets/demo.gif`），asciinema 源文件 [`demo/wenyan-demo.cast`](demo/wenyan-demo.cast)。`wenyan profile` 打印每模型 token 数、方差 % 与测得的压缩策略；v0.3.0 的录制展示 `wenyan regress` 在某个 pinned 分词器不可用时（离线 CI / air-gapped / HF 抖动）优雅降级为 `unavailable` 行并以 0 退出，而非崩出 traceback——§8 net-savings 伪证的重评路径不再被一个不可达分词器击垮。

### vs caveman

| 维度 | wenyan | [caveman](https://github.com/JuliusBrussee/caveman) |
|---|---|---|
| 目标语言 | 中文 | 英文 |
| 分词器维度 | per-model（DeepSeek/Qwen/GLM） | 单一 tokenizer |
| 压缩原语 | 助词 / 文言 / 成语 | function-word 截断 |
| 中文覆盖 | ✓ | ✗（无空格/function-word 可截） |
| 英文覆盖 | ✗（caveman owns） | ✓ |
| 部署形态 | Claude Code Skill | Claude Code Skill |

caveman 在英文压缩上明显更强（✓ vs wenyan 的 ✗）；wenyan 只做 caveman 结构上做不到的中文 + per-tokenizer 这一面。

## <img src="https://api.iconify.design/tabler/settings.svg?color=%237c3aed&width=24" alt="" /> 配置

模型注册表 [`wenyan/models.toml`](wenyan/models.toml) 把每个国产模型绑到真实 HF 分词器仓库：

| key | 类型 | 默认 | 含义 |
|---|---|---|---|
| `name` | string | — | 模型 id（deepseek / qwen / glm） |
| `repo` | string | — | HuggingFace 分词器仓库（只下分词器，不下权重） |
| `family` | string | `""` | 分词器族标注，用于表格展示 |

## <img src="https://api.iconify.design/tabler/route.svg?color=%237c3aed&width=24" alt="" /> 路线图

- [x] **m1 kill-gate**：per-tokenizer 中文 subword 画像（DeepSeek/Qwen/GLM，方差 >5% ✓）+ 助词 particle-strip 原型（per-model net-token 正 ✓）
- [x] 双语 README（zh 主 + `README.en.md`，顶部交叉链接）
- [ ] **m2 strategy picker**：完整 文言文 densify / 成语 sub 策略套件 + 读画像输出的 per-model 选择器 + SKILL.md per-strategy fragments
- [ ] **m3 regression + ship**：完整净 token 回归（task-success + retry tokens + net savings）+ PyPI / Gitee 发布
- [x] **m4 harness CI**：净 token 回归 harness 提交为可复现 artifact（committed 真实分词器计数 `RECORDED_COUNTS` + `wenyan harness` 一行重跑 + CI 门禁，见 [docs/regress.md](docs/regress.md)）

## <img src="https://api.iconify.design/tabler/scale.svg?color=%237c3aed&width=24" alt="" /> License

MIT — 见 [LICENSE](LICENSE)。欢迎在 [Issues](https://github.com/SuperMarioYL/wenyan/issues) 提反馈或开 PR。

## Share this

```
文言 wenyan —— 面向 DeepSeek/Qwen/GLM 编码 Agent 的 per-tokenizer 中文压缩 Skill。同一中文 prompt 在三个国产模型上 token 数差 >5%，按模型挑压缩策略。 https://github.com/SuperMarioYL/wenyan
```

<p align="center"><sub><a href="./LICENSE">MIT</a> © 2026 SuperMarioYL</sub></p>
