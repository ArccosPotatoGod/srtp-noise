# MicFrozen Simulation

高保真声学仿真平台，复现论文 **"Cancelling Speech Signals for Speech Privacy Protection against Microphone Eavesdropping"** (Gao et al., ACM MobiCom 2023) 中的 MicFrozen 超声麦克风干扰系统。

## 原理概述

MicFrozen 通过两种机制保护语音隐私：
1. **语音信号抵消**：发射反相超声信号在窃听麦克风处削弱原始语音。
2. **相干噪声耦合**：注入与语音信号耦合的噪声，使去噪技术（BSS、滤波、波束成形）难以分离。

仿真基于 Pyroomacoustics 提供的房间脉冲响应（RIR），模拟真实声学环境中的混响、多径效应及超声衰减。

## 项目结构

```
├── configs/                # YAML 配置文件
│   ├── base.yaml           #   默认仿真参数
│   └── grid.yaml           #   批量实验参数网格
├── src/                    # 核心模块
│   ├── config.py           #   ScenarioConfig 配置加载
│   ├── speaker.py          #   SpeakerModule 声源
│   ├── channel.py          #   ChannelModule 房间信道
│   ├── jammer.py           #   JammerModule 干扰者
│   ├── spy_mic.py          #   SpyMicrophoneModule 窃听麦克风
│   ├── attacker.py         #   AttackerModule 攻击者
│   ├── evaluator.py        #   Evaluator 评估器
│   ├── modulation.py       #   超声调制/解调辅助
│   └── report.py           #   结果报告生成
├── strategies/             # 可插拔算法策略
│   ├── canceling.py        #   抵消策略
│   ├── coherent.py         #   相干噪声 & 基线噪声
│   ├── denoiser.py         #   去噪器
│   ├── asr.py              #   语音识别
│   └── nonlinearity.py     #   非线性模型
├── tests/                  # 测试
├── runner.py               # 批量实验运行器
├── run_single.py           # 单次运行演示
├── run_grid.py             # 批量实验入口
└── requirements.txt
```

## 快速开始

### 环境准备

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 下载语音数据

```bash
python scripts/download_data.py
```

### 单次仿真

```bash
python run_single.py                    # 使用默认配置
python run_single.py --config my.yaml   # 使用自定义配置
```

### 批量实验

```bash
python run_grid.py                      # 使用 configs/grid.yaml
python run_grid.py --grid my_grid.yaml  # 使用自定义网格
```

输出结果保存在 `results/` 目录下，包括 CSV 数据表和对比图表。

## 核心指标

| 指标 | 说明 |
|------|------|
| SNR | 信噪比，反映语音信号被干扰的程度（越低越好） |
| CWER | Cooperative Word Error Rate，未被正确识别的词比例（越高越好） |

## 参考

- Gao et al., "Cancelling Speech Signals for Speech Privacy Protection against Microphone Eavesdropping", ACM MobiCom 2023.
- 详细设计文档：[docs/spec.md](docs/spec.md)
