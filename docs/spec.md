# MicFrozen仿真复现与改进：一周科研训练方案（Agent执行文档）

## 1. 项目目标
基于论文《Cancelling Speech Signals for Speech Privacy Protection against Microphone Eavesdropping》(Gao et al., MobiCom 2023)，在纯软件环境中复现其核心机制，并完成一个轻量级改进。最终产出物为一套仿真脚本、对比实验图表及汇报材料，可在一周内完成。

## 2. 环境准备
- **编程语言**：Python 3.9+
- **必需库**：
  - `numpy`, `scipy` (信号处理、滤波、ICA)
  - `librosa` (音频加载与特征提取，含MFCC等)
  - `soundfile` 或 `scipy.io.wavfile` (读写音频)
  - `matplotlib`, `seaborn` (可视化)
  - `scikit-learn` (FastICA实现)
  - `jiwer` (计算词错率WER，可选)
- **硬件需求**：普通PC即可，无需GPU。

## 3. 数据准备
- 从LibriSpeech `test-clean` 中选取5-10条短语音（每条10-15秒），作为受保护的私密语音信号 `s(t)`。
- 采样率统一为 16000 Hz（可听声）和 192000 Hz（用于超声波仿真，但非必需，也可在基带仿真）。
- 所有信号保存为 `float32` 的numpy数组。

## 4. 核心仿真模块设计

### 4.1 声学传播与衰减模型
**功能**：模拟声音在空气中的距离衰减和超声波/可听声的差异衰减。

**参数设定**：
- 声速 `v = 340 m/s`
- 声源位置 `src_pos = (0, 0)`，MicFrozen设备位置 `jammer_pos = (0.2, 0)` m，参考麦克风紧贴声源（`ref_mic_pos = (0.1, 0)` m）。
- 可听声随距离平方衰减：`atten_L(dist) = 1 / (dist + 0.1)`
- 超声波衰减更快，额外乘衰减系数 `atten_ultra(dist) = atten_L(dist) * exp(-alpha * dist)`，`alpha = 2.0` 以体现快速衰减。

**实现函数**：
```python
def propagate_signal(signal, src_pos, mic_pos, fs, v=340, is_ultrasound=False):
    dist = np.linalg.norm(np.array(mic_pos) - np.array(src_pos))
    delay_samples = int(dist / v * fs)
    # 衰减计算
    atten = 1 / (dist + 0.1)
    if is_ultrasound:
        atten *= np.exp(-2.0 * dist)
    # 延迟并衰减
    propagated = np.zeros_like(signal)
    if delay_samples < len(signal):
        propagated[delay_samples:] = signal[:len(signal)-delay_samples] * atten
    return propagated
```

### 4.2 超声波非线性解调
**功能**：模拟麦克风非线性产生的可听声干扰项。

根据论文公式(1)，输出解调后的噪声为 `N(t) = A2 * (n(t) + 0.5*n(t)^2)`。此处取 `A2 = 1`。
我们将基带干扰信号 `n(t)` 调制到载波频率 `fc = 40kHz`，但仿真中不需要高频载波，可直接生成解调后的可听带内干扰。

**实现**：
```python
def nonlinear_demod(noise_baseband):
    return noise_baseband + 0.5 * noise_baseband**2
```

### 4.3 逆通道自干扰抵消（Speech Signal Cancellation）
**原理**：参考麦克风 `M_r` 获取 `s(t - d/v)` 的延迟估计，然后生成反相版本 `-s(t - 2d/v)` 从jammer发出。在窃听麦克风处，直接路径信号与反相抵消信号叠加。

**仿真步骤**：
1. 声源到参考麦克风 `M_r` 的距离 `d_ref = 0.1 m`，得到 `s_ref = s(t - d_ref/v)`。
2. MicFrozen设备到窃听Mic的距离 `d_jam_to_spy` 动态设置（如1m, 3m, 5m）。
3. 反相抵消信号：`anti_signal = - s_ref`，再经过jammer发出，在窃听处产生 `s_cancel = propagate_signal(anti_signal, jammer_pos, spy_pos, fs)`。
4. 窃听麦克风同时接收声源直达 `s_direct = propagate_signal(s, src_pos, spy_pos, fs)` 和抵消信号 `s_cancel`，以及后续噪声。
5. 残余语音为 `s_res = s_direct + s_cancel`（注意求和，因为anti已反相）。

**注意**：相位差导致的不完全抵消体现在传播延迟的不同，尤其是在非共线位置（二维场景）。通过设置不同 `spy_pos` 角度可引入相位差。

**函数**：
```python
def apply_cancellation(s, src_pos, jammer_pos, spy_pos, fs):
    # 计算参考信号 (近似为直达声延迟)
    d_ref = np.linalg.norm(np.array(jammer_pos) - np.array(src_pos))  # 作为参考麦距离
    s_ref = propagate_signal(s, src_pos, jammer_pos, fs)  # 简化：参考麦在jammer位置
    # 生成反相抵消信号
    anti_signal = -s_ref
    # 经jammer发出到spy
    s_cancel = propagate_signal(anti_signal, jammer_pos, spy_pos, fs)
    # 声源直达spy
    s_direct = propagate_signal(s, src_pos, spy_pos, fs)
    # 残余
    residual = s_direct + s_cancel
    return residual, s_direct
```

### 4.4 相干噪声生成（Coherent Noise Coupling）
**目的**：生成与语音紧密耦合、难以通过盲源分离去除的噪声。

**实现公式**(17)的简化版：
```
n_co(t) = n_T(t) ∗ A · σ( s(t), n_M(t), n_F(t)·s(t) )
```
其中 `σ` 为 sigmoid 激活。为降低计算量，省略卷积部分，直接用非线性混合：
```python
def generate_coherent_noise(s, fs=16000):
    T = len(s)
    # 随机噪声 n_M
    n_M = np.random.randn(T) * 0.5
    # 频域乘积噪声 n_F * s (时域相乘)
    n_F = np.random.randn(T) * 0.3
    freq_component = n_F * s
    # 混合矩阵 A (1x3)
    A = np.random.randn(3) * 0.8 + 0.2
    # 非线性激活
    mixed = A[0]*s + A[1]*n_M + A[2]*freq_component
    noise_baseband = 1 - np.exp(-mixed) / (1 + np.exp(-mixed))  # sigmoid
    # 经非线性解调后的可听噪声
    N = nonlinear_demod(noise_baseband)
    return N
```

### 4.5 高斯噪声基线生成
```python
def generate_gaussian_noise(T, fs=16000, bandwidth=4000):
    # 带限高斯噪声
    from scipy.signal import butter, lfilter
    b, a = butter(4, bandwidth/(fs/2), btype='low')
    noise = np.random.randn(T)
    gaussian_noise = lfilter(b, a, noise)
    return gaussian_noise * 0.5  # 功率调节
```

## 5. 对手去噪攻击仿真
实现三种方法，输入含噪语音，输出增强后语音。

### 5.1 带阻/带通滤波
基于已知干扰频带，使用butterworth滤波器。
```python
def apply_bandstop(y, fs, f_low=300, f_high=3500):
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs
    low = f_low / nyq
    high = f_high / nyq
    b, a = butter(4, [low, high], btype='bandstop')
    return filtfilt(b, a, y)
```

### 5.2 盲源分离 (FastICA)
需双通道输入，仿真另一个麦克风（距离不同或放置位置不同）。生成双通道混合信号，然后解混。
```python
from sklearn.decomposition import FastICA
def ica_denoise(mix1, mix2):
    X = np.c_[mix1, mix2].T
    # 假设成分数为2
    ica = FastICA(n_components=2, max_iter=1000, random_state=0)
    S_ = ica.fit_transform(X)  # 估计的源信号
    # 选择与原始s最相关的一路作为恢复的语音
    # 为简化，直接返回能量较大的一路作为干扰残留，另一路可能是语音
    # 此处自动选取
    return S_[0]  # 需要根据实际实现选择
```

### 5.3 延时求和波束成形
模拟4个麦克风线性阵列，间距0.05m，目标方向为声源方向。实现对其他方向干扰的抑制。
```python
def delay_and_sum(multi_channel_signals, delays):
    # multi_channel_signals: (n_mics, samples)
    aligned = np.zeros_like(multi_channel_signals)
    for i, delay in enumerate(delays):
        aligned[i] = np.roll(multi_channel_signals[i], delay)
    return np.mean(aligned, axis=0)
```

## 6. 改进方案：自适应权重耦合噪声
**改进点**：根据估计的窃听距离动态调整相干噪声中语音分量的权重 `α`，使远距离时减少耦合以增大噪声压制，近距离时增加耦合以抵抗去噪。

**实现**：
```python
def adaptive_coherent_noise(s, estimated_distance, fs=16000):
    # 估计距离可以从信号强度推算，仿真中直接传入真实距离
    if estimated_distance < 2.0:
        alpha = 1.2  # 高耦合
    elif estimated_distance < 4.0:
        alpha = 0.8
    else:
        alpha = 0.4
    # 修改混合矩阵中语音的系数
    A = np.array([alpha * 1.0, 0.8, 0.5])  # A[0] 控制语音分量
    # 其余同原相干噪声生成
    ...
```

## 7. 实验设计与执行流程

### 7.1 单样本对比实验
对一条语音，绘制以下波形图/语谱图：
- 原始语音 `s`
- 只有高斯噪声干扰（传统UMJ）
- 加MicFrozen抵消 + 相干噪声
- 经ICA去噪后的两种方法的残余

### 7.2 定量指标计算
- **SNR** (信噪比)：`10*log10(||s||^2 / ||residual - s||^2)`
- **WER** (词错率)：使用开源的ASR引擎（如Whisper tiny模型）或直接调用Google STT API计算识别率，若无网络可用 `WER` 近似为基于MFCC距离的指标。作为备选，使用 `CER` (字符错误率)并手动标注几条。

### 7.3 批量实验矩阵
遍历以下条件并记录指标：
- **干扰方法**：高斯噪声、相干噪声(固定α)、自适应相干噪声
- **窃听距离**：1m, 2m, 3m, 4m, 5m
- **角度偏差** (2D场景)：0°, 15°, 30°, 45°
- **去噪攻击**：无、滤波、ICA、波束成形

### 7.4 结果图表
- 折线图：SNR vs 距离，不同干扰方法对比（含去噪后）
- 柱状图：WER对比
- 热力图：2D区域内CWER分布（角度-距离）

## 8. 最终交付物
1. **仿真代码**：一个主程序 `demo_main.py` 及多个模块文件，结构清晰。
2. **结果汇总**：生成 `results.csv` 及所有图表。
3. **汇报材料**：
   - 2-3页PPT或一张学术海报，包含问题背景、核心思路(抵消+耦合)、改进亮点、关键实验结果。
   - 重点突出“自适应权重”带来的去噪鲁棒性提升。

## 9. 一周时间规划（Agent可按此顺序调度）
- **Day 1-2**：搭建核心仿真模块(4.1-4.5)，跑通单条语音的干扰与抵消。
- **Day 3**：实现全部去噪攻击(5.1-5.3)并完成基线对比实验。
- **Day 4**：实现自适应改进(6)，进行定量批量实验，收集数据。
- **Day 5**：绘制图表，准备汇报PPT/海报文案。
- **Day 6**：整体调试，美化可视化，撰写README。
- **Day 7**：模拟汇报，最后修改。

---

执行本方案时，请严格遵循函数接口定义，确保每个模块可独立测试，最终在 `main` 中串联所有流程。优先保证可用性，再追求高完成度。