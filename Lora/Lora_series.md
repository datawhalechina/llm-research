# LoRa系列

本笔记力求 让读者通过每个TLDR小节 学习到论文核心思想。读者若想更多的细节，可以阅读本笔记每篇论文的完整精读版，同时也辅以阅读原论文和相应源码，如此方可了解到自己想深挖的细节。

- [LoRa系列](#lora系列)
  - [前沿之Parameter-Efficient FineTuning简介](#前沿之parameter-efficient-finetuning简介)
- [Lora](#lora)
  - [浓缩版(**TLDR)**](#浓缩版tldr)
    - [1. 动机(本征维度)](#1-动机本征维度)
    - [2. 创新(对模型变化做低秩分解)](#2-创新对模型变化做低秩分解)
    - [3. 实验结果](#3-实验结果)
    - [4. 核心代码](#4-核心代码)
  - [完整精读版](#完整精读版)
- [QLora](#qlora)
- [AdaLora](#adalora)
- [LongLora](#longlora)
- [SLora](#slora)
- [oLora](#olora)


## 前沿之Parameter-Efficient FineTuning简介

二阶段训练范式（预训练+微调）在自然语言处理（NLP）领域达到了相当不错的表现。其核心思想是首先通过大规模的无监督语料库注入通用知识到模型中，然后针对不同的下游任务进行有监督微调。然而，随着模型规模的不断增大（例如，GPT-3 175B），对每个下游任务进行全参数微调不仅需要庞大的计算资源，而且在训练完成后需要保存各自的全量权重，这带来了高昂的计算和存储成本。

为了应对这一挑战，研究人员开始深入研究在大模型的二阶段范式中采用资源消耗较低的迁移学习方法。沿着这一思路，一些卓越的高效微调方法应运而生。这些方法仅微调模型的小部分参数（或额外的小量参数），却能够在性能上达到与全参数微调相媲美的水平。因此，在保持性能不降的同时，实现了训练和存储开销的降低。

# Lora

## 浓缩版(**TLDR)**

### 1. 动机(本征维度)

ACL2021年一篇论文指出，预训练语言模型的本征维度其实较小。(PS，一个模型的原始纬度是D=768，而本征维度是d=200，代表着，在一个d=200维投影子空间内，原始目标函数可以被优化到满意程度，不比原始维度差多少)。子空间投影公式如下所示，详细细节就不多阐述了。

ACL 2021年的一篇论文指出，预训练语言模型的本征维度实际上相对较小。例如Roberta模型的词向量空间维度为D=768，而其本征维度为d=200。(本征维度的定义：若在一个d=x维的投影子空间内，原始目标函数可以被优化到令人满意的程度，即与原始维度D维空间的损失相当，则这个x的下界就是模型的本征维度)。以下是那篇论文中子空间投影的公式，详细的理论细节这里就不展开了。


$$
\theta^{D}=\theta_{0}^{D}+\theta^{d} M \quad\quad M=H G \Pi H B
$$

$$
\theta_{i}^{D}=\theta_{0, i}^{D}+\lambda_{i} P\left(\theta^{d-m}\right)_{i}
$$



> Lora 4.1 原文摘录
> 
> 
> A neural network contains many dense layers which perform matrix multiplication. The weight matrices in these layers typically have full-rank. When adapting to a specific task, Aghajanyan et al. (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace. Inspired by this, we hypothesize the updates to the weights also have a low “intrinsic rank” during adaptation.
> 

### 2. 创新(对模型变化做低秩分解)

![Untitled](image/Untitled%202.png)

因此，LoRA假设在微调以适应下游任务时，模型权重的变化量同样具有较小的本征秩。因此，可以基于低秩分解，使用两个低维梯形矩阵去拟合高维的参数变化矩阵。

下面的公式1是原先模型中的矩阵乘法，公式2是模型挂载了Lora后的矩阵乘法。其中 $W,W_0,\Delta W\in R^{D*D},\ B\in R^{D*d}, A\in R^{d*D}$。模型训练过程中，$W_0$是冻结的，仅仅训练B和A。由此可见，这个矩阵的训练参数从D\*D减少到2d\*D。
所以使用Lora微调，可以大幅减少**梯度所需的显存，进而也减少了优化器状态所占显存**。

$$
h = Wx
$$

$$
h = (W_0+\Delta W)x = (W_0+BA)x
$$

> 训练过程中，GPU显存主要用于四部分：模型参数、优化器的状态、梯度、中间计算激活值(忽略一些临时区缓存和内存碎片）。
用Adam全量微调时，梯度和参数一样大，优化器状态是梯度的两倍(一阶动量和二阶动量)。
> 

### 3. 实验结果

大模型GPT-3和小模型GPT-2，Roberta，Debert等挂载Lora进行参数高效微调，都取得了和全量微调相当甚至更好的效果。

![Untitled](image/Untitled%203.png)

### 4. 核心代码

- B零初始化，A正态输出化，如此BA就是零矩阵，保证了不会带来初始偏移。
  以外，还有一个缩放系数$\frac{\alpha}{r}$
  
    $(W_0+\Delta W)x = (W_0+ \frac{\alpha}{r} BA)x$
  
- 这里给出项目中调用peft库 使用lora微调chatglm3的例子，详细代码请参考self-llm项目-[在甄嬛对话数据集上用lora微调chatglm](https://github.com/datawhalechina/self-llm/blob/master/ChatGLM/06-ChatGLM3-6B-Lora%E5%BE%AE%E8%B0%83.ipynb)。感谢self-llm给出了精彩有趣的例子
  
    ```python
    from peft import TaskType, get_peft_model, LoraConfig
    model = AutoModelForCausalLM.from_pretrained(GLM_PATH, torch_dtype=torch.half, trust_remote_code=True, low_cpu_mem_usage=True)
    # self-attention中的q,k,v三矩阵都挂载lora块
    config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules={"query_key_value"}, r=8, lora_alpha=32)  
  
    model = get_peft_model(model, config)
    model
    ```
    
    ```python
    PeftModelForCausalLM(
      (base_model): LoraModel(
        (model): ChatGLMForConditionalGeneration(
          (transformer): ChatGLMModel(
            (embedding): Embedding(
              (word_embeddings): Embedding(65024, 4096)
            )
            (rotary_pos_emb): RotaryEmbedding()
            (encoder): GLMTransformer(
              (layers): ModuleList(
                (0-27): 28 x GLMBlock(
                  (input_layernorm): RMSNorm()
                  (self_attention): SelfAttention(
                    (query_key_value): Linear(
                      in_features=4096, out_features=4608, bias=True
                      (lora_dropout): ModuleDict(
                        (default): Identity()
                      )
                      (lora_A): ModuleDict(
                        (default): Linear(in_features=4096, out_features=8, bias=False)
                      )
                      (lora_B): ModuleDict(
                        (default): Linear(in_features=8, out_features=4608, bias=False)
                      )
                      (lora_embedding_A): ParameterDict()
    ...
            (output_layer): Linear(in_features=4096, out_features=65024, bias=False)
          )
        )
      )
    )
    ```
    
    ```python
    # 用甄嬛对话数据微调chatglm3后的效果
    ipt = tokenizer("<|system|>\n现在你要扮演皇帝身边的女人--甄嬛\n<|user|>\n {}\n{}".format("你是谁？你认识温太医吗", "").strip() + "<|assistant|>\n", return_tensors="pt").to(model.device)
    print(tokenizer.decode(model.generate(**ipt, max_length=256, do_sample=True)[0], skip_special_tokens=True).strip('[gMASK]sop '))
    ```
    
    ```python
    <|system|>
    现在你要扮演皇帝身边的女人--甄嬛
    <|user|>
     你是谁？你认识温太医吗<|assistant|>
     我是甄嬛，认识温太医。他原是旗人，以前在太医院当过太医。
    ```
    

## 完整精读版

# QLora

# AdaLora

# LongLora

# SLora

# oLora