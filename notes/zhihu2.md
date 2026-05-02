从主流开源模型看 OPD 在后训练中的角色
品斯基
品斯基
微信公众号：全栈帕鲁；小红书: 5358325275
92 人赞同了该文章
随着 Qwen3、GLM-5 和 MiMo-V2-Flash 等一系列开源模型陆续发布，人们逐渐意识到：On-Policy Distillation（OPD）正悄然成为 Large Reasoning Models（LRM）训练框架的标配。Thinking Machines Lab 的技术博客就曾指出，OPD 所使用的反向 KL（Reverse KL）具备两项颇为迷人的性质——众数寻求（mode seeking） 与缓解暴露偏差（exposure bias）：

Two other useful properties of reverse KL are that it is “mode seeking” — it learns one specific behavior (the teacher’s) instead of spreading its distribution across several suboptimal options — and it reduces exposure bias.
正是这两项性质，直接决定了 OPD 在大语言模型后训练中的角色定位：前者展现出一种"胶水"式的特性，使得模型能力的整合几乎可以做到无损；后者则为蒸馏 LRM 提供了一条绝佳路径。下面，我们就从主流开源模型的实践出发，看看它们究竟是如何运用 OPD 的。

一、LRM蒸馏
在 LRM 流行之前，主流的模型蒸馏都是 off-policy 的，通常以 SFT 的形式进行：教师模型生成数据，让学生模型去拟合教师的数据分布。然而随着 LRM 的普及，这种做法的缺陷逐渐显现——off-policy 蒸馏出来的模型，往往只学到了推理的"形"，却没能真正继承教师模型的推理能力。

究其原因，在于学生模型训练时所见的轨迹，与它自己推理时实际走出的轨迹并不一致。举个例子：学生由于能力较弱，会在推理前期产生一些教师绝对不会出现的错误；这些错误在训练阶段学生从未接触过，一旦在推理中出现，偏差便会不断累积，最终导致结论错误。人们将这种现象归纳为"暴露偏差"（exposure bias）。

要缓解暴露偏差，就必须让学生在自己生成的轨迹上进行探索，即 on-policy；而相比完全放任学生自学的低效训练，用教师模型对每一步输出提供指导，则能让学生更快地掌握合理的推理模式，即 distillation。两者合起来，就是 On-policy Distillation。

在 Qwen3 的实践中，团队提出的 Strong-to-Weak Distillation 采取了两阶段策略：首先通过 Off-policy Distillation，让学生模型学习 /think 与 /no_think 的混合数据，使其具备基本的思考能力与模式切换能力；随后再通过 OPD，让学生自行探索该选择何种思考模式、又该如何具体地思考。

值得注意的是，清华大学近期的一篇《Rethinking OPD》论文通过实验指出：OPD 训练成功的关键因素之一，在于教师模型与学生模型具备一致的思考模式（thinking-pattern consistency）。作者建议在 OPD 训练之前，先用教师模型生成的轨迹对学生进行冷启动（即先做一轮 off-policy SFT，再进入 OPD 阶段）。这一结论印证了Qwen3的业界实践。


(流程图由GPT-image-v2生成)
二、对抗灾难性遗忘
随着Claude Code和OpenClaw等一众Vibe Coding和AgentOS工具的兴起，人们对模型的要求也越来越"全能"：既希望它具备良好的 Reasoning 能力以完成项目前期规划，又希望它拥有扎实的 Agentic 性能以支撑多轮工具调用，同时通用能力还要足够在线，才能胜任日常使用。由此，我们看到大模型的训练范式，已经从过去的标准三阶段（Pretraining → SFT → RLHF），演化为如今动辄七八个阶段的长链路训练，以便循序渐进地扩展上下文长度、叠加更多能力。

然而，训练阶段越多，模型也就越容易陷入"灾难性遗忘"（catastrophic forgetting） 的泥潭，用人话讲就是学了新的、忘了旧的。尽管业界已经通过数据回放、SFT 回炉以及 RL 中的 KL 正则等方式来缓解这一问题，但这些方法大多依赖前向 KL（Forward KL），而前向 KL 的众数覆盖（mode covering） 特性，会驱使学生模型尽可能覆盖教师模型的所有可能分布，最终训出的结果往往是"样样通、样样松"。陈丹琦团队的论文《Retaining by Doing》对Forward KL 与Reverse KL 在灾难性遗忘中的作用机理做了深入分析，这里就不再赘述。


（分布示例图片来源：https://blog.evjang.com/2016/08/variational-bayes.html）
GLM-5 给出的答卷，是 On-Policy Cross-Stage Distillation（OPCSD）。我们直接来看它的具体做法：学生模型选用最后一轮经过General RL 的模型，而教师模型则选择经过 Reasoning RL 和 General RL 的两个模型。多教师 OPD 的训练通过提示词路由来实现——如果提示词采样自 Reasoning RL 的训练语料，便以 Reasoning RL 模型作为教师；如果提示词来自 General RL，便以 General RL 后的模型作为教师。


可惜 GLM-5 的技术报告并未对这一设计展开深入讨论。笔者猜测，同时引入 Reasoning 与 General 双教师，是为了避免仅使用 Reasoning 单教师时模型坍缩回纯 Reasoning能力；类比数据回放，或许可以称它为模型回放。而之所以没有把 Agentic RL 之后的模型也纳入教师行列，大概是因为 Agentic 轨迹多为 long-horizon 任务，与 Reasoning、General 的分布差异较大，贸然加入反而会让训练变得不稳定。

三、模型能力整合
在 GLM-5 的技术报告中，我们可以清晰地看到：OPD 已不仅仅是蒸馏小模型的工具，更成为提升模型能力的必要手段。这里有两个关键点：其一，教师与学生模型同源，仅在后训练 RL 阶段存在细微差别；其二，通过提示词路由实现多教师 OPD。前者的理论与实验依据，可以参考前述《Rethinking OPD》的论文；而后者的提示词路由，则确保了能力整合过程中各路信号互不污染。

具体而言，对于从某个 Domain 采样出来的提示词，路由机制保证了这条样本只学习该 Domain 的教师；而 Reverse KL 的 mode seeking 特性，则保证了学生轨迹会紧紧锁定到该教师的某一种思考模式上。

反之，如果使用 mode covering 的 Forward KL，学生在 Reasoning prompt 上会试图"顺带顾及一点" General 的风格（以防 General 那边的分布对该 prompt 也有非零概率），最终导致两种能力相互污染。而 mode seeking 则让学生能够在每个 prompt 域内干净地锁定对应教师的模式，各项能力之间互不稀释。

事实上，早在 MiMo-V2-Flash 的技术报告中，多教师 OPD（Multi-Teacher On-Policy Distillation，MOPD）就已经被提出。不过相比 GLM-5，MiMo-V2-Flash 的做法要复杂、激进得多，因此我们把MiMo-V2-Flash放在GLM-5后面介绍。


首先，在教师模型的选择上，MOPD 的教师数量远多于 OPCSD，且对 Domain 的划分也更加精细。MOPD 先按 agentic 与 non-agentic 将能力一分为二：agentic 进一步细分为 search、coding、general tool use；non-agentic 则细分为 math、general reasoning、safety alignment。不过，教师与学生模型依旧保持同源同参数量，教师之间的差异仅体现在后训练 RL 上。

其次，MOPD 还支持一种左脚踩右脚上天的 co-evolution cycle，蒸馏后的学生模型可以重新进入 Domain-Specialized Training，迭代出性能更强的教师模型，然后再进行下一轮蒸馏。可惜小米的技术报告并未透露这种循环能做几轮、具体带来多少收益。

最后，在优化目标上，OPCSD 与 MOPD 也存在细微差别。OPCSD 沿用了 GRPO 的框架，仅将优势函数替换为 Reverse KL，同时将 GRPO 的 group size 设为 1 以提升 data throughput（此时 GRPO 实际上退化为逐 token 的 Reverse KL 蒸馏）。


而在 MOPD 中，除了教师模型提供的监督信号，优势函数还额外加入了结果奖励折算的优势项：


从直观上看，把 MOPD 训好，要比 OPCSD 难上不少，可惜小米的技术报告并未提供更多细节。不过换个角度说，大模型后训练中的多教师 OPD 才刚刚引起人们的关注，其中值得探讨的问题还有很多，例如：某个教师发生模式坍塌会怎样？Domain 该如何划分？不同 Domain 的提示词若相近甚至重叠，训练又会走向何方？多教师蒸馏能否混合 Forward KL 与 Reverse KL，以兼收两者之长？

四、多模态能力整合/修复
上述应用都聚焦于纯文本大模型，但实际上，在多模态场景下，OPD 同样大有用武之地。

在 Qwen3.5-Omni 的技术报告中，作者注意到一个现象：对于相同的 prompt，语音输入的回复质量与文本输入的回复质量之间仍存在巨大差距，这一现象在语音对话中尤为明显。报告虽未深入分析原因，但给出了相应的解决方案。

具体而言，作者在 Thinker 模型的后训练阶段引入 OPD，目标是让同一个模型在音频输入条件下，学习它自己在文本输入条件下的行为。做法是：准备 audio-text 配对的 query提示词，以 text 输入的轨迹作为教师，以 audio 输入的轨迹作为学生，用 OPD 的方式完成训练。

有趣的是，这里的教师和学生其实是同一个模型，唯一的区别仅在于输入模态。这种做法为多模态 OPD 打开了全新的思路。毕竟，很多时候相同的信息会体现在不同的模态之上，而大模型对不同模态输入的表现，往往并非天然对齐。


(流程图由GPT-image-v2生成)
五、总结
回顾全文，OPD 在后训练中的角色，其实经历了一条清晰的跃迁路径：从最初被视为"LRM 蒸馏的替代方案"，到被 GLM-5 用作"对抗灾难性遗忘的黏合剂"，以及 MiMo-V2-Flash 的"多教师能力整合的主干框架"，最后在 Qwen3.5-Omni 手中延伸为"跨模态对齐的桥梁"。OPD 从配角走向主角，背后的技术支点其实只有一个——Reverse KL 所提供的 mode seeking 特性。

换言之，当模型的训练阶段越来越多、能力边界越来越宽、输入模态越来越杂，后训练真正稀缺的已经不是数据或算力，而是一种能够让新能力长进来、又不把旧能力挤出去的机制，OPD恰好表现出这种潜力。

当然，OPD的故事才刚刚展开：多教师该如何划分与路由、Forward KL 与 Reverse KL 能否协同、co-evolution cycle 的收益边界在哪里、OPD能否刷新多模态对齐范式——这些问题都还有待解答。可以预见的是，未来一段时间内，围绕 OPD 的工程与理论探索，仍将是最值得关注的主线之一。
