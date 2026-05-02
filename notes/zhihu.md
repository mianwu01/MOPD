2026 年了，一个 LLM 的训练流程并不陌生——pre-train，SFT，RLHF/RLVR。但实际这是一个领域 LLM 的训练方案，比如 Coder/Match/文本专家，怎么整合成一个混合通用模型呢？

最近的 DeepSeek V4技术报告把Post-Train 讲流程讲的更细致了，它先是 pre-train 得到一个 Base-Model，然后先按领域（数学、代码、agent、指令跟随等）分别训练 10 几个专家模型，每个专家都走过 SFT + GRPO 的完整 RL 流程，在自己领域里练到极致。然后关键的一步来了：把这些专家”合成”一个统一模型时，不是让 student 去抄 teacher 的输出分布，而是让 student 自己先 rollout 生成回答，再让多个 teacher 在 student 自己写出来的轨迹上逐 token 给反馈。


这实际就是多专家的 on-policy 蒸馏（OPD）。

前代V3.2已经有”专家蒸馏”的雏形，他们训练了 5 个专家模型（Math、Coding、Reasoning、Agentic Coding、Agentic Search），用大量 RL 把每个练到强，然后把这些专家的知识蒸馏进最终的 V3.2 模型。

但是 V3.2 的蒸馏更像是”专家生成数据 + student 当 SFT 数据训”那种 off-policy 路线，”once the specialist models are prepared, they are used to produce the domain-specific data for the final checkpoint”，然后通过后续的 RL 来弥补差距 ，整合阶段还是用一个混合 GRPO 的 RL stage 在跑。

OPD 方法本身已经不新鲜了，Qwen3 的技术报告里就用它做 Strong-to-Weak Distillation，将旗舰级大模型的能力高效、低成本地迁移到小模型中。Qwen3 团队明确说”distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency”，约 1⁄10 的 GPU 算力做到 RL 同等效果，而且 pass@64 的探索空间还更广。GLM-5、MiMo 也都在 post-training pipeline 里采用了 OPD。V4 的不同在于把 OPD 推到极致——直接用它替代了原本 V3.2 那个 mixed RL 的整合阶段，作为多专家融合的主线。


为什么要使用这种方法呢？OPD 和 RL 一样，本质都是 reverse KL，相比下 pre-train&SFT&传统蒸馏 都是 forward KL。前者是学习者（student）采样出轨迹，由环境反馈（RLVR），Reward Model 反馈（RLHF），或 多个领域强专家反馈（多Expert OPD）。后者都是来自静态数据或 Teacher 的采样数据，一般监督学习的正向 KL 拉齐过程。

之前有讲过，reverse KL 是 On-Policy 的，它更不容易灾难性遗忘。这也是为什么它适合做多专家融合，Pre-train 之后获得 BaseModel，从它出发训练一个领域专家，它对其他领域的灾难性遗忘没有关系，它可以先毫无顾忌地进行领域特化。最后的 On-Policy Distillation 整合时极大地减少遗忘，减少各个领域专家之间的分布稀释知识抹除。


从“RL比SFT更不容易遗忘”到“反观推荐系统缺陷”
272 赞同 · 10 评论 文章
做个比喻。你已经会一口标准发音的英语，现在跟一位带方言口音的英语老师学语法。老师讲课时，”语法知识”和”方言口音”两个 mode 是混在一起的。


SFT 这种 forward KL 让你的分布去匹配老师的分布——老师在哪里有概率，你就得在哪里有概率。结果你不仅学到了语法，连口音也学了过来；更关键的是，你原本标准发音的那部分概率被挤掉了。这就是 mode-covering 的代价：为了覆盖老师所有 mode，你放弃了自己原本占据的 mode。 Reverse KL 反过来：你自己先开口（student rollout），老师只在你说出的话上反馈。你用标准发音说，老师就在标准发音的语境里纠正语法；他不会主动把你往方言那个 mode 拉，因为只要你不采样到那里，loss 就不要求你在那里分配概率。结果你只学到语法知识，原本的发音完整保留——这就是 mode-seeking，被丢掉的是老师的方言 mode，不是你自己的能力。



仔细思考下 LLM Post-Train 的多专家整合过程，特化的 ExpertModel 非常像学校教育里你面对的学科老师和领域教授，在学校里的学习不就是在做一个多 Expert 整合到你这个通用Student 模型里吗？

做完这个类比，我发现对于 LLM 训练的理解竟然能解释人类的学习过程，并且可以更清晰地看出其中的低效点，那些大多数人做错了的点。这个 forward → reverse 的框架立起来之后，再回头看那些 LLM 时代被鼓吹的”新学习法”，也能分辨出哪些是普适的、哪些只对天才适用。

先讲个故事。

我中学的同桌问我，为什么每道数学题都会？我确实每道都能做出来——但难题要花一个小时，简单题反而做得比他慢。他正好相反：简单题秒解，难题完全不会，得听我或老师讲。

我学数学是 RLVR 的过程：不太听课，直接硬刚题目。一开始挺痛苦，像 RL 初期 reward 在 0 附近转悠很久，开窍那一下来得很慢。但有信号之后，我自己做题（rollout）、答案对错就是 verified reward，会快速收敛到更高水平。我同桌则是纯 forward KL 的蒸馏——基本上只听别人讲，结果只会他见过的题型，稍微改改就懵了。

我的同桌和我是两个极端（RL 和 SFT，reverse 和 forward KL）。但那群学霸不一样——他们认真预习和听课（pre-train + SFT）拉齐基础概率分布，每一道难题先做足够的尝试（rollout），即使做不出来也是带着自己的预测和卡点去问老师或我（"数学偏科的我"是他们的 Math Expert）。所以学霸听讲解时拿到的是 reverse KL 反馈——在他们自己生成的轨迹上做 per-token 的纠正，而不是把答案当 SFT 数据从头抄一遍。


所以，从这个故事总结下，RL 可能上限更高，但消耗也巨大，同样的学习时间我可能也就能学明白一两门课。纯粹的 Pre-Train&SFT学习，forward KL 过程是无效的，上限低，不泛化，简单题没问题（其实简单题是常见题而已，充分见过了）。更高效的学霸模式，Pre-Train&SFT对齐基本分布，借助专家的 On-Policy 蒸馏整合获取更致密的反馈信号。

从这个故事回到 OPD，它并非一种绝对更好的蒸馏方式，它有它的适应前提。“Rethinking On-Policy Distillation of Large Language Models”指出，即使 teacher 更强，如果 student 和 teacher 的 top-k token 分布重叠率（overlap ratio）低，OPD 就失败。论文实测：用 Qwen3-4B-GRPO（一个 base+RL 的 teacher）蒸 Qwen3-1.7B-Base 比用 Qwen3-4B-Non-thinking 效果显著好，尽管两个 teacher benchmark 分数差不多——区别只在于前者和 student 共享 base 的”思考模式”，初始 overlap 高。并且早期的 thinking-pattern mismatch 造成的损失，后面训练补不回来。

为了让 OPD 跑起来，文章讲了两个方法，都需要 forward-KL 类的预备步骤：

Off-policy cold start：先用 teacher 生成的 rollout 做 SFT（这是标准 forward KL，teacher forcing），把 student 拉到 teacher 分布附近，然后再上 OPD。实测这一步既加快早期收敛，也提高最终性能上限——不只是热身效应，而是天花板都被抬高了。
Teacher-aligned prompts：用 teacher post-training 时见过的 prompt 来做 OPD。但要注意混合一些 OOD prompt，否则 student entropy 会塌缩，丧失探索能力。
类比人类学习的故事，就是你开始自己做题前，还是要先认真读书预习，认真听老师讲，拉齐你们的基本概率分布！


有些自媒体鼓吹 LLM 时代的学习方法，比如Gabriel Petersson。他高中辍学通过自学进入OpenAI，他的“递归式学习法”也因此备受关注——高强度提问（他建议每天问AI一百个问题）和深度思考。我只能说这是天才的学习方法，是人牛逼，不是方法牛逼。正常人都问不出一两个高质量问题。

LLM 的出现并没有改变最优的学习模式，它是能加速这套模式。比如人的 pre-train 过程，原来靠书本阅读和老师讲解，现在你可以用 LLM 做信息整合，就好比给自己做合成数据的 Pre-train。

这里也推荐下我的方案，推荐领域和 AI 领域我的 pre-train 数据整合：

一份推荐系统的：https://blog.recsys-frontier.com/category/推荐技术报告
一份AI 领域的：https://blog.recsys-frontier.com/category/AI技术报告
这些作为我泛读的重要来源，是首先建立起领域前沿的基础概率空间，这是我后续能深度思考的底料，是能对 LLM 问出100 个高质量问题的先决条件。否则你和前沿领域知识是有一道看不见的信息信息隔离墙的，不用说 LLM 的领域专家，你就是突然碰到了顶级的人类专家，你也不知道他在说什么！

做深度思考的时候，原本你需要找到一个领域专家做反馈，现在这个领域专家就在 LLM 聊天框里。但是如果你不是把大模型的提问过程当成一套先思考再和 LLM 确认反馈的 On-Policy Distillation 模式，而是让它快速地告诉你答案。那么只是在错误的学习模式上，加速的更猛了！这个错误，多数人上学的时候就是 Forward KL only，用上 LLM 它一样是这么错的。

LLM 没有改写学习的本质。它只是把 reverse KL 阶段需要的”高质量反馈”从稀缺资源变成了随手可得的资源。但前提是——你得先用 forward KL 把自己拉到能听懂反馈的地方。

reference
https://arxiv.org/pdf/2505.09388

https://arxiv.org/html/2604.13016v1

https://thinkingmachines.ai/blog/on-policy-distillation/

https://arxiv.org/pdf/2604.0062