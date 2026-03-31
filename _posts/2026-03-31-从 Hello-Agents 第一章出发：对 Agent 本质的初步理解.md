---
layout:     post
title:      "从Hello-Agents第一章出发：对Agent本质的初步理解"
subtitle:   "一切是个循环"
date:       2026-03-31
author:     vingo
catalog:    true
tags:
  - Agent
  - LLM
  - 学习笔记
---
# 从Hello-Agents第一章出发：对Agent本质的初步理解

## 前言

最近开始系统性学习 Agent，参考了 Datawhale 的 Hello-Agents 教程第一章。
这一章主要介绍了Agent的定义、历史以及一个简单的Agent最小用例。

教程整体偏概念入门，但在阅读过程中，还是有一些新的想法，这里做一次整理，包括一些重要的take away和自己的想法。

## Agent 的核心机制

### 概念描述

正如文字描述中所写：

> 智能体并非一次性完成任务，而是通过一个持续的循环与环境进行交互，这个核心机制被称为 **智能体循环 (Agent Loop)**

通常可以抽象为一个 `环境 →  感知 → 思考 → 行动 → 环境`的循环，LLM接入的正是其中思考的部分，大部分tool calling则是完成了感知以及行动的部分。

### 看下代码

#### 感知

具体在代码中，以教程中给出的代码demo为例。

`full_prompt`作为输入的一部分，也可以视为对环境最初步的感知。事实上，如果将所有输入视为感知，那所有的prompt都可视为LLM对外部的感知。只不过感知的来源在初始循环主要来自于用户，而在后续来源中来自于tool calling的返回结果。

```
full_prompt = "\n".join(prompt_history)
```

而实际上的输入还包括了提前定义好的`AGENT_SYSTEM_PROMPT`，里面会有一些类似于角色指定、可用工具注册、提示之类的指定，了解到的有些里面还会包括一些具体的例子。

比较特别的是这里解答了一个之前有的疑惑，即agent中在不同层面中使用“**硬编码**”与“**软编码**”的分界线是在哪里。这里的“硬编码”是指有固定格式的语句，“软编码”则是带有一定随机性、每次格式不同的输出结果。

首先LLM的输出结果一定属于是上述软编码的范畴，而各项tool更接近传统函数定义，需要固定的格式，那二者一定在某一个阶段需要进行一次转化，特别是将硬编码转成软编码。目前看来是由如下的system prompt进行了限制，说实话，比我想象的要简单粗暴很多。

会有的疑惑是感觉这样并不能绝对保证模型按照要求格式输出，还是会产生不可解析的语句，除了像demo里这样校验报错，还能有什么解决办法？

```
# 输出格式要求:

你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：

1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]
```

#### 思考

`llm_output`即为思考的结果，也是LLM接入整个agent最核心的部分，又或者说，在agent诞生之前，LLM的用法基本如此。

```
llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
```

#### 行动与感知

对于tool的调用即为行动，`observation`则是新的感知，即调用tool的返回结果，可以看到也会被加入到prompt作为下一轮新的输入给LLM。

```
observation = available_tools[tool_name](**kwargs)
observation_str = f"Observation: {observation}"
prompt_history.append(observation_str)
```

## 与workflow的区别

文中将二者的区别描述为`Workflow 是让 AI 按部就班地执行指令，而 Agent 则是赋予 AI 自由度去自主达成目标。`个人认为讲得并不是特别清楚，因为所谓AI的自由度也是在规则的基础上去建立的，如果给一个无限复杂的规则集，理论上workflow也可以达成类似的结果。

但二者事实上的不同，个人认为一在于量变产生质变，无限复杂的规则集只在理论上可行，当其膨胀到一定程度上，无论是维护还是使用都将变得不可能。另一点，也是我认为最重要的一点，是模型具有**推理**的功能，如果映射到规则集中，则接近于**根据已有规则产生新规则**的能力，这是传统workflow无论如何也做不到的一点，因此更接近二者的本质区别。

## 遗留的问题

* agent框架代码中正则表达式使用很多，可以单独开一篇学习一下
* 刚刚有提到的，在LLM没有根据想要的格式进行输出，如何进行纠偏
