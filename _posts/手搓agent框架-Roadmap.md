---
layout:     post   				        	# 使用的布局（不需要改）
title:      "手搓Agent框架：Roadmap" 				# 标题 
subtitle:   "搓，接着搓" 		# 副标题
date:       2026-04-24 					    # 时间
author:     Vingo 						    # 作者
catalog: true 							    # 是否归档
tags:								        # 标签
    - agent
    - frame
    - roadmap
---
## 0. 前言

Agent的初期知识补充的差不多了，基础的范式、结构、调用原理、框架也都了解过了，按理来说下一步是开始写一些agent的项目。也尝试了用AutoGen或者LangGraph实现一些很简单的idea，但是都会遇到这样那样的不够自由的地方，虽然都是开源库但是想在源码上修改还是十分费劲；以及感觉用下来更多是在调函数，并没有真正学到什么。

虽然说重复造轮子的人很蠢，但是只是拿轮子来用就以为自己会造轮子的人更蠢。因此在此之前，我想先尝试手搓一个框架，这样对于细节的实现会有更深入的理解。当然，之后的agent项目也会尝试直接在这个框架的基础上进行开发，过程中遇到问题就优化框架增加功能。所以在之后的一段时间框架和agent项目应该是会并行开发。

现在的agent项目给我的感觉更多是在比拼idea，看谁能在同一套工具下面玩出不同颜色的、长在不同地方的花。框架则是更偏工程实现，对整体架构的考验会更大。相信在开发框架的过程中也能发现更多的痛点，希望能有更多的idea。

先说点场面话，整体的目标是构建一个可扩展、可观测、易使用的Agent框架，拒绝过度抽象，保证灵活使用。先完成基础框架，有可使用的稳定架构，再逐步扩展高阶能力以及易用性方面的次要能力。

按照不同模块，写了个list后扔给ai完善了两轮，一轮补充缺失的点一轮补充具体描述，再经过一些个人喜好的调整和一些更深入的交流，得到了如下的一个list。不同优先级被定义为：
- `P0`：必须先做；不做会阻塞主流程
- `P1`：强烈建议；影响效果和可维护性
- `P2`：增强项；不阻塞主流程

## 1. 核心模块（Core Runtime）

### 1.1 LLM Gateway（模型网关）
- `P0`
  - 自动读取配置（model/api_key/base_url）
  - 流式/非流式统一返回结构（text/tool_calls/finish_reason/usage）
  - 统一message结构 (role/context)
- `P1`
  - 请求级成本统计与预算控制
  - provider fallback（主模型失败自动切备）
  - 统一异常分类（网络超时、限流、鉴权）
- `P2`
  - 响应缓存（可选）
  - 本地模型调用（兼容 OpenAI API 风格）
  

### 1.2 Tool System（工具系统）
- `P0`
  - 工具基类
  - 工具注册与调用
  - 工具 schema（参数校验）
  - 统一tool call 协议（`name + args + call_id`）
- `P1`
  - 固定目录自动加载（插件扫描 + 白名单）
  - 工具错误重试策略（可重试/不可重试）
- `P2`
  - 工具权限分级（read-only/network/shell）
  - 工具调用并发与限流

### 1.3 Orchestrator（运行时编排）
- `P0`
  - 标准 loop：`think -> decide -> tool -> observe -> answer`
  - 终止条件：max_steps / timeout / fail_count
  - 统一输出：final_answer + steps + tool_traces
- `P1`
  - 状态机化（idle/running/waiting_tool/finished/error）
  - 中断恢复（checkpoint）
- `P2`
  - 多轮会话恢复执行
  - 任务优先级调度

### 1.4 Agent 基础能力
- `P0`
  - ReAct Agent（最小可用）
  - Planner Agent（任务拆解）
- `P1`
  - Reviewer Agent（自检与纠错）
- `P2`
  - Router Agent（多 Agent 分发）
  - 多 Agent 协作模式（串行/并行）

---

## 2. 关键增强模块（Memory & Skills）

### 2.1 Memory（记忆系统）
- `P0`
  - 短期记忆（会话历史窗口）
  - 长期记忆存储接口（先抽象，后具体实现）
  - top-k 检索注入（禁止全量注入）
- `P1`
  - 压缩（summarize）与提取（facts/preferences）
  - 长短期管理策略（promote/forget）
- `P2`
  - 冲突事实解决与版本演进
  - 用户画像分层记忆

### 2.2 Skill System（技能系统）
- `P0`
  - skill 抽象（name/description/trigger/steps/version）
  - skill 执行器（可调用 tool、可产出中间状态）
- `P1`
  - 触发策略（规则 + LLM 判断）
  - 版本管理（兼容旧 skill）
- `P2`
  - skill 市场化加载（外部包）
  - skill 评测与排名

---

## 3. 支撑模块（Config / Observability / QA）

### 3.1 Config（配置系统）
- `P0`
  - 自动加载配置（env + file）
  - 运行时热切换（temperature/model/max_tokens）
- `P1`
  - 分环境配置（dev/test/prod）
  - 配置校验与启动自检
- `P2`
  - 远程配置中心（可选）

### 3.2 可调试性（Observability）
- `P0`
  - 结构化日志（session_id/turn_id/call_id）
  - trace 链路（LLM/tool/memory 全链路）
- `P1`
  - 回放系统（按 session 重放）
  - 调试模式（verbose + 中间状态输出）
- `P2`
  - 调试面板 UI（可视化步骤/耗时/错误）
  - 运行指标看板（吞吐、延迟、成功率）

### 3.3 评价与测试（Evaluation & Testing）
- `P0`
  - 单元测试：message/config/tool_executor/orchestrator
  - 集成测试：mock LLM + mock tool，完整 loop
- `P1`
  - 回归任务集（固定 benchmark）
  - 关键路径稳定性测试（超时/重试/失败恢复）
- `P2`
  - 自动评测流水线（CI nightly）
  - 多模型对比评测

---

## 4. 边缘模块（DX & Productization）

### 4.1 CLI 命令化
- `P0`
  - `chat` / `run` / `tools` 三个子命令
- `P1`
  - `replay` / `eval` / `config` 子命令
- `P2`
  - profile、session 管理、批处理执行

### 4.2 Demo
- `P0`
  - 最小闭环 demo（问答 + 一个工具）
- `P1`
  - memory demo、multi-agent demo
- `P2`
  - 端到端场景 demo（真实任务脚本）

### 4.3 文档
- `P0`
  - 快速开始、架构总览、扩展指南（tool/agent）
- `P1`
  - 运维与调试手册
- `P2`
  - 最佳实践与案例库

## 5. 开发计划（按里程碑）

### M1：打通核心闭环
目标：单 Agent 稳定可用
- 完成：
  - LLM Gateway `P0`
  - Tool System `P0`
  - Orchestrator `P0`
  - ReAct Agent `P0`
  - Config `P0`
  - Observability `P0`
  - 测试：UT + 最小集成测试 `P0`
- 验收：
  - 能稳定执行 `user -> llm -> tool -> llm -> answer`
  - 出错可定位（有 trace）

### M2：提升效果与可维护性
目标：记忆+规划能力可用，框架可调优
- 完成：
  - Memory `P0 + P1`
  - Planner/Reviewer `P1`
  - Tool 自动加载 `P1`
  - 回放 `P1`
  - 回归集 `P1`
- 验收：
  - 长对话不明显退化
  - 能回放并复现关键失败 case

### M3：能力扩展与产品化
目标：可扩展、可演示、可持续演进
- 完成：
  - Router Agent `P2（可提前到P1）`
  - Skill System `P0 + P1`
  - CLI 扩展 `P1`
  - Demo + 文档 `P0/P1`
  - 调试面板 `P2`
- 验收：
  - 具备多 Agent/skill 扩展能力
  - 新人可按文档在 30 分钟内跑通 demo

## 总结

**LET'S GOOOO!**
