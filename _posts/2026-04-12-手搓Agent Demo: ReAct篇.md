---
layout:     post   				        	# 使用的布局（不需要改）
title:      "手搓Agent Demo：ReAct篇" 				# 标题 
subtitle:   "" 		# 副标题
date:       2026-04-12 					    # 时间
author:     Vingo 						    # 作者
catalog: true 							    # 是否归档
tags:								        # 标签
    - Agent
    - ReAct
    - coding
---

## 前言

了解完几种基本的agent范式之后，尝试实现一些简单的demo。这里主要用作过程中遇到问题的记录以及一些思路。

此篇种的代码实现还是主要依靠古法编程，因为想初期还是仔细了解一些实现的细节，后续的Plan-and-Solve以及Reflection会更多的脱离教程，使用agent进行编程。

## 通用LLM类

首先实现一个自己的LLM类，用以接入之后所有agent。这里借用教程中的实现如下：

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()

class MyLLM:
    """
    用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在ß文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        llmClient = MyLLM()
        
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        print(e)
```

同时，需要在项目根目录或其他指定位置，创建一个`.env`文件，其中应包含如下内容：

```
LLM_MODEL_ID=glm-5.1
LLM_API_KEY=api_key
LLM_BASE_URL=https://api.openbitfun.com/v1
```

> 这里有一个小点注意，`LLM_BASE_URL`在写的时候，如果是用OpenAI协议，只需要写到服务跟路径，即如上的部分

运行如上代码，前半部分会正常生成，但后续会发生报错：
```
❌ 调用LLM API时发生错误: list index out of range
```
这是因为使用了`stream=True`的设置，这样模型会将生成的结果分块一部分一部分的返回，否则是在所有回复生成后统一返回。一个返回块大概长成这个样子：

```json
{
  "id": "...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "完整回答文本"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {...}
}
```

但有时分块的返回中并不都带有内容，比如起始/结束包。这时候去调`content = chunk.choices[0].delta.content`就会发生刚刚的报错。因而把单纯的获取content改为有校验的获取即可：

```python
# 某些兼容服务会返回空 choices 或无 content 的片段，需要跳过
choices = getattr(chunk, "choices", None)
if not choices:
    continue

delta = getattr(choices[0], "delta", None)
if not delta:
    continue

content = getattr(delta, "content", "") or ""
if not content:
    continue
```

## ReAct

接下来尝试实现一个ReAct范式的agent demo。

### Tool

React结构的agent离不开tool的帮助，这里尝试实现的是一个调用google查询api来实现信息搜素功能的tool。在[SerpApi](https://serpapi.com/)申请免费套餐获得API 可以后，在`.env`里配置`SERPAPI_API_KEY`即可。具体工具实现代码如下：

```python
import os
import sys

from dotenv import load_dotenv
from serpapi import SerpApiClient

# 加载 .env 文件中的环境变量
load_dotenv()

def search(query: str) -> str:
    """
    一个基于SerpApi的网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


if __name__ == "__main__":
    # 用法:
    # python search_tool.py "今天北京天气"
    # 或直接运行后手动输入问题
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("请输入要搜索的问题: ").strip()

    if not query:
        print("错误: 查询内容不能为空。")
        sys.exit(1)

    print("--- 搜索结果 ---")
    print(search(query))

```

在原有代码基础上加入了main函数测试调用，具体方法为
```terminal
python search_tool.py "你的问题"
```

此外，还需实现一个工具管理类，具体代码如下：

```python
from typing import Dict, Any
from search_tool import search

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])

# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")    

```

具体实现了对工具的注册、获取、打印所有可用工具这几个方法。

### ReAct agent

具体实现如下：

```python
import re
from my_llm import MyLLM
from tool_executor import ToolExecutor
from dotenv import load_dotenv
from search_tool import search

# 加载 .env 文件中的环境变量
load_dotenv()

# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(self, llm_client: MyLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = [] # 每次运行时重置历史记录
        current_step = 0
        print(f"[AGENT][START] Question: {question}")

        while current_step < self.max_steps:
            current_step += 1
            step_prefix = f"[AGENT][STEP {current_step}]"
            print(f"{step_prefix}[BEGIN]")

            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
            print(f"{step_prefix}[PROMPT_READY] history_items={len(self.history)}")

            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print(f"{step_prefix}[ERROR] LLM未能返回有效响应。")
                break
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"{step_prefix}[THOUGHT] {thought}")

            if not action:
                print(f"{step_prefix}[WARN] 未能解析出有效的Action，流程终止。")
                break
            print(f"{step_prefix}[ACTION_RAW] {action}")

            # 4. 执行Action
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if not finish_match:
                    print(f"{step_prefix}[WARN] Finish格式无效: {action}")
                    break
                final_answer = finish_match.group(1).strip()
                print(f"{step_prefix}[FINISH] {final_answer}")
                return final_answer
            
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print(f"{step_prefix}[WARN] Action格式无效，跳过本轮。")
                continue

            print(f"{step_prefix}[TOOL_CALL] {tool_name}[{tool_input}]")
            
            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) # 调用真实工具
            
            print(f"{step_prefix}[OBSERVATION] {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        # 循环结束
        print("[AGENT][END] 已达到最大步数或提前终止。")
        return None


    def _parse_output(self, text: str):
        """
        解析LLM的输出，提取Thought和Action。
        """
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        """
        解析Action字符串，提取工具名称和输入。
        """
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None


if __name__ == '__main__':
    # 1. 初始化 LLM 客户端与工具执行器
    llm_client = MyLLM()
    tool_executor = ToolExecutor()

    # 2. 注册工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_description, search)

    # 3. 创建并运行 ReAct Agent
    agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor, max_steps=5)
    question = "英伟达最新的GPU型号是什么？"
    print(f"[AGENT][MAIN] Question: {question}")
    agent.run(question)


```

上述部分相比于教程中的源码做了两点优化，一是优化了打印信息，去除了一些不同阶段的重复打印，并加上了详细的信息来源，如agent还是tool还是llm。

另一点是在对`Finish[]`的识别语句进行了优化。原有语句为：
```python
final_answer = re.match(r"Finish\[(.*)\]", action).group(1)

```
当中括号中的内容发生换行的时候，原有写法会导致识别失效，并且如果输出格式错误，会直接导致报错`AttributeError: 'NoneType' object has no attribute 'group'`。因而替换成了如下写法：

```python
finish_match = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
if not finish_match:
    print(f"{step_prefix}[WARN] Finish格式无效: {action}")
    break
final_answer = finish_match.group(1).strip()
```

这样加入`re.DOTALL`允许跨行匹配，更容易匹配成功；即使有其他非标准格式的输出，也不会直接报错。

这样最终的输出结果如下：

```terminal
工具 'Search' 已注册。
[AGENT][MAIN] Question: 英伟达最新的GPU型号是什么？
[AGENT][START] Question: 英伟达最新的GPU型号是什么？
[AGENT][STEP 1][BEGIN]
[AGENT][STEP 1][PROMPT_READY] history_items=0
🧠 正在调用 glm-5.1 模型...
✅ 大语言模型响应成功:
Thought: 这是一个关于最新科技产品的事实性问题。由于GPU型号更新非常快，我的内部知识可能不是最新的，我需要使用搜索引擎来获取英伟达最新GPU型号的准确信息。
Action: Search[英伟达最新GPU型号 2024]
[AGENT][STEP 1][THOUGHT] 这是一个关于最新科技产品的事实性问题。由于GPU型号更新非常快，我的内部知识可能不是最新的，我需要使用搜索引擎来获取英伟达最新GPU型号的准确信息。
[AGENT][STEP 1][ACTION_RAW] Search[英伟达最新GPU型号 2024]
[AGENT][STEP 1][TOOL_CALL] Search[英伟达最新GPU型号 2024]
[TOOL][Search][REQUEST] query=英伟达最新GPU型号 2024
[TOOL][Search][RESULT] source=organic_results top=3
[AGENT][STEP 1][OBSERVATION] [1] 加速生成式AI、G-SYNC 创新、RTX Remix 公开测试版、多路 ...
NVIDIA 隆重开启2024 年，在CES 2024 展会上推出了全新GeForce RTX 40 SUPER 系列显卡、全新RTX 游戏、全新G-SYNC 技术、生成式AI 领域的创新、针对RTX ...

[2] 一文彻底读懂：英伟达GPU分类、架构演进和参数解析
Quadro系列是英伟达专业级GPU产品线，针对商业和专业应用领域进行了优化。常见的产品型号如NVIDIA RTX A6000、A5000等。 Quadro GPU具备强大的计算能力、大 ...

[3] 2024-geforce-rtx-laptops-available-now
NVIDIA Studio 2024 款微星绝影16 AI Studio 可配备GeForce RTX 4090 笔记本电脑GPU，从而达到移动设备可以实现的顶级AI 性能。尽管拥有如此强大的性能，但 ...
[AGENT][STEP 2][BEGIN]
[AGENT][STEP 2][PROMPT_READY] history_items=2
🧠 正在调用 glm-5.1 模型...
✅ 大语言模型响应成功:
Thought: 根据搜索结果，英伟达在2024年CES上推出了GeForce RTX 40 SUPER系列显卡，这是目前最新的消费级GPU产品线。同时搜索结果也提到了RTX 4090笔记本GPU等高端型号。综合来看，RTX 40 SUPER系列（包括RTX 4080 SUPER、RTX 4070 Ti SUPER和RTX 4070 SUPER）是英伟达最新的GPU型号。

Action: Finish[英伟达最新的GPU型号是GeForce RTX 40 SUPER系列，包括RTX 4080 SUPER、RTX 4070 Ti SUPER和RTX 4070 SUPER，这些产品于2024年CES展会上发布。]
[AGENT][STEP 2][THOUGHT] 根据搜索结果，英伟达在2024年CES上推出了GeForce RTX 40 SUPER系列显卡，这是目前最新的消费级GPU产品线。同时搜索结果也提到了RTX 4090笔记本GPU等高端型号。综合来看，RTX 40 SUPER系列（包括RTX 4080 SUPER、RTX 4070 Ti SUPER和RTX 4070 SUPER）是英伟达最新的GPU型号。
[AGENT][STEP 2][ACTION_RAW] Finish[英伟达最新的GPU型号是GeForce RTX 40 SUPER系列，包括RTX 4080 SUPER、RTX 4070 Ti SUPER和RTX 4070 SUPER，这些产品于2024年CES展会上发布。]
[AGENT][STEP 2][FINISH] 英伟达最新的GPU型号是GeForce RTX 40 SUPER系列，包括RTX 4080 SUPER、RTX 4070 Ti SUPER和RTX 4070 SUPER，这些产品于2024年CES展会上发布。
```
