# Auto Skill 质量控制设计文档

## 背景

当前的 Auto Skill 创建逻辑主要依赖工具调用次数、最终回答文本是否看起来成功，以及是否存在失败后恢复的迹象。该方案能够工作，但存在以下问题：

1. **复杂度代理过粗**：仅靠 `auto_skill_min_tool_calls` 难以区分“复杂且可复用的流程”和“冗长但低价值的试错过程”。
2. **成功判定偏脆弱**：字符串级别的 `error` / `failed` 判断容易误伤诸如 `no error found` 之类的正常结果。
3. **缺少内容质量闸门**：当前生成的 `SKILL.md` 更接近执行流水账，容易夹带绝对路径、时间戳或一次性调试细节。
4. **技能池污染风险**：所有自动生成技能默认都进入可加载目录，会让低置信度流程过早影响后续任务。

## 目标

本次优化希望将 Auto Skill 从“成功一次即沉淀”升级为“通过质量评分后分层沉淀”，具体目标如下：

- 引入**评分制质量闸门**，替代单一布尔触发条件。
- 将自动生成技能划分为 **approved** 与 **candidate** 两层。
- 对技能内容执行**去噪与脱敏**，降低环境耦合。
- 为生成结果补充结构化质量元数据，便于后续调优。
- 保持现有 CLI / ACP 集成方式基本不变，确保向后兼容。

## 非目标

- 不引入额外数据库或远程服务。
- 不对 Skill Loader 的相关性检索策略做语义级重写。
- 不在本次实现中引入人工审核 UI。

## 总体方案

### 一、评分制质量闸门

新增 `AutoSkillQualityReport`，在落盘前对当前 turn 的 trace 进行评估。评分考虑以下维度：

1. **执行质量**
   - 最终结果是否成功
   - 是否达到多步流程阈值
   - 是否存在稳定收敛（最后若干步无错误）
   - 是否为“全程稳定”或“失败后恢复”

2. **可复用性**
   - 是否涉及多个工具
   - 是否形成 `discovery -> action -> validation` 的闭环
   - 最终结果是否具有足够信息量
   - 是否包含明显环境特定数据

当总分达到阈值时才允许生成 Auto Skill。

### 二、分层落盘

自动生成技能分两层保存：

- **approved**：直接保存到 `auto_skill_dir/`，可被 `SkillLoader` 自动扫描与加载。
- **candidate**：保存到 `auto_skill_dir/_candidates/`，默认不进入自动加载集合。

这样可以保留低置信度流程供后续人工检查，同时避免污染自动技能池。

### 三、内容清洗

新增文本清洗步骤，对生成内容做以下处理：

- 绝对路径替换为 `<path>`
- 时间戳替换为 `<timestamp>`
- UUID / 长随机 ID 替换为 `<id>`
- 长结果文本裁剪为简短“完成信号”摘要
- 保留用户请求语义，但弱化一次性环境细节

### 四、结构化元数据

为生成出的技能 frontmatter 增加如下质量信息：

```yaml
metadata:
  source: mini-agent
  trigger: complex-task
  auto_skill:
    tier: approved
    score: 8
    reasons:
      - final-result-successful
      - multi-step-workflow
    warnings:
      - environment-specific-data-detected
    metrics:
      total_steps: 5
      unique_tools: 2
      categories:
        - discovery
        - action
        - validation
```

这些元数据既能帮助调试，也为后续做排序、降权、清理提供依据。

## 配置设计

在 `ToolsConfig` 中新增以下配置项：

- `auto_skill_candidate_score`：达到该分值时落盘为 candidate。
- `auto_skill_approved_score`：达到该分值时落盘为 approved。

校验规则：

- 两个分值都必须为正整数。
- `auto_skill_approved_score >= auto_skill_candidate_score`

默认值建议：

- `auto_skill_candidate_score = 5`
- `auto_skill_approved_score = 8`

## 关键流程

### 1. 生成流程

1. 收集 turn trace
2. 抽取用户请求
3. 计算质量评分
4. 若分数不达标，则返回 `quality-gate`
5. 根据分数选择落盘层级（approved / candidate）
6. 对结果进行去噪和结构化摘要
7. 生成并校验 `SKILL.md`
8. 写入对应目录

### 2. 加载流程

`SkillLoader.discover_skills()` 在扫描 `SKILL.md` 时跳过 `_candidates/` 目录，仅加载 approved 技能。

### 3. CLI / ACP 刷新逻辑

当新技能写入后：

- 若 tier 为 `approved`，立即尝试增量加载到当前 `SkillLoader`
- 若 tier 为 `candidate`，只打印提示信息，不加入当前自动加载集合

## 测试策略

新增或增强以下测试场景：

1. 高质量 workflow 生成 approved 技能，并写入质量元数据。
2. 中等质量 workflow 生成 candidate 技能，并保存在 `_candidates/`。
3. candidate 技能不会被 `SkillLoader.discover_skills()` 自动加载。
4. 单步低质量 workflow 会被质量闸门拒绝。
5. 生成内容中的绝对路径、时间戳等会被脱敏。
6. 配置阈值不合法时抛出校验错误。

## 兼容性与迁移

- 现有 approved 技能仍位于 `auto_skill_dir/` 根目录，继续可用。
- 新增的 `_candidates/` 目录不会影响旧技能。
- `skills_external_dirs` 默认配置可保持不变，因为 `SkillLoader` 会跳过 `_candidates/`。

## 预期收益

实施后，Auto Skill 将具备以下改进：

- 低质量 workflow 不再轻易进入自动技能池。
- 技能内容更紧凑、可读、可复用。
- 后续可以基于质量元数据继续扩展晋升、降权、审查等机制。
- 对当前使用者而言，默认体验会更稳定，自动加载结果更可信。
