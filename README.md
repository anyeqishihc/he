[multi_agent_storyboard_system.py](https://github.com/user-attachments/files/27195065/multi_agent_storyboard_system.py)
"""
Multi-Agent Storyboard Automation System

A complete, runnable Python prototype for a different Agent workflow:

Function:
  Turn a story idea into a structured cinematic storyboard package.

Input:
  - story idea / logline
  - genre
  - target shot count

Output:
  - story analysis
  - character relationship map
  - scene beat plan
  - shot-by-shot storyboard
  - visual prompt package
  - QA report

Run:
  python multi_agent_storyboard_system.py --idea "前刑警收账人发现欠债青年和自己女儿有关" --genre "港式犯罪悬疑" --shots 12 --out ./storyboard_outputs

Optional with OpenAI API:
  export OPENAI_API_KEY="your_key"
  python multi_agent_storyboard_system.py --idea "前刑警收账人发现欠债青年和自己女儿有关" --genre "港式犯罪悬疑" --shots 12 --out ./storyboard_outputs --use-openai
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =========================
# Security / Stability Config
# =========================

MAX_IDEA_CHARS = 800
MAX_GENRE_CHARS = 80
MIN_SHOTS = 3
MAX_SHOTS = 40


# =========================
# Data Models
# =========================

@dataclass
class ProjectInput:
    idea: str
    genre: str
    shot_count: int
    output_dir: str
    language: str = "zh-CN"


@dataclass
class AgentResult:
    agent_name: str
    output: Dict[str, Any]
    created_at: str


@dataclass
class StoryboardPipelineResult:
    project: ProjectInput
    story_analysis: AgentResult
    character_map: AgentResult
    beat_plan: AgentResult
    storyboard: AgentResult
    visual_prompts: AgentResult
    qa_report: AgentResult
    saved_files: List[str]


# =========================
# Utilities
# =========================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def compact_multiline(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in textwrap.dedent(text).strip().splitlines()
        if line.strip()
    )


def validate_text(value: str, name: str, max_chars: int) -> str:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError(f"{name} cannot be empty.")
    if len(cleaned) > max_chars:
        raise ValueError(f"{name} is too long. Max characters: {max_chars}")
    return cleaned


def validate_shot_count(value: int) -> int:
    if value < MIN_SHOTS or value > MAX_SHOTS:
        raise ValueError(f"shot_count must be between {MIN_SHOTS} and {MAX_SHOTS}.")
    return value


# =========================
# Optional LLM Client
# =========================

class LLMClient:
    """Default mode is deterministic fallback output.

    The project runs without external dependencies.
    If --use-openai is enabled and OPENAI_API_KEY exists, it calls OpenAI.
    """

    def __init__(self, use_openai: bool = False, model: str = "gpt-4o-mini"):
        self.use_openai = use_openai
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

        if self.use_openai and not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when --use-openai is enabled.")

    def complete_json(self, system: str, user: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_openai:
            return fallback

        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("LLM output must be a JSON object.")
            return parsed
        except Exception as exc:
            return {
                "warning": "OpenAI call failed; deterministic fallback returned.",
                "error_type": exc.__class__.__name__,
                "fallback": fallback,
            }


# =========================
# Base Agent
# =========================

class BaseAgent:
    name = "BaseAgent"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, context: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError

    def result(self, output: Dict[str, Any]) -> AgentResult:
        return AgentResult(agent_name=self.name, output=output, created_at=now_iso())


# =========================
# Agent 1: Story Analysis
# =========================

class StoryAnalysisAgent(BaseAgent):
    name = "StoryAnalysisAgent"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        project: ProjectInput = context["project"]
        fallback = {
            "logline": project.idea,
            "genre": project.genre,
            "core_conflict": "债务、亲情软肋与旧警察身份之间的道德冲突。",
            "tone": ["冷峻", "压抑", "现实犯罪", "港式霓虹", "情感克制"],
            "theme": ["父亲的软肋", "身份失落", "暴力秩序", "年轻人的债务困局"],
            "visual_world": {
                "time": "雨夜或傍晚",
                "city": "潮湿、拥挤、霓虹反光的南方港口城市",
                "texture": "玻璃水迹、旧楼梯、钞票、手机屏幕冷光、烟雾"
            }
        }
        system = "你是影视故事分析 Agent。只输出 JSON 对象。"
        user = f"分析故事创意，提炼类型、冲突、主题与视觉世界。故事：{project.idea}；类型：{project.genre}"
        output = self.llm.complete_json(system, user, fallback)
        return self.result(output)


# =========================
# Agent 2: Character Relationship Map
# =========================

class CharacterMapAgent(BaseAgent):
    name = "CharacterMapAgent"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        project: ProjectInput = context["project"]
        fallback = {
            "characters": [
                {
                    "name": "老陈",
                    "role": "45岁前刑警，地下钱庄收账人",
                    "desire": "完成收账，维持女儿大学生活",
                    "weakness": "女儿是他的软肋",
                    "visual": "旧黑夹克、疲惫眼神、右腿微跛"
                },
                {
                    "name": "阿亮",
                    "role": "20岁欠债青年",
                    "desire": "逃离债务和追债人",
                    "weakness": "冲动、虚荣、害怕承担后果",
                    "visual": "染黄毛、黑色宽松T恤、廉价耳钉"
                },
                {
                    "name": "陈雨",
                    "role": "老陈读大学的女儿",
                    "desire": "摆脱父亲阴影，追求普通生活",
                    "weakness": "不知道父亲真实工作",
                    "visual": "大学生、帆布包、干净但疲惫"
                }
            ],
            "relationships": [
                "老陈追债阿亮",
                "阿亮与陈雨有隐秘联系",
                "老陈的职业暴力开始反噬家庭",
                "陈雨成为老陈道德选择的核心压力点"
            ],
            "dramatic_question": "老陈会继续完成收账，还是保护与女儿有关的欠债青年？"
        }
        system = "你是角色关系设计 Agent。只输出 JSON 对象。"
        user = f"根据故事建立人物关系网与人物动机。故事：{project.idea}"
        output = self.llm.complete_json(system, user, fallback)
        return self.result(output)


# =========================
# Agent 3: Beat Plan
# =========================

class BeatPlanAgent(BaseAgent):
    name = "BeatPlanAgent"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        project: ProjectInput = context["project"]
        fallback = {
            "structure": "三幕式短片分镜结构",
            "beats": [
                {"beat": 1, "name": "建立世界", "purpose": "展示老陈的收账职业与城市气质"},
                {"beat": 2, "name": "目标出现", "purpose": "阿亮被锁定，欠债压力明确"},
                {"beat": 3, "name": "意外关联", "purpose": "老陈发现阿亮手机里有女儿的信息"},
                {"beat": 4, "name": "道德拉扯", "purpose": "老陈在职业规则与父亲身份之间摇摆"},
                {"beat": 5, "name": "沉默选择", "purpose": "老陈做出不说破的决定，留下悬念"}
            ],
            "target_shot_count": project.shot_count,
            "rhythm": "慢建立，中段紧张，结尾克制留白"
        }
        system = "你是影视节拍规划 Agent。只输出 JSON 对象。"
        user = json.dumps({
            "project": asdict(project),
            "story_analysis": context["story_analysis"].output,
            "character_map": context["character_map"].output,
        }, ensure_ascii=False)
        output = self.llm.complete_json(system, user, fallback)
        return self.result(output)


# =========================
# Agent 4: Shot-by-Shot Storyboard
# =========================

class StoryboardAgent(BaseAgent):
    name = "StoryboardAgent"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        project: ProjectInput = context["project"]
        shots = []
        base_shots = [
            ("远景", "雨夜天桥下，霓虹倒映在积水里，老陈一瘸一拐走入画面。", "建立城市与人物状态"),
            ("中近景", "老陈低头看手机上的欠款名单，屏幕冷光照亮疲惫的脸。", "交代收账任务"),
            ("跟拍", "阿亮穿过狭窄巷子，染黄头发在红色招牌下闪过。", "引出欠债青年"),
            ("手持近景", "老陈抓住阿亮衣领，把他抵在铁门边，雨水从屋檐滴落。", "制造冲突"),
            ("插入特写", "阿亮手机摔到地上，屏幕亮起：陈雨的未读消息。", "揭示女儿关联"),
            ("特写", "老陈眼神瞬间停住，拳头没有落下。", "情绪转折"),
            ("双人中景", "阿亮喘息着解释，老陈沉默听着，背景是卷帘门和霓虹。", "人物关系复杂化"),
            ("侧脸特写", "老陈点燃烟，烟雾遮住半张脸。", "道德拉扯"),
            ("主观镜头", "老陈看见远处女儿发来的语音通话请求。", "软肋压力"),
            ("静态远景", "老陈转身离开，阿亮留在雨中不知所措。", "选择与悬念"),
            ("背影镜头", "老陈拖着微跛的右腿走进人群，霓虹吞没他的背影。", "人物孤独"),
            ("空镜", "地上的手机还亮着，雨水不断落下。", "留白结尾"),
        ]

        for index in range(project.shot_count):
            shot_type, description, purpose = base_shots[index % len(base_shots)]
            shots.append({
                "shot_id": index + 1,
                "shot_type": shot_type,
                "description": description,
                "camera": {
                    "lens": "35mm or 50mm cinematic lens",
                    "movement": "static / slow handheld / slow push-in",
                    "angle": "eye level, realistic crime drama"
                },
                "lighting": "cold neon, wet reflections, low-key contrast, soft rain haze",
                "purpose": purpose,
                "dialogue_hint": "少台词，以停顿和眼神推动情绪"
            })

        fallback = {
            "shot_count": project.shot_count,
            "storyboard": shots,
            "continuity_rules": [
                "老陈始终疲惫克制，右腿微跛",
                "阿亮始终保留染黄发和年轻欠债人的慌张气质",
                "整体光线保持冷峻港式犯罪质感",
                "关键道具：手机、欠款名单、雨水、霓虹、香烟"
            ]
        }
        system = "你是专业影视分镜 Agent。只输出 JSON 对象。"
        user = json.dumps({
            "project": asdict(project),
            "beat_plan": context["beat_plan"].output,
        }, ensure_ascii=False)
        output = self.llm.complete_json(system, user, fallback)
        return self.result(output)


# =========================
# Agent 5: Visual Prompt Package
# =========================

class VisualPromptAgent(BaseAgent):
    name = "VisualPromptAgent"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        storyboard = context["storyboard"].output.get("storyboard", [])
        prompts = []

        for shot in storyboard:
            prompt = compact_multiline(f"""
            港式犯罪悬疑电影剧照，{shot.get('description', '')}
            镜头：{shot.get('shot_type', '')}，平视真实摄影，冷峻霓虹光，潮湿街道反光，细雨，轻微体积烟雾，低饱和，高动态范围，电影胶片颗粒，真实人物皮肤质感，35mm/50mm 镜头，浅景深，严肃克制，非夸张动作。
            """)
            prompts.append({
                "shot_id": shot.get("shot_id"),
                "image_prompt_zh": prompt,
                "negative_prompt": "cartoon, anime, fantasy armor, clean studio, over saturated, low quality, distorted face, extra fingers, watermark, text error"
            })

        fallback = {
            "prompt_count": len(prompts),
            "visual_style_bible": {
                "genre": "港式犯罪悬疑",
                "color": "冷蓝、暗绿、霓虹红、潮湿黑",
                "texture": "雨水、旧楼、手机冷光、烟雾、胶片颗粒",
                "camera": "真实电影摄影，不夸张，不MV化"
            },
            "prompts": prompts
        }
        system = "你是影视视觉 Prompt Agent。只输出 JSON 对象。"
        user = json.dumps({"storyboard": storyboard}, ensure_ascii=False)
        output = self.llm.complete_json(system, user, fallback)
        return self.result(output)


# =========================
# Agent 6: Quality Assurance
# =========================

class QualityAssuranceAgent(BaseAgent):
    name = "QualityAssuranceAgent"

    def run(self, context: Dict[str, Any]) -> AgentResult:
        project: ProjectInput = context["project"]
        storyboard = context["storyboard"].output.get("storyboard", [])
        prompts = context["visual_prompts"].output.get("prompts", [])

        checks = {
            "shot_count_match": len(storyboard) == project.shot_count,
            "has_story_conflict": "core_conflict" in context["story_analysis"].output,
            "has_character_relationships": "relationships" in context["character_map"].output,
            "has_visual_prompts": len(prompts) == project.shot_count,
            "has_continuity_rules": "continuity_rules" in context["storyboard"].output,
        }
        score = sum(1 for value in checks.values() if value) / len(checks)
        output = {
            "checks": checks,
            "score": round(score, 2),
            "status": "pass" if score >= 0.8 else "needs_revision",
            "suggestions": [] if score >= 0.8 else [
                "检查分镜数量是否匹配",
                "补充人物关系与冲突",
                "为每个镜头生成对应视觉 prompt"
            ]
        }
        return self.result(output)


# =========================
# Markdown Exporter
# =========================

class MarkdownExporter:
    def export(self, result: StoryboardPipelineResult, output_dir: Path) -> str:
        lines = [
            "# Multi-Agent Storyboard Package",
            "",
            f"**Idea:** {result.project.idea}",
            f"**Genre:** {result.project.genre}",
            f"**Shot Count:** {result.project.shot_count}",
            "",
            "## Storyboard",
            ""
        ]

        storyboard = result.storyboard.output.get("storyboard", [])
        for shot in storyboard:
            lines.extend([
                f"### Shot {shot.get('shot_id')}: {shot.get('shot_type')}",
                f"- Description: {shot.get('description')}",
                f"- Lighting: {shot.get('lighting')}",
                f"- Purpose: {shot.get('purpose')}",
                ""
            ])

        lines.extend(["## Visual Prompts", ""])
        prompts = result.visual_prompts.output.get("prompts", [])
        for item in prompts:
            lines.extend([
                f"### Prompt {item.get('shot_id')}",
                item.get("image_prompt_zh", ""),
                "",
                "Negative Prompt:",
                item.get("negative_prompt", ""),
                ""
            ])

        path = output_dir / "storyboard_package.md"
        save_text(path, "\n".join(lines))
        return str(path)


# =========================
# Orchestrator
# =========================

class StoryboardAgentOrchestrator:
    def __init__(self, llm: LLMClient, exporter: Optional[MarkdownExporter] = None):
        self.llm = llm
        self.exporter = exporter or MarkdownExporter()
        self.story_agent = StoryAnalysisAgent(llm)
        self.character_agent = CharacterMapAgent(llm)
        self.beat_agent = BeatPlanAgent(llm)
        self.storyboard_agent = StoryboardAgent(llm)
        self.visual_agent = VisualPromptAgent(llm)
        self.qa_agent = QualityAssuranceAgent(llm)

    def run(self, project: ProjectInput) -> StoryboardPipelineResult:
        output_dir = ensure_dir(project.output_dir)
        context: Dict[str, Any] = {"project": project}
        saved_files: List[str] = []

        story_analysis = self.story_agent.run(context)
        context["story_analysis"] = story_analysis
        path = output_dir / "01_story_analysis.json"
        save_json(path, asdict(story_analysis))
        saved_files.append(str(path))

        character_map = self.character_agent.run(context)
        context["character_map"] = character_map
        path = output_dir / "02_character_map.json"
        save_json(path, asdict(character_map))
        saved_files.append(str(path))

        beat_plan = self.beat_agent.run(context)
        context["beat_plan"] = beat_plan
        path = output_dir / "03_beat_plan.json"
        save_json(path, asdict(beat_plan))
        saved_files.append(str(path))

        storyboard = self.storyboard_agent.run(context)
        context["storyboard"] = storyboard
        path = output_dir / "04_storyboard.json"
        save_json(path, asdict(storyboard))
        saved_files.append(str(path))

        visual_prompts = self.visual_agent.run(context)
        context["visual_prompts"] = visual_prompts
        path = output_dir / "05_visual_prompts.json"
        save_json(path, asdict(visual_prompts))
        saved_files.append(str(path))

        qa_report = self.qa_agent.run(context)
        context["qa_report"] = qa_report
        path = output_dir / "06_qa_report.json"
        save_json(path, asdict(qa_report))
        saved_files.append(str(path))

        result = StoryboardPipelineResult(
            project=project,
            story_analysis=story_analysis,
            character_map=character_map,
            beat_plan=beat_plan,
            storyboard=storyboard,
            visual_prompts=visual_prompts,
            qa_report=qa_report,
            saved_files=saved_files,
        )

        md_path = self.exporter.export(result, output_dir)
        saved_files.append(md_path)
        result.saved_files = saved_files

        pipeline_path = output_dir / "pipeline_result.json"
        save_json(pipeline_path, asdict(result))
        saved_files.append(str(pipeline_path))
        result.saved_files = saved_files
        save_json(pipeline_path, asdict(result))

        return result


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Storyboard Automation System")
    parser.add_argument("--idea", required=True, help="Story idea / logline")
    parser.add_argument("--genre", default="港式犯罪悬疑", help="Genre style")
    parser.add_argument("--shots", type=int, default=12, help="Target shot count")
    parser.add_argument("--out", default="./storyboard_outputs", help="Output directory")
    parser.add_argument("--use-openai", action="store_true", help="Enable OpenAI JSON generation")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        idea = validate_text(args.idea, "idea", MAX_IDEA_CHARS)
        genre = validate_text(args.genre, "genre", MAX_GENRE_CHARS)
        shot_count = validate_shot_count(args.shots)
        output_dir = ensure_dir(args.out)

        project = ProjectInput(
            idea=idea,
            genre=genre,
            shot_count=shot_count,
            output_dir=str(output_dir),
        )
        llm = LLMClient(use_openai=args.use_openai, model=args.model)
        orchestrator = StoryboardAgentOrchestrator(llm)
        result = orchestrator.run(project)

        print("\n=== Multi-Agent Storyboard Pipeline Finished ===")
        print(f"QA status: {result.qa_report.output['status']}")
        print(f"QA score: {result.qa_report.output['score']}")
        print("\nSaved files:")
        for file in result.saved_files:
            print(f"- {file}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
