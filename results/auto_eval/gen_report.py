#!/usr/bin/env python3
"""Generate HTML briefing report for Gemini ER 1.5 auto-eval results."""
import json, base64, pathlib, datetime

ROOT = pathlib.Path(__file__).parent

def load_json(path):
    with open(path) as f:
        return json.load(f)

def obj_label(o):
    if isinstance(o, dict):
        return o.get("label", o.get("name", str(o)))
    return str(o)

def img_b64(path):
    if pathlib.Path(path).exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

d1 = load_json(ROOT / "20260321_014258_affordance_pass.json")      # lenient 85pt PASS
d2 = load_json(ROOT / "20260321_015353_manipulation_partial_75pt.json")  # strict 75pt PARTIAL

scene_b64 = img_b64(ROOT / "scene_latest.jpg")
dashboard_b64 = img_b64(ROOT / "dashboard_latest.png")

def score_color(score):
    if score >= 80: return "#4ade80"
    if score >= 60: return "#facc15"
    return "#f87171"

def result_badge(r):
    colors = {"pass": "#4ade80", "fail": "#f87171", "partial": "#facc15", "warn": "#facc15"}
    c = colors.get(r.lower(), "#94a3b8")
    return f'<span style="background:{c};color:#0f172a;padding:2px 10px;border-radius:999px;font-size:0.75rem;font-weight:700;text-transform:uppercase;">{r}</span>'

def criteria_table(eval_data):
    rows = ""
    for c in eval_data.get("criteria_results", []):
        score = c.get("score", "")
        score_str = f"{score}/15" if score != "" else ""
        rows += f"""
        <tr>
          <td style="padding:10px 12px;border-bottom:1px solid #1e293b;color:#cbd5e1;font-size:0.85rem;">{c['criterion']}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1e293b;text-align:center;">{result_badge(c['result'])}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1e293b;text-align:center;color:#94a3b8;font-size:0.85rem;">{score_str}</td>
          <td style="padding:10px 12px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:0.82rem;">{c.get('comment','')}</td>
        </tr>"""
    return rows

def action_steps_html(gemini_data):
    """Render action steps from either format."""
    steps = gemini_data.get("steps", [])
    if steps:
        items = ""
        for s in steps:
            obj = f' <span style="color:#38bdf8;">({s["object"]})</span>' if "object" in s else ""
            bbox = ""
            if "bbox" in s:
                b = s["bbox"]
                bbox = f'<div style="font-size:0.75rem;color:#64748b;margin-top:3px;">bbox: [{", ".join(f"{x:.3f}" for x in b)}]</div>'
            items += f'<div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;"><div style="width:28px;height:28px;border-radius:50%;background:#334155;color:#e2e8f0;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.8rem;flex-shrink:0;">{s["step"]}</div><div><div style="color:#e2e8f0;font-size:0.9rem;">{s["description"]}{obj}</div>{bbox}</div></div>'
        return items
    # flat format (d2 style)
    actions = ["pick", "lift", "move", "place"]
    items = ""
    for i, act in enumerate(actions, 1):
        if act in gemini_data:
            v = gemini_data[act]
            details = json.dumps(v, ensure_ascii=False, indent=None)
            items += f'<div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;"><div style="width:28px;height:28px;border-radius:50%;background:#334155;color:#e2e8f0;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:0.8rem;flex-shrink:0;">{i}</div><div><div style="color:#e2e8f0;font-size:0.9rem;font-weight:600;text-transform:capitalize;">{act}</div><div style="font-size:0.78rem;color:#64748b;font-family:\'JetBrains Mono\',monospace;margin-top:3px;">{details}</div></div></div>'
    return items

def gauge_svg(score, color):
    """Simple arc gauge."""
    angle = min(score / 100, 1.0) * 180  # 0-180 degrees (semicircle)
    r = 60
    cx, cy = 75, 75
    # convert angle to path
    import math
    rad = math.radians(180 - angle)
    ex = cx + r * math.cos(rad)
    ey = cy - r * math.sin(rad)
    large = 1 if angle > 180 else 0
    return f"""<svg width="150" height="90" viewBox="0 0 150 90">
      <path d="M15,75 A{r},{r} 0 0,1 135,75" fill="none" stroke="#1e293b" stroke-width="12" stroke-linecap="round"/>
      <path d="M15,75 A{r},{r} 0 {large},1 {ex:.1f},{ey:.1f}" fill="none" stroke="{color}" stroke-width="12" stroke-linecap="round"/>
      <text x="75" y="68" text-anchor="middle" fill="{color}" font-size="22" font-weight="700" font-family="Inter,sans-serif">{score}</text>
      <text x="75" y="83" text-anchor="middle" fill="#64748b" font-size="11" font-family="Inter,sans-serif">/ 100</text>
    </svg>"""

eval1 = d1["eval"]
eval2 = d2["eval"]
s1 = eval1["score"]
s2 = eval2["score"]
c1 = score_color(s1)
c2 = score_color(s2)

objects1 = ", ".join([obj_label(o) for o in d1["design"].get("detected_objects", [])])
objects2 = ", ".join([obj_label(o) for o in d2["design"].get("detected_objects", [])])

gemini1_json = json.dumps(d1["gemini"], ensure_ascii=False, indent=2)
gemini2_json = json.dumps(d2["gemini"], ensure_ascii=False, indent=2)

scene_img_tag = f'<img src="data:image/jpeg;base64,{scene_b64}" style="width:100%;border-radius:8px;display:block;" alt="Scene"/>' if scene_b64 else '<div style="background:#1e293b;border-radius:8px;height:200px;display:flex;align-items:center;justify-content:center;color:#475569;">No scene image</div>'

dashboard_section = ""
if dashboard_b64:
    dashboard_section = f"""
    <div class="section">
      <div class="section-title">Dashboard</div>
      <img src="data:image/png;base64,{dashboard_b64}" style="width:100%;border-radius:8px;" alt="Dashboard"/>
    </div>"""

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Gemini Robotics ER 1.5 — Embodied Reasoning Evaluation Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin:0; padding:0; }}
  body {{ background:#0a0f1e; color:#e2e8f0; font-family:'Inter',sans-serif; font-size:15px; line-height:1.6; }}
  .page {{ max-width:1200px; margin:0 auto; padding:40px 24px 80px; }}

  /* Header */
  .header {{ text-align:center; margin-bottom:56px; }}
  .header-tag {{ display:inline-block; background:#1e293b; color:#38bdf8; font-size:0.72rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; padding:5px 16px; border-radius:999px; margin-bottom:18px; border:1px solid #334155; }}
  .header h1 {{ font-size:2.6rem; font-weight:700; line-height:1.15; background:linear-gradient(135deg,#f1f5f9,#38bdf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; }}
  .header .subtitle {{ color:#64748b; font-size:1rem; max-width:600px; margin:0 auto; }}
  .header .meta {{ margin-top:16px; font-size:0.8rem; color:#475569; }}

  /* Pipeline */
  .pipeline {{ display:flex; align-items:center; justify-content:center; gap:0; margin-bottom:56px; flex-wrap:wrap; gap:4px; }}
  .pipe-node {{ background:#0f172a; border:1px solid #334155; border-radius:10px; padding:12px 18px; text-align:center; min-width:120px; }}
  .pipe-node .icon {{ font-size:1.4rem; margin-bottom:4px; }}
  .pipe-node .label {{ font-size:0.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; }}
  .pipe-node .name {{ font-size:0.85rem; font-weight:600; color:#e2e8f0; margin-top:2px; }}
  .pipe-arrow {{ color:#334155; font-size:1.4rem; padding:0 4px; }}

  /* Section */
  .section {{ background:#0f172a; border:1px solid #1e293b; border-radius:14px; padding:28px; margin-bottom:28px; }}
  .section-title {{ font-size:0.72rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:#38bdf8; margin-bottom:20px; padding-bottom:12px; border-bottom:1px solid #1e293b; }}

  /* Grid */
  .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
  .grid-3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px; }}
  @media(max-width:768px) {{ .grid-2,.grid-3 {{ grid-template-columns:1fr; }} }}

  /* Card */
  .card {{ background:#0a0f1e; border:1px solid #1e293b; border-radius:10px; padding:20px; }}
  .card-title {{ font-size:0.78rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px; }}

  /* Code block */
  .code-block {{ background:#020617; border:1px solid #1e293b; border-radius:8px; padding:16px; font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#94a3b8; overflow-x:auto; white-space:pre; line-height:1.7; max-height:340px; overflow-y:auto; }}

  /* Eval result */
  .eval-header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:20px; }}
  .eval-overall {{ font-size:1.5rem; font-weight:700; }}
  .eval-summary {{ color:#94a3b8; font-size:0.88rem; margin-top:6px; }}

  /* Table */
  table {{ width:100%; border-collapse:collapse; }}
  th {{ padding:10px 12px; text-align:left; font-size:0.72rem; font-weight:600; color:#64748b; text-transform:uppercase; letter-spacing:1px; border-bottom:1px solid #1e293b; }}
  td {{ vertical-align:top; }}

  /* Tag list */
  .tag-list {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:6px; }}
  .tag {{ background:#1e293b; color:#94a3b8; padding:3px 10px; border-radius:999px; font-size:0.75rem; }}
  .tag.green {{ background:#052e16; color:#4ade80; border:1px solid #166534; }}
  .tag.red {{ background:#2d0a0a; color:#f87171; border:1px solid #7f1d1d; }}
  .tag.blue {{ background:#0c1a2e; color:#38bdf8; border:1px solid #0e4a7a; }}

  /* Roadmap */
  .roadmap-row {{ display:flex; gap:16px; padding:14px 0; border-bottom:1px solid #1e293b; }}
  .roadmap-row:last-child {{ border-bottom:none; }}
  .roadmap-phase {{ font-size:0.72rem; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:1px; min-width:90px; padding-top:2px; }}
  .roadmap-content {{ flex:1; }}
  .roadmap-title {{ font-weight:600; color:#e2e8f0; font-size:0.9rem; }}
  .roadmap-desc {{ color:#64748b; font-size:0.82rem; margin-top:3px; }}

  /* Divider */
  .divider {{ height:1px; background:linear-gradient(90deg,transparent,#334155,transparent); margin:40px 0; }}

  /* Footer */
  .footer {{ text-align:center; color:#334155; font-size:0.78rem; padding-top:40px; }}
</style>
</head>
<body>
<div class="page">

  <!-- Header -->
  <div class="header">
    <div class="header-tag">Research Briefing</div>
    <h1>Gemini Robotics ER 1.5<br/>Embodied Reasoning Evaluation</h1>
    <p class="subtitle">Doosan E0509 로봇 시스템 연동 전 Gemini ER 모델의 공간 추론·조작 계획 능력 평가 보고서</p>
    <div class="meta">생성일: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} &nbsp;|&nbsp; 모델: gemini-robotics-er-1.5-preview &nbsp;|&nbsp; 평가자: GPT-4o</div>
  </div>

  <!-- Pipeline -->
  <div class="pipeline">
    <div class="pipe-node"><div class="icon">📷</div><div class="label">Input</div><div class="name">RealSense D435</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="icon">🧠</div><div class="label">Task Designer</div><div class="name">GPT-4o</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="icon">🤖</div><div class="label">Reasoner</div><div class="name">Gemini ER 1.5</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="icon">⚖️</div><div class="label">Evaluator</div><div class="name">GPT-4o</div></div>
    <div class="pipe-arrow">→</div>
    <div class="pipe-node"><div class="icon">📊</div><div class="label">Output</div><div class="name">This Report</div></div>
  </div>

  <!-- Scene -->
  <div class="section">
    <div class="section-title">Scene — RealSense D435 캡처</div>
    <div class="grid-2" style="align-items:start;">
      <div>{scene_img_tag}</div>
      <div>
        <div class="card" style="margin-bottom:16px;">
          <div class="card-title">씬 설명 (GPT-4o 분석)</div>
          <p style="color:#e2e8f0;font-size:0.9rem;">{d2['design']['scene_summary']}</p>
        </div>
        <div class="card">
          <div class="card-title">감지된 물체</div>
          <div class="tag-list">
            {''.join(f'<span class="tag blue">{obj_label(o)}</span>' for o in d2["design"].get("detected_objects", []))}
          </div>
          <div style="margin-top:16px;">
            {"".join(f'''<div style="display:flex;gap:10px;margin-bottom:8px;align-items:center;">
              <span class="tag">{obj_label(o)}</span>
              <span style="color:#64748b;font-size:0.78rem;">shape: {o.get("shape","—")} &nbsp;|&nbsp; ~{o.get("size_estimate_mm","?")}mm</span>
            </div>''' for o in d2["design"].get("detected_objects",[]) if isinstance(o,dict))}
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Eval 1: Lenient / Affordance -->
  <div class="section">
    <div class="section-title">Evaluation 1 — Lenient Affordance Test (85pt / PASS)</div>
    <div class="grid-2" style="align-items:start;margin-bottom:24px;">
      <div class="card">
        <div class="card-title">GPT-4o Task Design</div>
        <div style="margin-bottom:10px;"><span style="color:#64748b;font-size:0.78rem;">Task Type</span><div style="color:#38bdf8;font-weight:600;margin-top:2px;">{d1['design']['chosen_task']}</div></div>
        <div style="margin-bottom:10px;"><span style="color:#64748b;font-size:0.78rem;">Instruction to Gemini</span><div style="color:#e2e8f0;font-size:0.88rem;margin-top:4px;background:#0a0f1e;padding:10px;border-radius:6px;border:1px solid #1e293b;">{d1['design']['instruction_for_gemini']}</div></div>
        <div><span style="color:#64748b;font-size:0.78rem;">Objects Detected</span><div style="color:#e2e8f0;margin-top:2px;font-size:0.88rem;">{objects1}</div></div>
      </div>
      <div class="card">
        <div class="card-title">Gemini ER Response — Action Steps</div>
        {action_steps_html(d1['gemini'])}
      </div>
    </div>
    <!-- Criteria table -->
    <div style="margin-bottom:20px;">
      <table>
        <thead><tr><th>평가 기준</th><th style="text-align:center;">결과</th><th style="text-align:center;">점수</th><th>코멘트</th></tr></thead>
        <tbody>{criteria_table(eval1)}</tbody>
      </table>
    </div>
    <div class="eval-header">
      <div>
        <div class="eval-overall" style="color:{c1};">PASS &nbsp;·&nbsp; {s1}점</div>
        <div class="eval-summary">{eval1.get('summary','')}</div>
      </div>
      {gauge_svg(s1, c1)}
    </div>
    <div class="grid-2">
      <div>
        <div style="font-size:0.72rem;color:#4ade80;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Strengths</div>
        {''.join(f'<div style="display:flex;gap:8px;margin-bottom:6px;"><span style="color:#4ade80;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">{s}</span></div>' for s in eval1.get('strengths',[]))}
      </div>
      <div>
        <div style="font-size:0.72rem;color:#f87171;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Weaknesses</div>
        {''.join(f'<div style="display:flex;gap:8px;margin-bottom:6px;"><span style="color:#f87171;">✗</span><span style="color:#94a3b8;font-size:0.85rem;">{w}</span></div>' for w in eval1.get('weaknesses',[]))}
      </div>
    </div>
    <!-- Raw JSON -->
    <details style="margin-top:20px;">
      <summary style="cursor:pointer;color:#38bdf8;font-size:0.82rem;font-weight:600;">Gemini Raw JSON Output ▸</summary>
      <div class="code-block" style="margin-top:12px;">{gemini1_json}</div>
    </details>
  </div>

  <!-- Eval 2: Strict / Manipulation -->
  <div class="section">
    <div class="section-title">Evaluation 2 — Strict Manipulation Test (75pt / PARTIAL)</div>
    <div class="grid-2" style="align-items:start;margin-bottom:24px;">
      <div class="card">
        <div class="card-title">GPT-4o Task Design (Strict Mode)</div>
        <div style="margin-bottom:10px;"><span style="color:#64748b;font-size:0.78rem;">Task Category</span><div style="color:#38bdf8;font-weight:600;margin-top:2px;">{d2['design']['chosen_task']}</div></div>
        <div style="margin-bottom:10px;"><span style="color:#64748b;font-size:0.78rem;">Instruction to Gemini</span><div style="color:#e2e8f0;font-size:0.88rem;margin-top:4px;background:#0a0f1e;padding:10px;border-radius:6px;border:1px solid #1e293b;">{d2['design']['instruction_for_gemini']}</div></div>
        <div><span style="color:#64748b;font-size:0.78rem;">Expected Output Keys</span><div class="tag-list" style="margin-top:6px;">{''.join(f'<span class="tag">{k}</span>' for k in d2["design"].get("expected_output_keys",[]))}</div></div>
      </div>
      <div class="card">
        <div class="card-title">Gemini ER Response — Action Plan</div>
        {action_steps_html(d2['gemini'])}
        <div style="margin-top:12px;padding:10px;background:#0a0f1e;border-radius:6px;border:1px solid #1e293b;">
          <div style="font-size:0.75rem;color:#64748b;margin-bottom:4px;">bbox_norm</div>
          <div style="font-family:'JetBrains Mono',monospace;color:#38bdf8;font-size:0.82rem;">[{", ".join(f"{x:.3f}" for x in d2['gemini'].get('bbox_norm',[]))}]</div>
          <div style="font-size:0.75rem;color:#64748b;margin-top:8px;margin-bottom:4px;">grasp_point_norm</div>
          <div style="font-family:'JetBrains Mono',monospace;color:#38bdf8;font-size:0.82rem;">[{", ".join(f"{x:.3f}" for x in d2['gemini'].get('grasp_point_norm',[]))}]</div>
        </div>
      </div>
    </div>
    <!-- Criteria table -->
    <div style="margin-bottom:20px;">
      <table>
        <thead><tr><th>평가 기준 (7항목 × 15점)</th><th style="text-align:center;">결과</th><th style="text-align:center;">점수</th><th>코멘트</th></tr></thead>
        <tbody>{criteria_table(eval2)}</tbody>
      </table>
    </div>
    <div class="eval-header">
      <div>
        <div class="eval-overall" style="color:{c2};">PARTIAL &nbsp;·&nbsp; {s2}점</div>
        <div class="eval-summary">{eval2.get('summary','')}</div>
      </div>
      {gauge_svg(s2, c2)}
    </div>
    <div class="grid-2">
      <div>
        <div style="font-size:0.72rem;color:#4ade80;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Strengths</div>
        {''.join(f'<div style="display:flex;gap:8px;margin-bottom:6px;"><span style="color:#4ade80;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">{s}</span></div>' for s in eval2.get('strengths',[]))}
      </div>
      <div>
        <div style="font-size:0.72rem;color:#f87171;font-weight:600;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Weaknesses</div>
        {''.join(f'<div style="display:flex;gap:8px;margin-bottom:6px;"><span style="color:#f87171;">✗</span><span style="color:#94a3b8;font-size:0.85rem;">{w}</span></div>' for w in eval2.get('weaknesses',[]))}
      </div>
    </div>
    <!-- Raw JSON -->
    <details style="margin-top:20px;">
      <summary style="cursor:pointer;color:#38bdf8;font-size:0.82rem;font-weight:600;">Gemini Raw JSON Output ▸</summary>
      <div class="code-block" style="margin-top:12px;">{gemini2_json}</div>
    </details>
  </div>

  <!-- Interpretation -->
  <div class="section">
    <div class="section-title">결과 해석 — 로봇 연결 전 베이스라인</div>
    <div class="grid-3">
      <div class="card">
        <div class="card-title">잘 되는 것 (Strong)</div>
        <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px;">
          <div style="display:flex;gap:8px;"><span style="color:#4ade80;flex-shrink:0;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">씬 물체 감지 및 bbox 예측</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#4ade80;flex-shrink:0;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">grasp point 추출 (bbox 내부)</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#4ade80;flex-shrink:0;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">pick/lift/move/place 시퀀스 생성</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#4ade80;flex-shrink:0;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">gripper 크기 선택 (물체 크기 기반)</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#4ade80;flex-shrink:0;">✓</span><span style="color:#94a3b8;font-size:0.85rem;">상단 접근 방향 (pregrasp z > 0)</span></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">개선 필요 (Weak)</div>
        <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px;">
          <div style="display:flex;gap:8px;"><span style="color:#f87171;flex-shrink:0;">✗</span><span style="color:#94a3b8;font-size:0.85rem;">장애물 고려한 접근 방향 계획</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#f87171;flex-shrink:0;">✗</span><span style="color:#94a3b8;font-size:0.85rem;">collision_risk 구체적 근거 제시</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#f87171;flex-shrink:0;">✗</span><span style="color:#94a3b8;font-size:0.85rem;">좌표 정밀도 (픽셀 → 물리 단위 변환)</span></div>
        </div>
      </div>
      <div class="card">
        <div class="card-title">로봇 연결 후 기대</div>
        <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px;">
          <div style="display:flex;gap:8px;"><span style="color:#38bdf8;flex-shrink:0;">→</span><span style="color:#94a3b8;font-size:0.85rem;">실제 깊이 정보로 정밀 좌표 보정</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#38bdf8;flex-shrink:0;">→</span><span style="color:#94a3b8;font-size:0.85rem;">IK solver로 joint angle 변환</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#38bdf8;flex-shrink:0;">→</span><span style="color:#94a3b8;font-size:0.85rem;">GR00T 파인튜닝으로 도메인 적응</span></div>
          <div style="display:flex;gap:8px;"><span style="color:#38bdf8;flex-shrink:0;">→</span><span style="color:#94a3b8;font-size:0.85rem;">safety clamp으로 물리 위반 방지</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Roadmap -->
  <div class="section">
    <div class="section-title">로봇 연결 로드맵</div>
    <div class="roadmap-row">
      <div class="roadmap-phase">Phase 1<br/><span style="color:#64748b;font-size:0.7rem;">현재</span></div>
      <div class="roadmap-content">
        <div class="roadmap-title">Gemini ER 베이스라인 검증 ✓</div>
        <div class="roadmap-desc">RealSense → GPT-4o 설계 → Gemini 추론 → GPT-4o 평가 파이프라인 완성. bbox/grasp/action sequence 출력 확인.</div>
      </div>
    </div>
    <div class="roadmap-row">
      <div class="roadmap-phase">Phase 2</div>
      <div class="roadmap-content">
        <div class="roadmap-title">로봇 통신 레이어 연결</div>
        <div class="roadmap-desc">Doosan E0509 ROS2 드라이버 연결. /joint_states 수신, /dsr01/motion/move_joint 발행. 안전 클램프(±2.86°/step) 적용.</div>
      </div>
    </div>
    <div class="roadmap-row">
      <div class="roadmap-phase">Phase 3</div>
      <div class="roadmap-content">
        <div class="roadmap-title">좌표 변환 파이프라인</div>
        <div class="roadmap-desc">bbox_norm → 픽셀 → RealSense 깊이 → 카메라 3D → 로봇 베이스 좌표계. Camera-to-robot extrinsic 보정 필요.</div>
      </div>
    </div>
    <div class="roadmap-row">
      <div class="roadmap-phase">Phase 4</div>
      <div class="roadmap-content">
        <div class="roadmap-title">GR00T 파인튜닝 & 배포</div>
        <div class="roadmap-desc">LeRobot 포맷으로 시연 데이터 수집 → GR00T-N1.6-3B 파인튜닝 (4 GPU, 10k steps) → SmolVLA 경량 모델 병행.</div>
      </div>
    </div>
  </div>

  {dashboard_section}

  <!-- Latency -->
  <div class="section">
    <div class="section-title">성능 지표</div>
    <div class="grid-3">
      <div class="card" style="text-align:center;">
        <div class="card-title">Eval 1 Latency</div>
        <div style="font-size:2rem;font-weight:700;color:#38bdf8;">{d1.get('total_latency_s', 0):.1f}s</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">Affordance / Lenient</div>
      </div>
      <div class="card" style="text-align:center;">
        <div class="card-title">Eval 2 Latency</div>
        <div style="font-size:2rem;font-weight:700;color:#38bdf8;">{d2.get('total_latency_s', 0):.1f}s</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">Manipulation / Strict</div>
      </div>
      <div class="card" style="text-align:center;">
        <div class="card-title">Embodied Reasoning</div>
        <div style="font-size:2rem;font-weight:700;color:#4ade80;">Good</div>
        <div style="color:#64748b;font-size:0.8rem;margin-top:4px;">GPT-4o 종합 판정</div>
      </div>
    </div>
  </div>

  <div class="footer">
    Gemini Robotics ER 1.5 — Doosan E0509 Integration Research &nbsp;|&nbsp; {datetime.datetime.now().strftime('%Y-%m-%d')}
  </div>

</div>
</body>
</html>"""

out = ROOT / "gemini_er_eval_report.html"
out.write_text(html, encoding="utf-8")
print(f"Report saved: {out}")
print(f"File size: {out.stat().st_size / 1024:.1f} KB")
