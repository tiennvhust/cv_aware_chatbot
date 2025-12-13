#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 13:11:06 2025

@author: tienn
"""

import json
from datetime import datetime
from collections import defaultdict
import re
from typing import List, Dict, Any, Tuple

DATE_FMT_VARIANTS = ["%Y-%m-%d", "%Y-%m", "%Y/%m", "%Y"]

def parse_date(s: str) -> datetime:
    s = s.strip()
    if not s or s.lower() in ["present", "current", "now"]:
        return datetime.today()
    for fmt in DATE_FMT_VARIANTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # fallback: try to extract year-month
    m = re.match(r"(\d{4})[-/](\d{1,2})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return datetime(y, mo, 1)
    # year only
    m = re.match(r"(\d{4})", s)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1)
    raise ValueError(f"Unrecognized date format: {s}")

def months_between(a: datetime, b: datetime) -> int:
    if b < a:
        a, b = b, a
    return (b.year - a.year) * 12 + (b.month - a.month) + (1 if b.day >= a.day else 0)

def normalize_skill(s: str) -> str:
    return s.strip().lower()

def load_data(path: str = "work_history.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def format_atomic_data(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for block in data:
        frm = parse_date(block.get("start_date", ""))
        to = parse_date(block.get("end_date", ""))
        org_type = block.get("type", "")
        name = block.get("name", "")
        role = block.get("role", "")
        location = block.get("location", "")
        text = block.get("text", "")
        skills = block.get("skills", [])
        rows.append({
            "start_date": frm,
            "end_date": to,
            "type": org_type,
            "name": name,
            "location": location,
            "role": role,
            "text": text,
            "skills": skills,
        })
    return rows

def compute_skill_experience(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    skill_data = defaultdict(lambda: {"months": 0, "examples": []})
    for r in rows:
        duration = months_between(r["start_date"], r["end_date"])
        for sk in r["skills"]:
            key = normalize_skill(sk)
            skill_data[key]["months"] += duration
            skill_data[key]["examples"].append({
                "name": r["name"],
                "role": r["role"],
                "location": r["location"],
                "start_date": r["start_date"].strftime("%Y-%m"),
                "end_date": r["end_date"].strftime("%Y-%m"),
                "text": r["text"],
            })
    return skill_data

def total_experience_years(rows: List[Dict[str, Any]]) -> float:
    # Merge overlapping periods across all roles
    intervals = [(r["start_date"], r["end_date"]) for r in rows]
    intervals.sort(key=lambda x: x[0])
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    months = sum(months_between(s, e) for s, e in merged)
    return round(months / 12.0, 2)

def highest_education(profile: Dict[str, Any]) -> Dict[str, str]:
    # Define order of levels
    order = {
        "high school": 1,
        "associate": 2,
        "bachelor's": 3,
        "master's": 4,
        "phd": 5,
        "doctorate": 5,
    }
    edu = profile.get("education", [])
    best = None
    best_rank = -1
    for e in edu:
        lvl = e.get("level", "").strip().lower()
        rank = order.get(lvl, 0)
        if rank > best_rank:
            best_rank = rank
            best = e
    return best or {}

def score_job_fit(required_skills: List[str], skill_exp: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    matches = []
    gaps = []
    total = len(required_skills)
    covered_months = 0
    for sk in required_skills:
        k = normalize_skill(sk)
        if k in skill_exp and skill_exp[k]["months"] > 0:
            matches.append({
                "skill": sk,
                "months": skill_exp[k]["months"],
                "years": round(skill_exp[k]["months"] / 12.0, 2),
                "examples": skill_exp[k]["examples"][:3]  # show a few
            })
            covered_months += skill_exp[k]["months"]
        else:
            gaps.append({"skill": sk})
    coverage = round((len(matches) / total) * 100.0, 1) if total > 0 else 0.0
    avg_years_per_matched = round((covered_months / 12.0) / max(len(matches), 1), 2) if matches else 0.0
    return {
        "coverage_percent": coverage,
        "matched": matches,
        "missing": gaps,
        "avg_years_per_matched_skill": avg_years_per_matched
    }

def format_years(months: int) -> str:
    years = months / 12.0
    return f"{years:.2f} years"

def answer_skill_familiarity(skill_list: List[str], skill_exp: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    for sk in skill_list:
        k = normalize_skill(sk)
        data = skill_exp.get(k)
        if not data or data["months"] == 0:
            lines.append(f"- {sk}: No documented experience found.")
            continue
        lines.append(f"- {sk}: {format_years(data['months'])} total. Examples:")
        for ex in data["examples"][:3]:
            lines.append(f"  • {ex['role']} at {ex['name']} ({ex['from']} → {ex['to']}), {ex['text']}")
    return "\n".join(lines)

def answer_years_of_experience(rows: List[Dict[str, Any]]) -> str:
    years = total_experience_years(rows)
    return f"Approximately {years} years of total professional experience (accounting for overlapping periods)."

def answer_highest_education(profile: Dict[str, Any]) -> str:
    best = highest_education(profile)
    if not best:
        return "No education entries found."
    return f"Highest education: {best.get('level')} in {best.get('field')} from {best.get('school')}."

def answer_job_suitability(required_skills: List[str], skill_exp: Dict[str, Dict[str, Any]]) -> str:
    fit = score_job_fit(required_skills, skill_exp)
    lines = []
    lines.append(f"Skill coverage: {fit['coverage_percent']}% of required skills.")
    if fit["matched"]:
        lines.append("Matched skills:")
        for m in fit["matched"]:
            lines.append(f"- {m['skill']}: {m['years']} (examples shown below)")
            for ex in m["examples"]:
                lines.append(f"  • {ex['role']} at {ex['name']} ({ex['from']} → {ex['to']}), {ex['text']}")
    if fit["missing"]:
        lines.append("Missing skills:")
        for g in fit["missing"]:
            lines.append(f"- {g['skill']}")
    lines.append(f"Average years per matched skill: {fit['avg_years_per_matched_skill']}")
    return "\n".join(lines)

def parse_user_input(s: str) -> Tuple[str, List[str]]:
    s = s.strip()
    if ":" in s:
        prefix, rest = s.split(":", 1)
        prefix = prefix.strip().lower()
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        return prefix, parts
    return s.lower(), []