from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import re
from jsonschema import validate, ValidationError

app = FastAPI(title="Prompt Factory: Text -> JSON (image/marketing/agent)", version="1.0")

# -----------------------------
# 1) SCHEMAS
# -----------------------------
IMAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["primary", "constraints", "output"],
    "properties": {
        "primary": {"type": "string"},
        "style": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "array", "items": {"type": "string"}},
        "environment": {"type": "array", "items": {"type": "string"}},
        "composition": {
            "type": "object",
            "properties": {
                "shot": {"type": "string"},
                "lens": {"type": "string"},
                "aperture": {"type": "string"},
                "angle": {"type": "string"},
            },
            "additionalProperties": False
        },
        "quality": {
            "type": "object",
            "properties": {
                "detail": {"type": "string"},
                "resolution": {"type": "string"}
            },
            "additionalProperties": False
        },
        "constraints": {
            "type": "object",
            "required": ["negative"],
            "properties": {
                "negative": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "output": {
            "type": "object",
            "required": ["aspect_ratio", "num_images"],
            "properties": {
                "aspect_ratio": {"type": "string"},
                "num_images": {"type": "integer", "minimum": 1, "maximum": 8}
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

MARKETING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["product", "platform", "deliverables"],
    "properties": {
        "product": {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string"},
                "category": {"type": "string"},
                "key_benefits": {"type": "array", "items": {"type": "string"}},
                "price_point": {"type": "string"}
            },
            "additionalProperties": False
        },
        "audience": {
            "type": "object",
            "properties": {
                "segment": {"type": "string"},
                "pain_points": {"type": "array", "items": {"type": "string"}},
                "desires": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "platform": {"type": "string"},
        "objective": {"type": "string"},
        "tone": {"type": "array", "items": {"type": "string"}},
        "deliverables": {
            "type": "object",
            "required": ["hook_titles", "script_60s", "caption", "hashtags", "cta"],
            "properties": {
                "hook_titles": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "script_60s": {"type": "string"},
                "caption": {"type": "string"},
                "hashtags": {"type": "array", "items": {"type": "string"}},
                "cta": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            },
            "additionalProperties": False
        },
        "compliance": {
            "type": "object",
            "properties": {
                "avoid_claims": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

AGENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["role", "goals", "constraints", "outputs"],
    "properties": {
        "role": {"type": "string"},
        "goals": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "tools_allowed": {"type": "array", "items": {"type": "string"}},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "process": {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}},
                "ask_clarifying_if": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        },
        "outputs": {
            "type": "object",
            "required": ["format", "acceptance_criteria"],
            "properties": {
                "format": {"type": "string"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "minItems": 1}
            },
            "additionalProperties": False
        },
        "evaluation": {
            "type": "object",
            "properties": {
                "checklist": {"type": "array", "items": {"type": "string"}}
            },
            "additionalProperties": False
        }
    },
    "additionalProperties": False
}

ENVELOPE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["meta", "task"],
    "properties": {
        "meta": {
            "type": "object",
            "required": ["version", "lang", "task"],
            "properties": {
                "version": {"type": "string"},
                "lang": {"type": "string"},
                "task": {"type": "string", "enum": ["image", "marketing", "agent"]}
            },
            "additionalProperties": False
        },
        "task": {
            "oneOf": [IMAGE_SCHEMA, MARKETING_SCHEMA, AGENT_SCHEMA]
        }
    },
    "additionalProperties": False
}

# -----------------------------
# 2) API MODELS
# -----------------------------
TaskType = Literal["image", "marketing", "agent"]

class ConvertRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str = "vi"
    task: TaskType = "image"

class ConvertResponse(BaseModel):
    prompt_json: Dict[str, Any]
    valid: bool
    errors: Optional[List[str]] = None

# -----------------------------
# 3) PARSERS (rule-based baseline)
# -----------------------------
STYLE_KEYWORDS = ["photorealistic", "commercial", "cinematic", "editorial", "minimal", "3d", "anime", "watercolor", "film"]
SHOT_KEYWORDS = {
    "close-up": ["cận cảnh", "close-up", "macro"],
    "wide": ["toàn cảnh", "wide", "panorama"],
    "medium": ["trung cảnh", "medium shot"]
}
ASPECT_PATTERNS = [
    (re.compile(r"\b(1:1)\b"), "1:1"),
    (re.compile(r"\b(4:5)\b"), "4:5"),
    (re.compile(r"\b(16:9)\b"), "16:9"),
    (re.compile(r"\b(9:16)\b"), "9:16"),
]
LENS_PATTERN = re.compile(r"\b(\d{2,3}mm)\b", re.IGNORECASE)
APERTURE_PATTERN = re.compile(r"\b(f\/\d+(\.\d+)?)\b", re.IGNORECASE)

def extract_aspect_ratio(text: str) -> Optional[str]:
    for pat, ar in ASPECT_PATTERNS:
        if pat.search(text):
            return ar
    return None

def extract_lens(text: str) -> Optional[str]:
    m = LENS_PATTERN.search(text)
    return m.group(1) if m else None

def extract_aperture(text: str) -> Optional[str]:
    m = APERTURE_PATTERN.search(text)
    return m.group(1) if m else None

def extract_shot(text: str) -> Optional[str]:
    low = text.lower()
    for shot, keys in SHOT_KEYWORDS.items():
        if any(k in low for k in keys):
            return shot
    return None

def extract_styles(text: str) -> List[str]:
    low = text.lower()
    return [s for s in STYLE_KEYWORDS if s in low]

def extract_negative(text: str) -> List[str]:
    low = text.lower()
    neg = []
    if "negative:" in low:
        part = low.split("negative:", 1)[1]
        neg += [x.strip() for x in part.split(",") if x.strip()]
    if "--no" in low:
        part = low.split("--no", 1)[1]
        neg += [x.strip() for x in re.split(r"[,\n]", part) if x.strip()]
    for kw in ["tránh", "không có"]:
        if kw in low:
            part = low.split(kw, 1)[1]
            neg += [x.strip() for x in re.split(r"[,\n;]", part) if x.strip()]
    out = []
    for x in neg:
        if x and x not in out:
            out.append(x)
    return out

def naive_subject_environment(text: str) -> (List[str], List[str]):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    env_cues = ["background", "bối cảnh", "trên", "trong", "at ", "in ", "studio", "ánh sáng", "light"]
    subject, env = [], []
    for p in parts:
        (env if any(cue in p.lower() for cue in env_cues) else subject).append(p)
    return subject[:6], env[:6]

def parse_image(text: str) -> Dict[str, Any]:
    ar = extract_aspect_ratio(text) or "1:1"
    lens = extract_lens(text)
    ap = extract_aperture(text)
    shot = extract_shot(text)
    styles = extract_styles(text)
    negative = extract_negative(text)
    subject, environment = naive_subject_environment(text)

    composition = {}
    if shot: composition["shot"] = shot
    if lens: composition["lens"] = lens
    if ap: composition["aperture"] = ap

    obj: Dict[str, Any] = {
        "primary": text.strip(),
        "constraints": {"negative": negative or ["text", "watermark", "blurry", "low quality"]},
        "output": {"aspect_ratio": ar, "num_images": 1},
        "quality": {"detail": "high", "resolution": "4k"}
    }
    if styles: obj["style"] = styles
    if subject: obj["subject"] = subject
    if environment: obj["environment"] = environment
    if composition: obj["composition"] = composition
    return obj

def parse_marketing(text: str) -> Dict[str, Any]:
    # Baseline: assume user text contains product/benefits/platform hints; if not, keep generic.
    low = text.lower()
    platform = "tiktok" if "tiktok" in low else ("facebook" if "facebook" in low or "fb" in low else "social")
    name = "Sản phẩm"  # fallback
    # crude product name guess: take first phrase before comma
    first = text.split(",")[0].strip()
    if len(first) >= 3:
        name = first

    benefits = []
    for kw in ["healthy", "tốt cho sức khỏe", "ít đường", "giàu protein", "plant-based", "thuần thực vật", "ngon", "thơm"]:
        if kw in low:
            benefits.append(kw)

    deliverables = {
        "hook_titles": [f"{name}: nghe là muốn thử liền", f"{name}: lý do bạn nên đổi gu hôm nay"],
        "script_60s": f"HOOK: {name}...\nVẤN ĐỀ: ...\nGIẢI PHÁP: ...\nBẰNG CHỨNG: ...\nCTA: ...",
        "caption": f"{name} – mô tả ngắn gọn + lợi ích + lời kêu gọi hành động.",
        "hashtags": ["#fyp", "#viral", "#xuhuong"],
        "cta": ["Comment “INBOX” để nhận menu", "Đặt ngay hôm nay"]
    }

    obj: Dict[str, Any] = {
        "product": {
            "name": name,
            "category": "F&B" if any(k in low for k in ["latte", "cà phê", "matcha", "trà", "sữa"]) else "general",
            "key_benefits": benefits
        },
        "platform": platform,
        "objective": "conversion",
        "tone": ["ngắn gọn", "bắt trend", "thuyết phục"],
        "deliverables": deliverables,
        "compliance": {
            "avoid_claims": ["chữa bệnh", "cam kết giảm cân", "tuyên bố y khoa tuyệt đối"],
            "constraints": ["không nói quá", "không công kích đối thủ"]
        }
    }
    return obj

def parse_agent(text: str) -> Dict[str, Any]:
    # Baseline agent spec: role + goals inferred from text
    role = "AI Prompt Engineer / Automation Agent"
    goals = [
        "Chuyển yêu cầu dạng text thành JSON đúng schema",
        "Tự phát hiện thiếu thông tin và đề xuất câu hỏi bổ sung ngắn gọn",
        "Xuất kết quả nhất quán, dễ dùng cho hệ thống downstream"
    ]
    constraints = [
        "Không bịa thông tin cụ thể (giá, thành phần, thông số) nếu người dùng chưa cung cấp",
        "Luôn trả về JSON hợp lệ, đúng schema",
        "Nếu thiếu dữ liệu quan trọng: gắn cờ trong output thay vì đoán bừa"
    ]
    tools_allowed = ["schema_validator", "keyword_parser"]

    obj: Dict[str, Any] = {
        "role": role,
        "goals": goals,
        "tools_allowed": tools_allowed,
        "constraints": constraints,
        "process": {
            "steps": [
                "Chuẩn hóa text (trim, bỏ ký tự thừa).",
                "Nhận diện task type hoặc dùng task user chọn.",
                "Trích xuất entity/constraints/outputs theo rule.",
                "Validate bằng JSON Schema.",
                "Nếu fail: trả errors + gợi ý sửa."
            ],
            "ask_clarifying_if": [
                "Không có tên sản phẩm/đối tượng chính",
                "Không có nền tảng (TikTok/FB) cho marketing task",
                "Thiếu output format/acceptance criteria cho agent task"
            ]
        },
        "outputs": {
            "format": "JSON only",
            "acceptance_criteria": [
                "JSON parse được",
                "Validate pass theo schema của task",
                "Không chứa field thừa"
            ]
        },
        "evaluation": {
            "checklist": [
                "Schema valid",
                "Negative/constraints hợp lý",
                "Không suy diễn thông tin nhạy cảm"
            ]
        }
    }
    return obj

def build_envelope(text: str, lang: str, task: TaskType) -> Dict[str, Any]:
    if task == "image":
        task_obj = parse_image(text)
    elif task == "marketing":
        task_obj = parse_marketing(text)
    else:
        task_obj = parse_agent(text)

    return {
        "meta": {"version": "1.0", "lang": lang, "task": task},
        "task": task_obj
    }

def validate_envelope(obj: Dict[str, Any]) -> (bool, List[str]):
    # Validate envelope first
    try:
        validate(instance=obj, schema=ENVELOPE_SCHEMA)
        return True, []
    except ValidationError as e:
        return False, [e.message]

# -----------------------------
# 4) ROUTES
# -----------------------------
@app.post("/convert", response_model=ConvertResponse)
def convert(req: ConvertRequest):
    prompt_json = build_envelope(req.text, req.lang, req.task)
    ok, errs = validate_envelope(prompt_json)
    return ConvertResponse(prompt_json=prompt_json, valid=ok, errors=errs or None)

@app.get("/schema")
def schema():
    return {
        "envelope": ENVELOPE_SCHEMA,
        "image": IMAGE_SCHEMA,
        "marketing": MARKETING_SCHEMA,
        "agent": AGENT_SCHEMA
    }
