import re
import os
import pandas as pd
import logging
try:
    import yake
except ImportError:
    yake = None

INPUT_CSV = os.path.join(os.path.dirname(__file__), "topics.csv")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# INPUT_CSV = "topics.csv"
OUTPUT_CSV = "topics_enriched.csv"

# --- Instruction patterns (ordered by specificity) ---
INSTRUCTION_PATTERNS = [
    (r"^(share your perspective on)\s+", "share your perspective on"),
    (r"^(share your opinion on)\s+", "share your opinion on"),
    (r"^(share your view on)\s+", "share your view on"),
    (r"^(talk about)\s+", "talk about"),
    (r"^(tell me about)\s+", "tell me about"),
    (r"^(describe)\s+", "describe"),
    (r"^(discuss)\s+", "discuss"),
    (r"^(explain)\s+", "explain"),
    (r"^(compare)\s+", "compare"),
    (r"^(contrast)\s+", "contrast"),
    (r"^(give tips for)\s+", "give tips for"),
    (r"^(share tips for)\s+", "share tips for"),
    (r"^(give advice on)\s+", "give advice on"),
]

# --- Topic type heuristics ---
TYPE_RULES = [
    ("event", re.compile(r"\b(event|incident|election|vote|policy|law|bill|protest|conflict|summit|court|decision)\b", re.I)),
    ("advice", re.compile(r"\b(tips|advice|ways|strategies|how to)\b", re.I)),
    ("compare", re.compile(r"\b(compare|contrast|difference|similar)\b", re.I)),
    ("explain", re.compile(r"\b(explain|how|why|method|process)\b", re.I)),
    ("opinion", re.compile(r"\b(opinion|perspective|view|agree|disagree)\b", re.I)),
    ("experience", re.compile(r"\b(experience|time when|moment|memorable|happened)\b", re.I)),
    ("story", re.compile(r"\b(story|narrate|describe a time)\b", re.I)),
]

# --- Constraint extraction ---
CONSTRAINT_RULES = [
    ("recent", re.compile(r"\brecent\b|\brecently\b|\blast (week|month|year)\b|\bthis (week|month|year)\b", re.I)),
    ("past", re.compile(r"\bpast\b|\bchildhood\b|\bin school\b|\bin college\b|\bwhen i was\b", re.I)),
    ("school", re.compile(r"\bschool\b|\bclass\b|\bteacher\b|\bstudent\b", re.I)),
    ("work", re.compile(r"\bwork\b|\bjob\b|\boffice\b|\bmanager\b|\bcolleague\b", re.I)),
    ("personal", re.compile(r"\bmy\b|\bfriend\b|\bfamily\b|\brelationship\b", re.I)),
]

# --- Anchor rules by type ---
EXPECTED_ANCHORS_BY_TYPE = {
    "event": ["time", "place", "what_happened", "your_view"],
    "experience": ["time", "place", "who", "what_happened", "reflection"],
    "story": ["time", "place", "who", "what_happened", "result"],
    "opinion": ["position", "reason_1", "reason_2", "example"],
    "advice": ["steps", "example", "why_it_works"],
    "compare": ["item_a", "item_b", "similarities", "differences", "conclusion"],
    "explain": ["definition", "how_it_works", "example", "impact"],
    "general": ["main_point", "supporting_points", "example"],
}

# --- YAKE setup ---
def extract_yake_phrases(text: str, top_k: int = 8):
    if yake is None:
        return []
    kw = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=top_k)
    pairs = kw.extract_keywords(text)
    phrases = []
    seen = set()
    for phrase, _score in pairs:
        p = phrase.strip().lower()
        p = re.sub(r"\s+", " ", p)
        if p and p not in seen:
            seen.add(p)
            phrases.append(p)
    return phrases

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def split_instruction_and_content(topic_raw: str):
    t = normalize_spaces(topic_raw)
    low = t.lower()

    for pattern, label in INSTRUCTION_PATTERNS:
        m = re.match(pattern, low)
        if m:
            instr = label
            content = t[len(m.group(0)):].strip()
            return instr, content

    # fallback: first verb-like word
    first = low.split(" ", 1)[0]
    return first, t[len(first):].strip() if len(t.split()) > 1 else t

def classify_topic_type(text: str) -> str:
    for ttype, rx in TYPE_RULES:
        if rx.search(text):
            return ttype
    return "general"

def detect_constraints(text: str):
    out = []
    for name, rx in CONSTRAINT_RULES:
        if rx.search(text):
            out.append(name)
    # de-dupe
    return sorted(set(out))

def expected_anchors(topic_type: str):
    return EXPECTED_ANCHORS_BY_TYPE.get(topic_type, EXPECTED_ANCHORS_BY_TYPE["general"])

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Missing {INPUT_CSV} in current directory.")

    # Your file is Windows-1252 encoded in many cases; this handles smart quotes etc.
    df = pd.read_csv(INPUT_CSV, encoding="cp1252")

    if "topic" not in df.columns or "category" not in df.columns:
        raise ValueError("Expected columns: topic, category")

    rows = []
    for idx, row in df.iterrows():
        topic_raw = normalize_spaces(str(row["topic"]))
        category = normalize_spaces(str(row["category"]))

        instr, content = split_instruction_and_content(topic_raw)

        ttype = classify_topic_type(topic_raw)
        constraints = detect_constraints(topic_raw)
        anchors = expected_anchors(ttype)

        topic_kps = extract_yake_phrases(content if content else topic_raw, top_k=8)

        rows.append({
            "topic_id": idx + 1,
            "category": category,
            "topic_raw": topic_raw,
            "instruction": instr,
            "topic_content": content if content else topic_raw,
            "topic_type": ttype,
            "constraints": "|".join(constraints) if constraints else "",
            "expected_anchors": "|".join(anchors) if anchors else "",
            "topic_keyphrases": "|".join(topic_kps) if topic_kps else ""
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved enriched topics to: {OUTPUT_CSV}")
    print(f"Rows: {len(out)} | Columns: {list(out.columns)}")

if __name__ == "__main__":
    main()
