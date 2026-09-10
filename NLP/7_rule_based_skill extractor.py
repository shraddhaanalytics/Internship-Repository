import re
import spacy
import pandas as pd

df = pd.read_csv("C:/Users/HP/Desktop/Internship Final Submission/NLP/cleaned_job_dataset after EDA.csv")

SKILL_CATALOG = {
    "python": ("Python", "Programming"),
    "sql": ("SQL", "Database"),
    "java": ("Java", "Programming"),
    "scala": ("Scala", "Programming"),
    "r": ("R", "Programming"),
    "machine learning": ("Machine Learning", "Machine Learning"),
    "deep learning": ("Deep Learning", "Machine Learning"),
    "natural language processing": ("Natural Language Processing", "Machine Learning"),
    "computer vision": ("Computer Vision", "Machine Learning"),
    "statistics": ("Statistics", "Analytics"),
    "data analysis": ("Data Analysis", "Analytics"),
    "big data": ("Big Data", "Analytics"),
    "aws": ("AWS", "Cloud"),
    "azure": ("Azure", "Cloud"),
    "google cloud": ("Google Cloud", "Cloud"),
    "postgresql": ("PostgreSQL", "Database"),
    "postgres": ("PostgreSQL", "Database"),
    "mongodb": ("MongoDB", "Database"),
    "mysql": ("MySQL", "Database"),
    "sql server": ("SQL Server", "Database"),
    "sqlserver": ("SQL Server", "Database"),
    "power bi": ("Power BI", "Analytics"),
    "powerbi": ("Power BI", "Analytics"),
    "power-bi": ("Power BI", "Analytics"),
    "tableau": ("Tableau", "Analytics"),
    "excel": ("Excel", "Analytics"),
    "git": ("Git", "DevOps"),
    "docker": ("Docker", "DevOps"),
    "kubernetes": ("Kubernetes", "DevOps"),
    "tensorflow": ("TensorFlow", "Machine Learning"),
    "tensor flow": ("TensorFlow", "Machine Learning"),
    "pytorch": ("PyTorch", "Machine Learning"),
    "scikit-learn": ("Scikit-learn", "Machine Learning"),
    "scikit learn": ("Scikit-learn", "Machine Learning"),
    "xgboost": ("XGBoost", "Machine Learning"),
    "spark": ("Spark", "Big Data Tool"),
    "hadoop": ("Hadoop", "Big Data Tool"),
    "kafka": ("Kafka", "Big Data Tool"),
    "airflow": ("Airflow", "Big Data Tool"),
    "communication": ("Communication", "Soft Skill"),
    "leadership": ("Leadership", "Soft Skill"),
}
 


def normalize_skill(raw_term):
    term = raw_term.lower().strip()
    term = re.sub(r"[^a-z0-9\s\+\#\-]", "", term)
    term = re.sub(r"\s+", " ", term).strip()
    stripped = re.sub(r"\s*\d+(\.\d+)*$", "", term).strip()
    for candidate in (term, stripped):
        if candidate in SKILL_CATALOG:
            return SKILL_CATALOG[candidate]
    return None

REGEX_PATTERNS = {
    "python": r"\bpython\s*3?(\.\d+)?\b|\bpython programming\b",
    "sql": r"\bsql\b(?!\s*server)",
    "sql server": r"\bsql\s*server\b",
    "java": r"\bjava\b(?!script)",
    "scala": r"\bscala\b",
    "aws": r"\baws\b",
    "azure": r"\bazure\b",
    "google cloud": r"\bgoogle\s+cloud\b",
    "machine learning": r"\bmachine[\s-]learning\b|\bml\b",
    "deep learning": r"\bdeep[\s-]learning\b",
    "natural language processing": r"\bnatural\s+language\s+processing\b|\bnlp\b",
    "computer vision": r"\bcomputer\s+vision\b",
    "statistics": r"\bstatistics\b",
    "data analysis": r"\bdata\s+analysis\b",
    "big data": r"\bbig\s+data\b",
    "postgresql": r"\bpostgres(?:ql)?\b",
    "mongodb": r"\bmongodb\b",
    "mysql": r"\bmysql\b",
    "power bi": r"\bpower[\s-]?bi\b",
    "tableau": r"\btableau\b",
    "excel": r"\bexcel\b",
    "git": r"\bgit\b(?!hub)",
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b",
    "tensorflow": r"\btensor\s*flow\b",
    "pytorch": r"\bpytorch\b",
    "scikit-learn": r"\bscikit[\s-]learn\b",
    "xgboost": r"\bxgboost\b",
    "spark": r"\bspark\b",
    "hadoop": r"\bhadoop\b",
    "kafka": r"\bkafka\b",
    "airflow": r"\bairflow\b",
    "communication": r"\bcommunication\b",
    "leadership": r"\bleadership\b",
}

def extract_regex(text):
    lower = text.lower()
    found = set()
    for key, pattern in REGEX_PATTERNS.items():
        if re.search(pattern, lower):
            norm = normalize_skill(key)
            if norm:
                found.add(norm)
    return found

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
        ruler = _nlp.add_pipe("entity_ruler", before="ner")
        patterns = [
            {"label": "SKILL", "pattern": "Python"}, {"label": "SKILL", "pattern": "SQL"},
            {"label": "SKILL", "pattern": "Java"}, {"label": "SKILL", "pattern": "Scala"},
            {"label": "SKILL", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "deep"}, {"LOWER": "learning"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "natural"}, {"LOWER": "language"}, {"LOWER": "processing"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "computer"}, {"LOWER": "vision"}]},
            {"label": "SKILL", "pattern": [{"LOWER": "big"}, {"LOWER": "data"}]},
            {"label": "SKILL", "pattern": "Statistics"},
            {"label": "SKILL", "pattern": [{"LOWER": "data"}, {"LOWER": "analysis"}]},
            {"label": "CLOUD_PLATFORM", "pattern": "AWS"}, {"label": "CLOUD_PLATFORM", "pattern": "Azure"},
            {"label": "CLOUD_PLATFORM", "pattern": [{"LOWER": "google"}, {"LOWER": "cloud"}]},
            {"label": "DATABASE", "pattern": "PostgreSQL"}, {"label": "DATABASE", "pattern": "MongoDB"},
            {"label": "DATABASE", "pattern": "MySQL"},
            {"label": "DATABASE", "pattern": [{"LOWER": "sql"}, {"LOWER": "server"}]},
            {"label": "TOOL", "pattern": [{"LOWER": "power"}, {"LOWER": "bi"}]}, {"label": "TOOL", "pattern": "Tableau"},
            {"label": "TOOL", "pattern": "Excel"}, {"label": "TOOL", "pattern": "Git"}, {"label": "TOOL", "pattern": "Docker"},
            {"label": "TOOL", "pattern": "Kubernetes"},
            {"label": "TECHNOLOGY", "pattern": "TensorFlow"}, {"label": "TECHNOLOGY", "pattern": "PyTorch"},
            {"label": "TECHNOLOGY", "pattern": "scikit-learn"}, {"label": "TECHNOLOGY", "pattern": "XGBoost"},
            {"label": "TECHNOLOGY", "pattern": "Spark"}, {"label": "TECHNOLOGY", "pattern": "Hadoop"},
            {"label": "TECHNOLOGY", "pattern": "Kafka"}, {"label": "TECHNOLOGY", "pattern": "Airflow"},
        ]
        ruler.add_patterns(patterns)
    return _nlp

def extract_ner(text):
    nlp = get_nlp()
    doc = nlp(text)
    found = set()
    for ent in doc.ents:
        if ent.label_ in {"SKILL", "TOOL", "TECHNOLOGY", "DATABASE", "CLOUD_PLATFORM"}:
            norm = normalize_skill(ent.text)
            if norm:
                found.add(norm)
    return found

def extract_skills(text):
    combined = extract_regex(text) | extract_ner(text)
    return sorted(combined, key=lambda x: x[0])