# ==========================================
# Course Enrollment Guidance Chatbot (Streamlit)
# + Intent Train/Test Metrics
# + Course Eligibility (Rule-based prereqs)
# + Recommendations (Weighted + optional Cosine Similarity)
# + Voice Input (mic) + Voice Reply (TTS)
# ==========================================

import re
import tempfile
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# OPTIONAL VOICE IMPORTS
# =========================
MIC_AVAILABLE = True
SR_AVAILABLE = True
PYDUB_AVAILABLE = True
GTTS_AVAILABLE = True

try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    MIC_AVAILABLE = False

try:
    import speech_recognition as sr
except Exception:
    SR_AVAILABLE = False

try:
    from pydub import AudioSegment
except Exception:
    PYDUB_AVAILABLE = False

try:
    from gtts import gTTS
except Exception:
    GTTS_AVAILABLE = False

# =========================
# ML IMPORTS (scikit-learn)
# =========================
SKLEARN_AVAILABLE = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    SKLEARN_AVAILABLE = False

# =========================
# PAGE CONFIG + CSS
# =========================
st.set_page_config(
    page_title="Course Enrollment Guidance Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        text-align: center;
        color: white;
    }
    .header-container h1 { margin: 0; font-size: 2.2rem; }
    .header-container p { margin: 0.35rem 0 0 0; opacity: 0.9; }

    .chat-bubble {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin: 0.4rem 0;
        border: 1px solid rgba(0,0,0,0.08);
        white-space: pre-wrap;
        line-height: 1.35;
    }
    .user-bubble { background: #1e3c72; color: white; margin-left: auto; width: fit-content; max-width: 75%; }
    .bot-bubble  { background: #eef6ff; color: #111; width: fit-content; max-width: 75%; border: 1px solid #2a5298; }

    .metric-card {
        background: #f6f7fb;
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #2a5298;
        color: #111 !important;
    }
    .metric-card * { color: #111 !important; }

    .small-note { font-size: 0.9rem; opacity: 0.9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# DATASET PATHS
# =========================
AI_COURSES_CSV = "ai_courses_expanded.csv"
CAREER_MAP_CSV = "career_mapping_expanded.csv"
INTENT_CSV = "intent_response_expanded.csv"
STUDENTS_CSV = "student_profiles_expanded.csv"

TOP_K_RECOMMEND = 5

# =========================
# HELPERS
# =========================
COURSE_ID_REGEX = re.compile(r"\b[A-Z]{2,5}\d{3}\b")


def extract_course_ids(text: str):
    return list(set(COURSE_ID_REGEX.findall((text or "").upper())))


def normalize_list_field(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in ("none", "nan"):
        return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]


def safe_lower(x):
    return str(x).strip().lower()


# =========================
# VOICE HELPERS
# =========================
def speech_to_text_from_mic_bytes(audio_bytes: bytes) -> str:
    """
    Convert mic bytes (often webm/ogg) -> WAV PCM -> SpeechRecognition (Google).
    """
    if not SR_AVAILABLE:
        return "__SR_NOT_INSTALLED__"
    if not PYDUB_AVAILABLE:
        return "__PYDUB_NOT_INSTALLED__"

    r = sr.Recognizer()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as in_f:
        in_f.write(audio_bytes)
        in_path = in_f.name

    wav_path = in_path.replace(".webm", ".wav")

    try:
        sound = AudioSegment.from_file(in_path)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")

        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)

        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return "__SR_ERROR__"
    except Exception as e:
        return f"__CONVERT_ERROR__: {e}"


def tts_audio_gtts_bytes(text: str) -> bytes:
    """
    Generate MP3 bytes using gTTS (internet required).
    """
    if not GTTS_AVAILABLE:
        return b""
    try:
        tts = gTTS(text=text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            with open(fp.name, "rb") as f:
                return f.read()
    except Exception:
        return b""


def tts_speak_pyttsx3(text: str):
    """
    Offline system voice (can be flaky inside Streamlit).
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass


# =========================
# COURSE CATALOG + PREREQ GRAPH
# =========================
class CourseCatalog:
    def __init__(self, courses_df: pd.DataFrame):
        self.df = courses_df.copy()

        self.df["course_id"] = self.df["course_id"].astype(str).str.strip().str.upper()
        self.df["course_name"] = self.df["course_name"].astype(str).str.strip()
        self.df["difficulty_level"] = self.df["difficulty_level"].astype(str).str.strip()
        self.df["course_type"] = self.df["course_type"].astype(str).str.strip()
        self.df["enrollment_status"] = self.df["enrollment_status"].astype(str).str.strip()

        self.df["prereq_list"] = self.df["prerequisites"].apply(normalize_list_field)
        self.df["career_list"] = self.df["career_relevance"].apply(normalize_list_field)

        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)

        for _, row in self.df.iterrows():
            cid = row["course_id"]
            prereqs = row["prereq_list"]
            prereq_course_ids = [p.upper() for p in prereqs if COURSE_ID_REGEX.fullmatch(p.upper())]
            for pre in prereq_course_ids:
                self.graph[pre].append(cid)
                self.in_degree[cid] += 1
            if cid not in self.in_degree:
                self.in_degree[cid] = self.in_degree.get(cid, 0)

        self.course_map = {row["course_id"]: row for _, row in self.df.iterrows()}

    def exists(self, course_id: str) -> bool:
        return course_id.upper() in self.course_map

    def get(self, course_id: str):
        return self.course_map[course_id.upper()]

    def topo_sort(self):
        indeg = dict(self.in_degree)
        q = deque([c for c, d in indeg.items() if d == 0])
        order = []
        while q:
            node = q.popleft()
            order.append(node)
            for nxt in self.graph.get(node, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    q.append(nxt)
        return order

    def missing_prereq_path(self, target: str, completed: set):
        target = target.upper()
        if not self.exists(target):
            return []
        visited = set()
        missing = set()

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            prereqs = self.get(node)["prereq_list"]
            prereq_course_ids = [p.upper() for p in prereqs if COURSE_ID_REGEX.fullmatch(p.upper())]
            for pre in prereq_course_ids:
                if pre not in completed:
                    missing.add(pre)
                dfs(pre)

        dfs(target)
        return sorted(list(missing))


# =========================
# STUDENT PROFILE
# =========================
class Student:
    def __init__(self, row: dict):
        self.student_id = str(row.get("student_id", "")).strip()
        self.education_level = str(row.get("education_level", "")).strip()
        self.programming_experience = str(row.get("programming_experience", "")).strip()
        self.math_background = str(row.get("math_background", "")).strip()
        self.ai_experience = str(row.get("ai_experience", "")).strip()
        self.career_goal = str(row.get("career_goal", "")).strip()
        self.preferred_learning_mode = str(row.get("preferred_learning_mode", "")).strip()
        self.available_hours_per_week = str(row.get("available_hours_per_week", "")).strip()

        self.completed_courses = set()
        self.selected_next_semester = set()

    def inferred_skills(self):
        skills = set()

        pe = safe_lower(self.programming_experience)
        if pe in ("beginner", "intermediate", "advanced"):
            skills.add("python")
            if pe in ("intermediate", "advanced"):
                skills.add("basic programming")

        mb = safe_lower(self.math_background)
        if mb in ("average", "strong"):
            skills.add("basic math")
        if mb == "strong":
            skills.add("linear algebra")
            skills.add("probability")
            skills.add("statistics")

        ae = safe_lower(self.ai_experience)
        if ae in ("basic", "intermediate", "advanced"):
            skills.add("ai fundamentals")
        if ae in ("intermediate", "advanced"):
                skills.add("machine learning")

        return skills


def prerequisites_satisfied(course_row, student: Student):
    prereqs = course_row["prereq_list"]
    if not prereqs:
        return True, [], [], []

    completed = set([c.upper() for c in student.completed_courses])
    coreq_pool = completed.union(set([c.upper() for c in student.selected_next_semester]))
    skills = set([s.lower() for s in student.inferred_skills()])

    missing_courses = []
    missing_skills = []
    known_skills = []

    for p in prereqs:
        p_up = p.upper()
        if COURSE_ID_REGEX.fullmatch(p_up):
            if p_up not in coreq_pool:
                missing_courses.append(p_up)
        else:
            if p.lower() not in skills:
                missing_skills.append(p)
            else:
                known_skills.append(p)

    ok = len(missing_courses) == 0 and len(missing_skills) == 0
    return ok, missing_courses, missing_skills, known_skills



def eligible_courses(catalog: CourseCatalog, student: Student):
    eligible = []
    blocked = []

    for _, row in catalog.df.iterrows():
        cid = row["course_id"]
        if cid in student.completed_courses:
            continue
        if safe_lower(row["enrollment_status"]) != "open":
            continue

        ok, missing_courses, missing_skills, _ = prerequisites_satisfied(row, student)
        if ok:
            eligible.append(row)
        else:
            blocked.append((row, missing_courses, missing_skills))  # store separately

    return eligible, blocked


# =========================
# RECOMMENDER
# =========================
def build_course_text(row):
    parts = [
        str(row.get("course_name", "")),
        str(row.get("course_type", "")),
        str(row.get("difficulty_level", "")),
        " ".join([c for c in row.get("career_list", [])]),
        " ".join(normalize_list_field(row.get("learning_outcomes", ""))),
        " ".join(normalize_list_field(row.get("prerequisites", ""))),
    ]
    return " ".join([p for p in parts if p and str(p).lower() != "nan"])


def unlocks_more_score(catalog: CourseCatalog, course_id: str):
    return len(catalog.graph.get(course_id.upper(), []))


def weighted_score(catalog: CourseCatalog, row, student: Student, career_course_set: set):
    w1, w2, w3, w4 = 3.0, 1.0, 1.0, 0.5
    cid = row["course_id"].upper()

    career_match = 1.0 if cid in career_course_set else 0.0
    unlocks = unlocks_more_score(catalog, cid)

    diff = safe_lower(row["difficulty_level"])
    pe = safe_lower(student.programming_experience)
    if pe == "beginner":
        difficulty_fit = 1.0 if diff in ("beginner",) else 0.2
    elif pe == "intermediate":
        difficulty_fit = 1.0 if diff in ("beginner", "intermediate") else 0.5
    else:
        difficulty_fit = 1.0

    mode_fit = 0.3
    if safe_lower(student.preferred_learning_mode) in ("online", "hybrid"):
        mode_fit = 1.0

    return (w1 * career_match) + (w2 * unlocks) + (w3 * difficulty_fit) + (w4 * mode_fit)


def recommend_courses(catalog: CourseCatalog, student: Student, career_map_df: pd.DataFrame, use_cosine=True, top_k=TOP_K_RECOMMEND):
    eligible, blocked = eligible_courses(catalog, student)

    career_goal = student.career_goal.strip()
    career_course_set = set()
    if career_goal:
        row = career_map_df[
            career_map_df["career_role"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "")
            == career_goal.lower().replace(" ", "")
        ]
        if len(row) > 0:
            suggested = normalize_list_field(row.iloc[0]["suggested_courses"])
            career_course_set = set([s.upper() for s in suggested])

    base_scores = np.array([weighted_score(catalog, r, student, career_course_set) for r in eligible], dtype=float) if eligible else np.array([], dtype=float)
    final_scores = base_scores.copy()

    if use_cosine and len(eligible) > 0 and SKLEARN_AVAILABLE:
        profile_text = f"{student.education_level} {student.career_goal} {student.programming_experience} {student.math_background} {student.ai_experience}"
        corpus = [profile_text] + [build_course_text(r) for r in eligible]
        vec = TfidfVectorizer()
        X = vec.fit_transform(corpus)
        sim = cosine_similarity(X[0:1], X[1:]).flatten()
        final_scores = final_scores + (1.5 * sim)

    ranked = sorted(list(zip(eligible, final_scores)), key=lambda x: x[1], reverse=True)
    return ranked[:top_k], blocked


# =========================
# INTENT MODEL
# =========================
class IntentModel:
    def __init__(self):
        self.pipeline = None
        self.label_list = None

    def train_and_eval(self, intent_df: pd.DataFrame, test_size=0.2, random_state=42):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is not installed. Install it to train/test the model.")

        df = intent_df.copy()
        df["intent"] = (
            df["intent"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            )
        df["user_query_example"] = df["user_query_example"].astype(str)

        X = df["user_query_example"].tolist()
        y = df["intent"].tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(set(y)) > 1 else None
        )

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300)),
        ])
        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

        labels = sorted(list(set(y)))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        report = classification_report(y_test, y_pred, zero_division=0)

        self.label_list = labels

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "labels": labels,
            "report": report,
            "test_size": test_size,
        }

    def predict(self, text: str) -> str:
        if self.pipeline is None:
            return "unknown"
        t = (text or "").strip()
        if not t:
            return "unknown"
        return (
            self.pipeline.predict([t])[0]
            .strip()
            .lower()
            .replace(" ", "_")
        )
# =========================
# CHATBOT CORE
# =========================
class CourseChatbot:
    def __init__(self, catalog: CourseCatalog, career_map_df: pd.DataFrame, intent_model: IntentModel, intent_df: pd.DataFrame):
        self.catalog = catalog
        self.career_map_df = career_map_df
        self.intent_model = intent_model

        self.response_map = defaultdict(list)
        tmp = intent_df.copy()
        tmp["intent"] = tmp["intent"].astype(str).str.strip()
        for _, r in tmp.iterrows():
            self.response_map[r["intent"]].append(str(r.get("bot_response", "")).strip())
        
    def chat(self, student: Student, text: str, use_cosine=True, top_k=TOP_K_RECOMMEND):
        t = (text or "").strip()
        if not t:
            return "I didn't catch that. Try again."
        if t.lower() in ("exit", "quit"):
            return "__EXIT__"

        course_ids = extract_course_ids(t)
        intent = self.intent_model.predict(t)
        if re.search(r"\b(remaining|requirements|missing|need)\b", t, re.I):
            if course_ids:
                return self.handle_check_prereq(student, t)
            return self.handle_remaining_requirements(student)


        if intent == "course_difficulty" or re.search(r"\b(hard|difficulty|difficult|challenge|tough|struggle|beginner|beginner_friendly|suitable)\b", t, re.I):
          return self.handle_course_difficulty(student, t)

        if course_ids and (
            "help" in t.lower() or
            "prereq" in intent or
            "prerequisite" in t.lower() or
            "can i take" in t.lower()
        ):
            return self.handle_check_prereq(student, t)
        
        if re.search(r"\b(online|hybrid|remote|study mode|distance learning)\b", t, re.I) or intent == "learning_mode":
            return self.handle_learning_mode(student, t)
        
        if "recommend" in t.lower() or "career" in t.lower() or intent == "recommend_courses":
            return self.handle_recommend(student, use_cosine=use_cosine, top_k=top_k)
        

        if "plan" in t.lower() or "topological" in t.lower() or "semester order" in t.lower():
            return self.handle_plan()

        return self.canned_response(intent)

    def canned_response(self, intent: str) -> str:
        candidates = self.response_map.get(intent, [])
        if not candidates:
            return "I can help with course recommendations, prerequisites, and planning. Try: 'Recommend courses' or 'Can I take AI101?'"
        return candidates[0]

    def handle_check_prereq(self, student: Student, text: str):
        ids = extract_course_ids(text)
        if not ids:
            return "Tell me the course code (example: 'Can I take AI101?')."
        cid = ids[0]
        if not self.catalog.exists(cid):
            return f"I couldn't find {cid} in the catalog."
        course = self.catalog.get(cid)
        ok, missing_courses, missing_skills, known_skills = prerequisites_satisfied(course, student)
        if ok:
            return f"✅ You are eligible for {cid} - {course['course_name']} (prereqs satisfied)."
        msg = f"❌ Not eligible for {cid} - {course['course_name']}\n"
        if missing_courses:
            msg += "Courses missing: " + ", ".join(missing_courses) + "\n"
        if missing_skills:
            msg += "Skills/knowledge missing: " + ", ".join(missing_skills) + "\n"
        if known_skills:
            msg += "Already known skills: " + ", ".join(known_skills) + "\n"
        missing_path = self.catalog.missing_prereq_path(cid, set([c.upper() for c in student.completed_courses]))
        if missing_path:
            msg += "Missing dependency path (DFS): " + ", ".join(missing_path)
        return msg
        
    def handle_course_difficulty(self, student: Student, user_text: str):
        course_ids = extract_course_ids(user_text)
        if course_ids:
            cid = course_ids[0]
        else:
            user_text_lower = user_text.lower()
            matched_courses = []
            for _, row in self.catalog.df.iterrows():
                course_name = row["course_name"].lower()
                words = [w for w in course_name.split() if len(w) > 2]  # skip tiny words like 'of', 'in'
                if all(w in user_text_lower for w in words):
                    matched_courses.append(row["course_id"])

            if matched_courses:
                cid = matched_courses[0]
            else:
                return "Please specify a course code or course name (e.g., ML201 or Machine Learning)."
        if not self.catalog.exists(cid):
            return f"I couldn't find {cid} in the catalog."
        course = self.catalog.get(cid)
        ok, missing_courses, missing_skills, known_skills = prerequisites_satisfied(course, student)
        
        msg = f"📘 {cid} – {course['course_name']}\n"
        msg += f"Difficulty: {course['difficulty_level']}\n\n"
        if ok:
            msg += "✅ You meet all prerequisites. This course should be manageable."
        else:
            msg += "⚠️ Warning: You may face challenges in this course.\n"
            if missing_courses:
                msg += "Missing courses: " + ", ".join(missing_courses) + "\n"
            if missing_skills:
                msg += "Missing skills/knowledge: " + ", ".join(missing_skills) + "\n"
            if known_skills:
                msg += "Already known skills: " + ", ".join(known_skills) + "\n"
            msg += "Recommendation: Review missing courses/skills before enrolling."
        return msg
    
    def handle_remaining_requirements(self, student: Student):
        if not student.selected_next_semester:
            return "You have not selected any next-semester courses to check requirements for."
        messages = []
        for cid in student.selected_next_semester:
            if not self.catalog.exists(cid):
                messages.append(f"{cid} not found in catalog.")
                continue
            course = self.catalog.get(cid)
            ok, missing_courses, missing_skills, known_skills = prerequisites_satisfied(course, student)
            msg = f"📌 {cid} – {course['course_name']}\n"
            if ok:
                msg += "✅ All prerequisites satisfied."
            else:
                if missing_courses:
                    msg += "Missing courses: " + ", ".join(missing_courses) + "\n"
                if missing_skills:
                    msg += "Missing skills/knowledge: " + ", ".join(missing_skills) + "\n"
            messages.append(msg)
        return "\n\n".join(messages)



    def handle_recommend(self, student: Student, use_cosine=True, top_k=TOP_K_RECOMMEND):
        ranked, blocked = recommend_courses(self.catalog, student, self.career_map_df, use_cosine=use_cosine, top_k=top_k)
        if not ranked:
            msg = "No eligible open courses found right now."
            if blocked:
                msg += "\nSome courses are blocked due to missing prerequisites."
            return msg

        lines = ["✅ Top course recommendations (ranked):"]
        for i, (row, score) in enumerate(ranked, start=1):
            cid = row["course_id"]
            nm = row["course_name"]
            diff = row["difficulty_level"]
            ctype = row["course_type"]
            careers = ", ".join(row["career_list"]) if row["career_list"] else "General"
            lines.append(f"{i}) {cid} - {nm} | {diff} | {ctype} | Careers: {careers} | Score={score:.2f}")

        if blocked:
            lines.append("\n⚠️ Blocked (not eligible) examples:")
            for row, missing_courses, missing_skills in blocked[:3]:
                line = f"- {row['course_id']} missing: "
                parts = []
                if missing_courses:
                    parts.append(", ".join(missing_courses))
                if missing_skills:
                    parts.append(", ".join(missing_skills))
                line += ", ".join(parts)
                lines.append(line)


        return "\n".join(lines)

    def handle_plan(self):
        order = self.catalog.topo_sort()
        if not order:
            return "I couldn't generate a plan order (graph might be empty or cyclic)."
        return "📌 Valid course order (Topological Sort):\n" + " -> ".join(order)
    
    def handle_learning_mode(self, student: Student, user_text: str):
        return (
            "Yes, many AI courses are available online and in hybrid formats.\n"
            "Start with Python and AI fundamentals online, then move to Machine Learning and Deep Learning courses."
       )



# =========================
# DATA LOADING
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    courses_df = pd.read_csv(AI_COURSES_CSV)
    career_map_df = pd.read_csv(CAREER_MAP_CSV)
    intent_df = pd.read_csv(INTENT_CSV)
    students_df = pd.read_csv(STUDENTS_CSV)
    return courses_df, career_map_df, intent_df, students_df

REQUIRED_COLUMNS = {
    "ai_courses_expanded.csv": ["course_id", "course_name", "difficulty_level", "prerequisites"],
    "intent_response_expanded.csv": ["intent", "user_query_example", "bot_response"],
    "career_mapping_expanded.csv": ["career_role", "suggested_courses"],
    "student_profiles_expanded.csv": ["student_id", "career_goal"]
}

def validate_csv(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{name} missing columns: {missing}")
        st.stop()
        validate_csv(intent_df, REQUIRED_COLUMNS["intent_response_expanded.csv"], "Intent CSV")

# =========================
# SESSION STATE INIT
# =========================
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "intent_model" not in st.session_state:
    st.session_state.intent_model = IntentModel()
if "eval_metrics" not in st.session_state:
    st.session_state.eval_metrics = None
if "student" not in st.session_state:
    st.session_state.student = None
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "voice_autosend" not in st.session_state:
    st.session_state.voice_autosend = True
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

# ✅ TTS storage (this is what fixes “no player shown”)
if "tts_audio_bytes" not in st.session_state:
    st.session_state.tts_audio_bytes = None
if "tts_audio_mime" not in st.session_state:
    st.session_state.tts_audio_mime = "audio/mp3"
if "tts_for_bot_index" not in st.session_state:
    st.session_state.tts_for_bot_index = None  # which bot message index should show audio


# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="header-container">
        <h1>🎓 Course Enrollment Guidance Chatbot</h1>
        <p>Intent Classification + Prereq Eligibility + Course Recommendations + Voice</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# EARLY WARNING IF SKLEARN MISSING
# =========================
if not SKLEARN_AVAILABLE:
    st.error(
        "❌ scikit-learn (sklearn) is not installed.\n\n"
        "Install it, then rerun:\n"
        "```bash\npython -m pip install -U scikit-learn\n```"
    )
    st.stop()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    test_size = st.slider("Test size (intent model evaluation)", 0.1, 0.4, 0.2, 0.05)
    use_cosine = st.checkbox("Use cosine similarity boost (recommendations)", value=True)
    top_k = st.slider("Top-K recommendations", 3, 10, TOP_K_RECOMMEND, 1)

    st.markdown("---")
    st.markdown("### 🎙 Voice Settings")

    VOICE_INPUT = st.checkbox("Enable Voice Input", value=True)
    VOICE_REPLY = st.checkbox("Enable Voice Reply", value=True)  # default ON so you can test
    st.session_state.voice_autosend = st.checkbox("Auto-send after voice input", value=True)

    tts_mode = st.selectbox(
        "Voice Reply Mode",
        ["Browser Audio (gTTS - needs internet)", "System Voice (pyttsx3 - offline)"],
        index=0,
    )

    if VOICE_INPUT:
        if not MIC_AVAILABLE:
            st.warning("Install mic recorder: `pip install streamlit-mic-recorder`")
        if not SR_AVAILABLE:
            st.warning("Install SpeechRecognition: `pip install SpeechRecognition`")
        if not PYDUB_AVAILABLE:
            st.warning("Install pydub: `pip install pydub` (ffmpeg also required)")

    if VOICE_REPLY and tts_mode.startswith("Browser") and not GTTS_AVAILABLE:
        st.warning("Install gTTS: `pip install gTTS` (and needs internet)")

    st.markdown("---")

    if st.button("📥 Load Datasets + Train/Test", use_container_width=True, type="primary"):
        st.cache_data.clear()  # Add this at the top of your app before loading CSVs
        courses_df, career_map_df, intent_df, students_df = load_data()

        metrics = st.session_state.intent_model.train_and_eval(intent_df, test_size=test_size)
        st.session_state.eval_metrics = metrics

        catalog = CourseCatalog(courses_df)
        st.session_state.catalog = catalog
        st.session_state.career_map_df = career_map_df
        st.session_state.intent_df = intent_df
        st.session_state.students_df = students_df
        st.session_state.chatbot = CourseChatbot(catalog, career_map_df, st.session_state.intent_model, intent_df)

        st.session_state.initialized = True
        st.success("Initialized! Datasets loaded + model trained/tested.")

    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.voice_text = ""
        st.session_state.last_audio_id = None
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_for_bot_index = None
        st.success("Chat cleared.")

    st.markdown("---")
    st.markdown("### 👤 Student Setup")

    if st.session_state.initialized:
        students_df = st.session_state.students_df
        student_ids = students_df["student_id"].astype(str).tolist()
        selected_sid = st.selectbox("Select student_id", student_ids)

        completed_in = st.text_input("Completed courses (comma separated)", placeholder="AI101, ML201")
        selected_next = st.text_input("Co-req courses (next semester, comma separated)", placeholder="DSA201, AI102")

        row = students_df[students_df["student_id"].astype(str) == str(selected_sid)]
        student = Student(row.iloc[0].to_dict())

        def parse_courses(s):
            if not s.strip():
                return set()
            return set([x.strip().upper() for x in s.split(",") if x.strip()])

        student.completed_courses = parse_courses(completed_in)
        student.selected_next_semester = parse_courses(selected_next)
        st.session_state.student = student

        st.markdown(
            f"<div class='small-note'>Skills inferred: {', '.join(sorted(student.inferred_skills())) or 'None'}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Initialize first to select a student profile.")

# =========================
# MAIN: MODEL METRICS
# =========================
if st.session_state.eval_metrics is not None:
    m = st.session_state.eval_metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><b>Accuracy</b><div style='font-size:1.5rem'>{m['accuracy']:.3f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><b>Precision (weighted)</b><div style='font-size:1.5rem'>{m['precision']:.3f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><b>Recall (weighted)</b><div style='font-size:1.5rem'>{m['recall']:.3f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><b>F1 (weighted)</b><div style='font-size:1.5rem'>{m['f1']:.3f}</div></div>", unsafe_allow_html=True)

    with st.expander("📊 Confusion Matrix + Classification Report", expanded=False):
        st.write("Confusion Matrix (labels sorted):")
        st.dataframe(pd.DataFrame(m["confusion_matrix"], index=m["labels"], columns=m["labels"]))
        st.text("Classification Report:")
        st.code(m["report"])

st.markdown("---")

# =========================
# MAIN: CHAT UI
# =========================
if not st.session_state.initialized:
    st.info("Click **Load Datasets + Train/Test** in the sidebar to start.")
    st.stop()

if st.session_state.student is None:
    st.warning("Select a student profile in the sidebar.")
    st.stop()


def do_send(text: str):
    text = (text or "").strip()
    if not text:
        return

    st.session_state.messages.append({"role": "user", "content": text})

    bot: CourseChatbot = st.session_state.chatbot
    student: Student = st.session_state.student

    reply = bot.chat(student, text, use_cosine=use_cosine, top_k=top_k)
    if reply == "__EXIT__":
        reply = "Chat ended safely."

    st.session_state.messages.append({"role": "bot", "content": reply})

    # ✅ Prepare TTS bytes (STORE ONLY; render later so player always shows)
    if VOICE_REPLY and reply:
        if tts_mode.startswith("Browser"):
            audio_bytes = tts_audio_gtts_bytes(reply)
            if audio_bytes:
                st.session_state.tts_audio_bytes = audio_bytes
                st.session_state.tts_audio_mime = "audio/mp3"
                # store index of the bot message that should show the player
                st.session_state.tts_for_bot_index = len(st.session_state.messages) - 1
            else:
                st.session_state.tts_audio_bytes = None
                st.session_state.tts_for_bot_index = None
                st.warning("TTS failed. Ensure internet is ON and gTTS works.")
        else:
            # system voice (may not work reliably in Streamlit)
            tts_speak_pyttsx3(reply)


# Show chat (and show audio player directly under the last bot reply)
st.markdown("## 💬 Chat")

for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

        # ✅ show audio player exactly under the bot message that created it
        if (
            VOICE_REPLY
            and tts_mode.startswith("Browser")
            and st.session_state.tts_for_bot_index == i
            and st.session_state.tts_audio_bytes
        ):
            # Use a stable key so Streamlit renders the player properly
            st.audio(
                st.session_state.tts_audio_bytes,
                format=st.session_state.tts_audio_mime,
                start_time=0,
            )

st.markdown("---")
st.markdown("### Ask a question")

# Typed input
user_text = st.text_input(
    "Type your question (or use microphone):",
    value=st.session_state.voice_text,
    placeholder="e.g., Recommend courses OR Can I take AI101? OR Give me plan order",
    key="typed_input",
)

# Mic input — process only NEW audio using audio["id"]
if VOICE_INPUT and MIC_AVAILABLE and SR_AVAILABLE:
    audio = mic_recorder(
        start_prompt="🎙 Start Recording",
        stop_prompt="⏹ Stop",
        just_once=True,
        key="mic",
    )

    if audio and isinstance(audio, dict) and "bytes" in audio:
        audio_id = audio.get("id", None)

        if audio_id is not None and audio_id != st.session_state.last_audio_id:
            st.session_state.last_audio_id = audio_id

            recognized = speech_to_text_from_mic_bytes(audio["bytes"])

            if recognized == "__SR_NOT_INSTALLED__":
                st.error("SpeechRecognition not installed. Run: `pip install SpeechRecognition`")
            elif recognized == "__PYDUB_NOT_INSTALLED__":
                st.error("pydub not installed. Run: `pip install pydub` (ffmpeg also required)")
            elif recognized == "__SR_ERROR__":
                st.error("Speech recognition service error. Check internet and try again.")
            elif isinstance(recognized, str) and recognized.startswith("__CONVERT_ERROR__"):
                st.error(recognized + "\n\nMake sure ffmpeg is installed and on PATH.")
            elif not str(recognized).strip():
                st.warning("Could not recognize speech. Try again.")
            else:
                st.session_state.voice_text = recognized.strip()
                st.success(f"Recognized: {st.session_state.voice_text}")

                if st.session_state.voice_autosend:
                    do_send(st.session_state.voice_text)
                    st.rerun()

colA, colB = st.columns([1, 1])

with colA:
    if st.button("Send", type="primary", use_container_width=True):
        do_send(user_text)
        st.session_state.voice_text = ""  # optional clear
        st.rerun()

with colB:
    if st.button("Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.voice_text = ""
        st.session_state.last_audio_id = None
        st.session_state.tts_audio_bytes = None
        st.session_state.tts_for_bot_index = None
        st.rerun()

st.markdown("### Example Questions")
ex1, ex2, ex3 = st.columns(3)
with ex1:
    if st.button("Recommend courses", use_container_width=True):
        do_send("Recommend courses")
        st.rerun()
with ex2:
    if st.button("Can I take AI101?", use_container_width=True):
        do_send("Can I take AI101?")
        st.rerun()
with ex3:
    if st.button("Give me a plan order", use_container_width=True):
        do_send("Give me a plan order")
        st.rerun()
