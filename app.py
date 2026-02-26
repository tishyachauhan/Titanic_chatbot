"""
app.py – Titanic Chat Agent (Single-file Streamlit deployment)
All modules merged: tools, intent, key_manager, agent, frontend
Deploy on Streamlit Community Cloud — FREE at streamlit.io/cloud
"""

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import io
import re
import base64
import time
import urllib.request
import os
from threading import Lock
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── MUST be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Insight",
    page_icon="⛴",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADER
# ═══════════════════════════════════════════════════════════════════════════════

DATA_URL  = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
DATA_PATH = "titanic.csv"

@st.cache_data(show_spinner="📦 Loading Titanic dataset…")
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    return df

DF = load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# TOOLS  (tools.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


@tool
def handle_offtopic(query: str) -> str:
    """Handles greetings and off-topic questions without calling Gemini."""
    q = query.lower().strip()
    if any(w in q for w in ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]):
        return (
            "👋 **Hello!** I'm Titanic Insight — your AI analyst for the 1912 Titanic dataset.\n\n"
            "I can answer questions about the **891 passengers** — survival rates, ages, "
            "ticket fares, embarkation ports, and more.\n\n"
            "**Try asking:**\n"
            "- *How many females in class 1 did not survive?*\n"
            "- *Show me a histogram of passenger ages*\n"
            "- *What was the survival rate by gender?*"
        )
    if any(w in q for w in ["thank", "thanks"]):
        return "😊 You're welcome! Ask me anything else about the Titanic data."
    if any(w in q for w in ["bye", "goodbye"]):
        return "👋 Goodbye! Come back anytime to explore the Titanic data."
    return (
        "🚢 I'm specialized in the **Titanic dataset** only.\n\n"
        "I can't answer general questions, but I can help with:\n"
        "- Survival rates by gender, class, or port\n"
        "- Passenger ages, fares, and embarkation stats\n"
        "- Custom filters like *'females in class 1 who survived'*\n\n"
        "What would you like to know about the Titanic?"
    )


@tool
def query_custom_filter(query: str) -> str:
    """Answers specific filter questions combining gender + class + survival."""
    q = query.lower()
    gender   = "female" if any(w in q for w in ["female", "women", "woman", "girl"]) else \
               "male"   if any(w in q for w in ["male", "men", "man", "boy"])        else None
    pclass   = 1 if any(w in q for w in ["class 1", "1st class", "first class"])  else \
               2 if any(w in q for w in ["class 2", "2nd class", "second class"]) else \
               3 if any(w in q for w in ["class 3", "3rd class", "third class"])  else None
    survived = 0 if any(w in q for w in ["did not survive", "not survive", "died", "death", "perish", "dead"]) else \
               1 if any(w in q for w in ["survived", "alive", "rescued", "made it"]) else None

    filtered = base_group = DF.copy()
    if gender:
        filtered   = filtered[filtered["sex"] == gender]
        base_group = base_group[base_group["sex"] == gender]
    if pclass:
        filtered   = filtered[filtered["pclass"] == pclass]
        base_group = base_group[base_group["pclass"] == pclass]
    if survived is not None:
        filtered = filtered[filtered["survived"] == survived]

    count      = len(filtered)
    group_size = len(base_group)
    pct        = count / group_size * 100 if group_size > 0 else 0
    gender_lbl = gender.capitalize() if gender else "All genders"
    class_lbl  = f"Class {pclass}"   if pclass  else "All classes"
    surv_lbl   = "Did not survive"   if survived == 0 else "Survived" if survived == 1 else "All outcomes"

    return (
        f"### 🔍 Custom Filter Result\n\n"
        f"| Filter | Value |\n|--------|-------|\n"
        f"| 👤 Gender | {gender_lbl} |\n"
        f"| 🚢 Class | {class_lbl} |\n"
        f"| 📊 Outcome | {surv_lbl} |\n\n"
        f"**Result: {count} passengers** out of {group_size} in this group ({pct:.1f}%)\n\n"
        f"> Out of **{group_size}** {gender_lbl.lower()} passengers in {class_lbl}, "
        f"**{count}** ({pct:.1f}%) — {surv_lbl.lower()}."
    )


@tool
def get_gender_percentage(query: str) -> str:
    """Returns male/female passenger counts and percentages."""
    counts = DF["sex"].value_counts()
    total  = len(DF)
    male   = counts.get("male", 0)
    female = counts.get("female", 0)
    return (
        f"### 👥 Gender Distribution\n\n"
        f"Out of **{total} passengers** on the Titanic:\n\n"
        f"| Gender | Count | Percentage |\n|--------|-------|------------|\n"
        f"| 🚹 Male | {male} | {male/total*100:.1f}% |\n"
        f"| 🚺 Female | {female} | {female/total*100:.1f}% |\n\n"
        f"> The majority of passengers were male, which was common for transatlantic voyages of that era."
    )


@tool
def get_average_fare(query: str) -> str:
    """Returns average, median, min and max ticket fare."""
    fare = DF["fare"].dropna()
    return (
        f"### 🎫 Ticket Fare Statistics\n\n"
        f"| Metric | Amount |\n|--------|--------|\n"
        f"| 💰 Average fare | ${fare.mean():.2f} |\n"
        f"| 📊 Median fare  | ${fare.median():.2f} |\n"
        f"| ⬇️ Cheapest ticket | ${fare.min():.2f} |\n"
        f"| ⬆️ Most expensive ticket | ${fare.max():.2f} |\n\n"
        f"> The average fare was ${fare.mean():.2f}, but the median was only ${fare.median():.2f} — "
        f"meaning a few very expensive 1st-class tickets pulled the average up significantly."
    )


@tool
def get_embarkation_counts(query: str) -> str:
    """Returns passenger counts by embarkation port."""
    port_map = {"C": "Cherbourg 🇫🇷", "Q": "Queenstown 🇮🇪", "S": "Southampton 🇬🇧"}
    counts   = DF["embarked"].dropna().map(port_map).value_counts()
    total    = len(DF)
    rows     = "\n".join(
        f"| {port} | {count} | {count/total*100:.1f}% |" for port, count in counts.items()
    )
    return (
        f"### ⚓ Passengers by Embarkation Port\n\n"
        f"| Port | Passengers | Share |\n|------|-----------|-------|\n{rows}\n\n"
        f"> **Southampton** was the primary departure point — the Titanic's maiden voyage started there on April 10, 1912."
    )


@tool
def get_survival_stats(query: str) -> str:
    """Returns survival stats overall, by gender, and by class."""
    total     = len(DF)
    survived  = int(DF["survived"].sum())
    died      = total - survived
    pct       = survived / total * 100
    by_gender = DF.groupby("sex")["survived"].mean() * 100
    by_class  = DF.groupby("pclass")["survived"].mean() * 100
    gender_rows = "\n".join(
        f"| {'🚺 Female' if g == 'female' else '🚹 Male'} | {r:.1f}% |" for g, r in by_gender.items()
    )
    class_rows = "\n".join(
        f"| {'🥇' if c==1 else '🥈' if c==2 else '🥉'} Class {c} | {r:.1f}% |" for c, r in by_class.items()
    )
    return (
        f"### 🆘 Survival Statistics\n\n"
        f"Of the **{total} passengers**, only **{survived} survived** ({pct:.1f}%).\n\n"
        f"| Outcome | Count | Rate |\n|---------|-------|------|\n"
        f"| ✅ Survived | {survived} | {pct:.1f}% |\n"
        f"| ❌ Did not survive | {died} | {100-pct:.1f}% |\n\n"
        f"#### By Gender\n| Gender | Survival Rate |\n|--------|---------------|\n{gender_rows}\n\n"
        f"#### By Passenger Class\n| Class | Survival Rate |\n|-------|---------------|\n{class_rows}\n\n"
        f"> *\"Women and children first\"* — the data confirms this: female survival rate was dramatically higher than male."
    )


@tool
def get_age_stats(query: str) -> str:
    """Returns age statistics: average, median, oldest, youngest."""
    age     = DF["age"].dropna()
    missing = DF["age"].isna().sum()
    return (
        f"### 🎂 Passenger Age Statistics\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| 📊 Average age | {age.mean():.1f} years |\n"
        f"| 📈 Median age  | {age.median():.1f} years |\n"
        f"| 👶 Youngest passenger | {age.min():.1f} years old |\n"
        f"| 👴 Oldest passenger   | {age.max():.0f} years old |\n"
        f"| ❓ Missing age data   | {missing} passengers |\n\n"
        f"> The oldest passenger was **{age.max():.0f} years old**, and the youngest was just "
        f"**{age.min():.1f} years** — a baby. The average age was {age.mean():.1f} years."
    )


@tool
def get_class_distribution(query: str) -> str:
    """Returns passenger count per class."""
    counts = DF["pclass"].value_counts().sort_index()
    total  = len(DF)
    rows   = "\n".join(
        f"| {'🥇' if c==1 else '🥈' if c==2 else '🥉'} Class {c} | {n} | {n/total*100:.1f}% |"
        for c, n in counts.items()
    )
    return (
        f"### 🚢 Passenger Class Distribution\n\n"
        f"| Class | Passengers | Share |\n|-------|-----------|-------|\n{rows}\n\n"
        f"> More than half the passengers travelled in **3rd class**, reflecting the large number "
        f"of emigrants seeking a new life in America."
    )


@tool
def get_dataset_overview(query: str) -> str:
    """Returns a full overview of the Titanic dataset."""
    total    = len(DF)
    survived = int(DF["survived"].sum())
    male     = int((DF["sex"] == "male").sum())
    female   = int((DF["sex"] == "female").sum())
    avg_age  = DF["age"].mean()
    avg_fare = DF["fare"].mean()
    missing  = {col: int(DF[col].isna().sum()) for col in DF.columns if DF[col].isna().sum() > 0}
    miss_rows = "\n".join(f"| {col} | {n} |" for col, n in missing.items())
    return (
        f"### 🚢 Titanic Dataset Overview\n\n"
        f"The dataset contains records for **{total} passengers** from the RMS Titanic's maiden voyage (April 1912).\n\n"
        f"| Metric | Value |\n|--------|-------|\n"
        f"| 👥 Total passengers | {total} |\n"
        f"| ✅ Survivors | {survived} ({survived/total*100:.1f}%) |\n"
        f"| ❌ Did not survive | {total-survived} ({(total-survived)/total*100:.1f}%) |\n"
        f"| 🚹 Male passengers | {male} |\n"
        f"| 🚺 Female passengers | {female} |\n"
        f"| 🎂 Average age | {avg_age:.1f} years |\n"
        f"| 🎫 Average fare | ${avg_fare:.2f} |\n\n"
        f"#### ❓ Missing Data\n| Column | Missing Values |\n|--------|----------------|\n{miss_rows}\n\n"
        f"> You can ask me about gender, age, fares, survival rates, embarkation ports, or passenger classes!"
    )


@tool
def plot_age_histogram(query: str) -> str:
    """Creates a histogram of passenger ages. Returns IMAGE: + base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ages = DF["age"].dropna()
    ax.hist(ages, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title("Distribution of Passenger Ages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of Passengers")
    ax.axvline(ages.mean(), color="red", linestyle="--", label=f"Mean: {ages.mean():.1f}")
    ax.legend()
    sns.despine(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_survival_by_gender(query: str) -> str:
    """Creates a bar chart of survival by gender. Returns IMAGE: + base64 PNG."""
    fig, ax = plt.subplots(figsize=(7, 4))
    data = DF.groupby(["sex", "survived"]).size().unstack()
    data.columns = ["Did Not Survive", "Survived"]
    data.plot(kind="bar", ax=ax, color=["#d9534f", "#5cb85c"], edgecolor="white")
    ax.set_title("Survival by Gender", fontsize=14, fontweight="bold")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    sns.despine(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_embarkation_bar(query: str) -> str:
    """Creates a bar chart of passengers per port. Returns IMAGE: + base64 PNG."""
    port_map = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    counts   = DF["embarked"].dropna().map(port_map).value_counts()
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="white")
    ax.set_title("Passengers by Embarkation Port", fontsize=14, fontweight="bold")
    ax.set_xlabel("Port")
    ax.set_ylabel("Number of Passengers")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    sns.despine(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_fare_distribution(query: str) -> str:
    """Creates histogram + boxplot of ticket fares. Returns IMAGE: + base64 PNG."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fares = DF["fare"].dropna()
    ax1.hist(fares, bins=40, color="#9467bd", edgecolor="white", alpha=0.85)
    ax1.set_title("Fare Distribution")
    ax1.set_xlabel("Fare ($)")
    ax1.set_ylabel("Count")
    ax2.boxplot(fares, vert=True, patch_artist=True, boxprops=dict(facecolor="#9467bd", alpha=0.7))
    ax2.set_title("Fare Boxplot")
    ax2.set_ylabel("Fare ($)")
    fig.suptitle("Ticket Fare Analysis", fontsize=14, fontweight="bold")
    sns.despine(fig)
    plt.tight_layout()
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_class_survival(query: str) -> str:
    """Creates a grouped bar chart of survival by class. Returns IMAGE: + base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = DF.groupby(["pclass", "survived"]).size().unstack()
    data.columns = ["Did Not Survive", "Survived"]
    data.plot(kind="bar", ax=ax, color=["#d9534f", "#5cb85c"], edgecolor="white")
    ax.set_title("Survival by Passenger Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Passenger Class")
    ax.set_ylabel("Number of Passengers")
    ax.set_xticklabels([f"Class {i}" for i in data.index], rotation=0)
    ax.legend()
    sns.despine(fig)
    return "IMAGE:" + _fig_to_b64(fig)


@tool
def plot_gender_pie(query: str) -> str:
    """Creates a pie chart of male vs female passengers. Returns IMAGE: + base64 PNG."""
    counts = DF["sex"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[g.capitalize() for g in counts.index],
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("Gender Distribution of Titanic Passengers", fontsize=14, fontweight="bold", pad=20)
    return "IMAGE:" + _fig_to_b64(fig)


ALL_TOOLS = [
    handle_offtopic, query_custom_filter,
    get_gender_percentage, get_average_fare, get_embarkation_counts,
    get_survival_stats, get_age_stats, get_class_distribution, get_dataset_overview,
    plot_age_histogram, plot_survival_by_gender, plot_embarkation_bar,
    plot_fare_distribution, plot_class_survival, plot_gender_pie,
]


# ═══════════════════════════════════════════════════════════════════════════════
# KEY MANAGER  (key_manager.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ApiKey:
    key: str
    index: int
    failures: int = 0
    last_failure_time: float = 0.0
    cooldown_seconds: int = 60
    is_exhausted: bool = False

    @property
    def is_in_cooldown(self) -> bool:
        if self.last_failure_time == 0:
            return False
        return (time.time() - self.last_failure_time) < self.cooldown_seconds

    @property
    def is_available(self) -> bool:
        return not self.is_exhausted and not self.is_in_cooldown

    def mark_failure(self, exhausted: bool = False):
        self.failures += 1
        self.last_failure_time = time.time()
        if exhausted:
            self.is_exhausted = True

    def mark_success(self):
        self.failures = 0
        self.last_failure_time = 0.0


class KeyManager:
    def __init__(self, keys: list):
        self._keys    = [ApiKey(key=k, index=i) for i, k in enumerate(keys) if k.strip()]
        self._current = 0
        self._lock    = Lock()
        if not self._keys:
            raise ValueError("No API keys provided!")

    def get_current_key(self) -> Optional[ApiKey]:
        with self._lock:
            for _ in range(len(self._keys)):
                key = self._keys[self._current % len(self._keys)]
                if key.is_available:
                    return key
                self._current += 1
            return None

    def next_key(self) -> Optional[ApiKey]:
        with self._lock:
            self._current += 1
            for _ in range(len(self._keys)):
                key = self._keys[self._current % len(self._keys)]
                if key.is_available:
                    return key
                self._current += 1
            return None

    def report_failure(self, key: ApiKey, exhausted: bool = False):
        key.mark_failure(exhausted=exhausted)

    def report_success(self, key: ApiKey):
        key.mark_success()


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT  (agent.py)
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a helpful and friendly data analyst specializing in the Titanic dataset.

Your job is to answer questions about Titanic passengers accurately and clearly.

RULES:
1. Always use the provided tools to fetch data — never make up statistics.
2. When a user asks for a chart, histogram, plot, or visualization → use a plot_ tool.
3. When a user asks for numbers/stats without visualization → use a text tool.
4. After using a tool, explain the result in a conversational, friendly tone.
5. If a question is unrelated to the Titanic dataset, politely say so.
6. For ambiguous questions (e.g. "tell me about passengers"), use get_dataset_overview.
7. If a visualization tool is called, mention that the chart is displayed above your text.

Titanic dataset columns: survived, pclass, sex, age, sibsp, parch, fare, embarked, name, ticket, cabin.
"""


def _make_executor(api_key: str) -> AgentExecutor:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def run_agent_with_rotation(key_manager: KeyManager, question: str, chat_history: list) -> dict:
    attempted = set()
    while True:
        key_obj = key_manager.get_current_key()
        if key_obj is None:
            raise RuntimeError("All API keys are rate-limited.")
        if key_obj.index in attempted:
            key_obj = key_manager.next_key()
            if key_obj is None or key_obj.index in attempted:
                raise RuntimeError("All API keys tried and failed.")
        attempted.add(key_obj.index)

        try:
            executor = _make_executor(key_obj.key)
            result   = executor.invoke({"input": question, "chat_history": chat_history})
            key_manager.report_success(key_obj)

            output    = result.get("output", "")
            image_b64 = None
            for step in result.get("intermediate_steps", []):
                tool_out = step[1] if isinstance(step, tuple) else ""
                if isinstance(tool_out, str) and tool_out.startswith("IMAGE:"):
                    image_b64 = tool_out[6:]
                    break

            return {"text": output.replace("IMAGE:", "").strip(), "image": image_b64}

        except Exception as e:
            err = str(e).lower()
            if any(w in err for w in ["429", "quota", "rate", "resource_exhausted"]):
                exhausted = any(w in err for w in ["daily", "per_day", "exceeded"])
                key_manager.report_failure(key_obj, exhausted=exhausted)
                key_manager.next_key()
                continue
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT LAYER  (intent.py)
# ═══════════════════════════════════════════════════════════════════════════════

CHART_WORDS = [
    "plot", "chart", "graph", "histogram", "visuali", "show me",
    "display", "draw", "diagram", "pie", "bar chart", "bar graph",
    "distribution", "breakdown", "illustrate",
]

def _wants_chart(q: str) -> bool:
    return any(w in q for w in CHART_WORDS)


TOOL_REGISTRY = {
    "handle_offtopic":         handle_offtopic,
    "query_custom_filter":     query_custom_filter,
    "get_gender_percentage":   get_gender_percentage,
    "get_average_fare":        get_average_fare,
    "get_embarkation_counts":  get_embarkation_counts,
    "get_survival_stats":      get_survival_stats,
    "get_age_stats":           get_age_stats,
    "get_class_distribution":  get_class_distribution,
    "get_dataset_overview":    get_dataset_overview,
    "plot_age_histogram":      plot_age_histogram,
    "plot_survival_by_gender": plot_survival_by_gender,
    "plot_embarkation_bar":    plot_embarkation_bar,
    "plot_fare_distribution":  plot_fare_distribution,
    "plot_class_survival":     plot_class_survival,
    "plot_gender_pie":         plot_gender_pie,
}

@dataclass
class Intent:
    topic: str
    wants_chart: bool
    chart_type: str
    text_tool: str
    chart_tool: str


TOPIC_MAP = [
    (
        ["male", "female", "gender", "sex", "men", "women", "boy", "girl"],
        Intent("gender", True, "pie", "get_gender_percentage", "plot_gender_pie"),
    ),
    (
        ["survival rate", "who survived", "how many survived", "how many died",
         "did not survive", "did survive", "rescued", "perish", "death toll",
         "show survival", "plot survival", "survival by"],
        Intent("survival", True, "grouped_bar", "get_survival_stats", "plot_survival_by_gender"),
    ),
    (
        ["fare", "ticket price", "ticket cost", "cost", "price", "paid", "money", "ticket", "cheap", "expensive"],
        Intent("fare", False, "boxplot", "get_average_fare", "plot_fare_distribution"),
    ),
    (
        ["age", "old", "young", "oldest", "youngest", "how old", "years old"],
        Intent("age", False, "histogram", "get_age_stats", "plot_age_histogram"),
    ),
    (
        ["embark", "port", "southampton", "cherbourg", "queenstown", "boarded", "board"],
        Intent("embarkation", True, "bar", "get_embarkation_counts", "plot_embarkation_bar"),
    ),
    (
        ["class", "1st", "2nd", "3rd", "first class", "second class", "third class", "pclass"],
        Intent("class", True, "grouped_bar", "get_class_distribution", "plot_class_survival"),
    ),
    (
        ["overview", "summary", "about", "dataset", "describe", "info",
         "total", "how many passenger", "passengers", "general", "tell me"],
        Intent("overview", False, "bar", "get_dataset_overview", "plot_embarkation_bar"),
    ),
]


def classify_intent(question: str) -> Optional[Intent]:
    q = question.lower()
    for triggers, base in TOPIC_MAP:
        if any(t in q for t in triggers):
            return Intent(
                topic=base.topic,
                wants_chart=base.wants_chart or _wants_chart(q),
                chart_type=base.chart_type,
                text_tool=base.text_tool,
                chart_tool=base.chart_tool,
            )
    return None


def resolve_intent(question: str) -> Optional[dict]:
    intent = classify_intent(question)
    if intent is None:
        return None

    text_fn     = TOOL_REGISTRY.get(intent.text_tool)
    text_result = text_fn.invoke(question) if text_fn else "No data available."
    if isinstance(text_result, str) and text_result.startswith("IMAGE:"):
        text_result = f"Here's your {intent.topic} data! 📊"

    image_b64 = None
    if intent.wants_chart:
        chart_fn = TOOL_REGISTRY.get(intent.chart_tool)
        if chart_fn:
            chart_result = chart_fn.invoke(question)
            if isinstance(chart_result, str) and chart_result.startswith("IMAGE:"):
                image_b64 = chart_result[6:]

    return {"text": text_result, "image": image_b64}


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI  (app.py / frontend)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
* { font-family: 'Nunito', sans-serif !important; }
p, li, div, span, label { font-size: 16px !important; }
h1 { font-size: 36px !important; font-weight: 800 !important; color: #0c7a6d !important; }
h3 { font-size: 20px !important; font-weight: 700 !important; color: #0e9e8e !important; }
[data-testid="stMarkdownContainer"] p  { font-size: 16px !important; line-height: 1.75 !important; }
[data-testid="stMarkdownContainer"] td { font-size: 15px !important; }
[data-testid="stMarkdownContainer"] th { font-size: 13px !important; }
[data-testid="stChatInputTextArea"] textarea { font-size: 16px !important; }
div[data-testid="stButton"] > button {
    background-color: #0e9e8e !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-size: 15px !important; font-weight: 600 !important;
}
div[data-testid="stButton"] > button:hover { background-color: #0c8a7c !important; }
#MainMenu, footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "key_manager"  not in st.session_state: st.session_state.key_manager  = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⛴ Titanic Insight")
    st.markdown("Powered by **Gemini AI** + **LangChain**")
    st.divider()

    st.markdown("### 🔑 Your Gemini API Key")
    st.markdown("Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)")

    api_key_input = st.text_input(
        label="Paste your key here",
        type="password",
        placeholder="AIzaSy...",
        help="Your key is never stored — only used for this session.",
    )

    if st.button("✅ Apply Key", use_container_width=True):
        key = api_key_input.strip()
        if not key:
            st.error("Please paste a key first.")
        elif not key.startswith("AIza"):
            st.error("That doesn't look like a valid Gemini key.")
        else:
            try:
                st.session_state.key_manager  = KeyManager([key])
                st.session_state.messages     = []
                st.success("✅ Key applied! You can now chat.")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    if st.session_state.key_manager:
        st.success("🟢 Agent ready")
    else:
        st.warning("🟡 Paste a Gemini key above to activate")

    st.divider()
    st.markdown("**Try asking:**")
    st.markdown("""
- What % of passengers were male?
- Show a histogram of passenger ages
- Average ticket fare?
- Passengers per embarkation port
- Survival rate by gender
- Show survival by class
- How old was the oldest passenger?
""")
    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("⛴ Titanic Insight")
st.markdown("Ask anything about the Titanic passengers — instant answers and charts.")
st.divider()

if not st.session_state.key_manager:
    st.warning("👈 **Paste your free Gemini API key in the sidebar to start chatting.**")
    st.stop()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            img_bytes = base64.b64decode(msg["image"])
            st.image(Image.open(io.BytesIO(img_bytes)), use_column_width=True)
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about the Titanic…")

if user_input:
    question = user_input.strip()
    if not question:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing…"):
            try:
                # Layer 1: instant intent resolution
                result = resolve_intent(question)

                # Layer 2: Gemini agent fallback
                if result is None:
                    lc_history = []
                    for m in st.session_state.messages[-11:-1]:
                        if m["role"] == "user":
                            lc_history.append(HumanMessage(content=m["content"]))
                        elif m["role"] == "assistant":
                            lc_history.append(AIMessage(content=m["content"]))

                    result = run_agent_with_rotation(
                        st.session_state.key_manager, question, lc_history
                    )

                text      = result.get("text", "")
                image_b64 = result.get("image")

                if image_b64:
                    img_bytes = base64.b64decode(image_b64)
                    st.image(Image.open(io.BytesIO(img_bytes)), use_column_width=True)
                st.markdown(text)

                st.session_state.messages.append({
                    "role": "assistant", "content": text, "image": image_b64,
                })

            except RuntimeError as e:
                st.warning(f"⏳ {e} Try again in ~60 seconds or apply a new key in the sidebar.")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
