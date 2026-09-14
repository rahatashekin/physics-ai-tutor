# 🧠 AI Tutoring System — Vision & Design Document

> **Master Document — সব brainstorming progress এখানে consolidated।**
> Last Updated: 2026-08-31
> Status: BRAINSTORMING PHASE (no code execution yet)

---

## Table of Contents

1. [Core Philosophy](#-1-core-philosophy)
2. [Student Genome](#-2-student-genome)
3. [Hint Cascade](#-3-hint-cascade)
4. [Tutor Intelligence — সব 6 Dimensions](#-4-tutor-intelligence--সব-6-dimensions)
5. [UX & Interface Design](#-5-ux--interface-design)
6. [Fundamental Modes](#-6-fundamental-modes-teach-vs-practice)
7. [Quiz & Assessment System](#-7-quiz--assessment-system)
8. [Interrupt-to-Assess Engine](#-8-interrupt-to-assess-engine)
9. [Memory Architecture](#-9-memory-architecture)
10. [Note System](#-10-note-system)
11. [Progress System](#-11-progress-system)
12. [Complete Routing Map](#-12-complete-routing-map)
13. [Product Form](#-13-product-form)
14. [Decisions Log](#-14-decisions-log)
15. [Open Questions](#-15-open-questions)

---

## 💡 1. Core Philosophy

### The Problem
Bangladesh এ AI education tools আসছে — কিন্তু সবগুলো basically "বইয়ের answer দেয় এমন chatbot"।

### Our Differentiator
> **একজন private tutor যে ছাত্রকে চেনে, ছাত্রের সাথে grow করে, আর ছাত্রকে শুধু উত্তর দেয় না — শেখায়।**

### The Three Revolutions

```
Revolution 1: KNOWS the student     → Student Genome
Revolution 2: ADAPTS the teaching   → Tutor Intelligence + Hint Engine
Revolution 3: MEASURES real learning → Progress Engine
```

বেশিরভাগ EdTech product এর মধ্যে মাত্র ১টা আছে (সেটাও shallow)। আমাদের system এ তিনটাই deeply integrated — এটাই differentiator।

### The "Teaching vs Answering" Principle

> **System এর #1 rule: ChatGPT উত্তর দেয়। আমাদের tutor শেখায়।**
>
> ChatGPT তে ছাত্র answer পায় কিন্তু next time আবার same প্রশ্ন করে।
> আমাদের system এ ছাত্র নিজে discover করে, তাই মনে থাকে।
>
> **তবে — সব পরিস্থিতিতে hint দেওয়া ভুল:**
> - Concept question ("আলোর প্রতিফলন কী?") → সরাসরি শেখাও (TEACH MODE)
> - Problem-solving ("ত্বরণ কত?") → guided discovery with hints (PRACTICE MODE)
> - System automatically detect করে কখন কোন mode লাগবে

---

## 🧬 2. Student Genome

প্রতিটা ছাত্রের একটা living, evolving profile — শুধু একটা number (3/5) না, একটা multi-dimensional map।

### Structure (5 Layers)

```
LAYER 1: IDENTITY (static)
  ├── name, class, school
  ├── curriculum (NCTB / Cambridge / custom)
  └── account_type (guest / registered)

LAYER 2: KNOWLEDGE MAP (evolving)
  ├── chapter_mastery: {ch1: 0.75, ch2: 0.3, ...}
  ├── topic_mastery: {
  │     "নিউটনের ২য় সূত্র": 0.9,
  │     "আলোর প্রতিসরণ": 0.2,
  │     "ওহমের সূত্র": 0.6
  │   }
  ├── misconceptions: [
  │     { topic: "বল", belief: "ওজন আর ভর একই",
  │       detected: "2026-08-15", resolved: false }
  │   ]
  ├── gap_topics: ["চ্যাপ্টার ৯ পূর্ণ অভ্যন্তরীণ"]
  └── hint_profile: {
        "নিউটনের ২য় সূত্র": {
            avg_hint_level: 0.5,   # usually L0-L1 (strong)
            trend: "improving"
        },
        "ওহমের সূত্র": {
            avg_hint_level: 2.5,   # needs L2-L3 (weak)
            trend: "stuck"
        }
      }

LAYER 3: LEARNING PROFILE (observed + selected)
  ├── preferred_style: "উদাহরণ" (student selected — or "স্বয়ংক্রিয়")
  ├── effective_style: "উপমা" (system observed over time)
  ├── hint_tolerance: "patient"
  │   (কতক্ষণ try করে? quickly "বলো" বলে নাকি ধৈর্য ধরে?)
  ├── language_comfort: "Banglish-heavy"
  └── pace: "moderate"

LAYER 4: BEHAVIORAL SIGNALS (real-time, this session)
  ├── current_mood: "curious"
  ├── recent_signals: ["understood", "confused", "mastered"]
  ├── engagement_trend: "increasing"
  └── session_energy: "high" (replied quickly 3 times)

LAYER 5: MEMORY (persistent across sessions)
  ├── session_summaries: [last 20 sessions]
  ├── quiz_history: {topic: [{score, hint_level, date}]}
  ├── hint_history: {
  │     topic: [{date, level_needed, correct, context}]
  │   }
  ├── unresolved_questions: [...]
  ├── student_notes: [...]
  └── revisit_flags: ["চ্যাপ্টার ৫ আর্কিমিডিস"]
```

### Key Design Decisions

**Observed vs Selected (Learning Profile):**
- `preferred_style` = student নিজে select করে (or স্বয়ংক্রিয়)
- `effective_style` = system observe করে কোন style এ student ভালো বোঝে + quiz score ভালো হয়
- System student এর choice কে priority দেয়, কিন্তু softly suggest করে:
  "তুমি 'সূত্র' mode select করেছো, কিন্তু দেখেছি 'উপমা' তে faster বোঝো। Try করবে?"

**Misconception Tracking:**
- Student ভুল answer দিলে বা ভুল ধারণা প্রকাশ করলে → misconceptions list এ store
- পরে related topic আসলে tutor proactively correct করে
- Misconception has `resolved` flag — system tracks if it's been fixed

**Hint Profile (per topic):**
- avg_hint_level: কত level hint লাগে usually
- trend: improving / stuck / declining
- This drives: starting hint level + progress calculation

---

## 🎯 3. Hint Cascade

Hint কোনো toggle বা separate mode না — এটা Practice Mode এর core engine।
Tutor সবসময় lower level থেকে শুরু করে (Genome অনুযায়ী), student আটকে থাকলে gradually deeper hint দেয়।

### 5 Levels — Progressive Revelation

```
Level 0: SOCRATIC (প্রশ্ন দিয়ে ভাবাও)
  → উত্তর দেয় না — counter-question করে
  → "তুমি কী মনে করো এখানে কোন সূত্র কাজ করছে?"
  → Goal: student নিজে connection খুঁজুক

Level 1: DIRECTIONAL (দিকনির্দেশনা)
  → Topic area দেখিয়ে দেয়, specific কিছু বলে না
  → "নিউটনের সূত্রগুলো মনে করো — কোনটা apply হবে?"
  → Goal: সঠিক area তে focus করাও

Level 2: STRUCTURAL (গঠনগত সাহায্য)
  → সূত্র দেখায়, values identify করতে সাহায্য করে
  → "F = ma — এখানে m = 5 kg দেওয়া আছে, a বের করতে হবে"
  → Goal: framework দাও, calculation student করুক

Level 3: STEP-BY-STEP (প্রায় সমাধান)
  → শেষ step ছাড়া সব করে দেয়
  → "F = ma, তাহলে 10 = 5 × a, মানে a = ?"
  → Goal: শুধু final answer student বলুক

Level 4: DIRECT ANSWER (সরাসরি উত্তর)
  → সম্পূর্ণ উত্তর + বিস্তারিত ব্যাখ্যা
  → "a = 2 m/s²। চলো বুঝি কেন..."
  → ⚠️ LAST RESORT — শুধু যখন student try করেছে কিন্তু পারেনি
```

### When Hint Cascade Applies vs Doesn't

```
APPLIES (Practice Mode):
  ✅ Problem-solving questions ("ত্বরণ কত?")
  ✅ Quiz/assessment scenarios
  ✅ Interrupt engine checks (recall probes, quick checks)

DOES NOT APPLY (Teach Mode):
  ❌ Concept questions ("আলোর প্রতিফলন কী?")
  ❌ Explanation requests ("F=ma বুঝিয়ে দাও")
  ❌ "Why" questions ("কেন পড়ন্ত বস্তু ত্বরিত?")
  → These get direct explanation/teaching instead
```

### Starting Hint Level (Genome-Driven)

| Condition | Starting Level |
|---|---|
| topic mastery > 0.7 | L0 (Socratic — তুমি জানো) |
| topic mastery 0.4–0.7 | L1 (Directional) |
| topic mastery < 0.4 | L2 (Structural) |
| "frustrated" signal | L3 (Step — দ্রুত help) |
| misconception flagged | L0 (ভাবতে দাও, then correct) |

Escalation: student wrong বা "জানি না" বললে → current level + 1 (until L4)

### Student Override (In-Chat)

| Student বলে | System করে | Mastery Impact |
|---|---|---|
| "সরাসরি বলো" | L4 direct answer | Credit 0.05 (minimal) |
| "hint দাও" / "ভাবতে দাও" | Stay at L0-L1 | Credit maximum |
| (কিছু বলে না — default) | Genome decides | Context-appropriate |

Student এর "সরাসরি বলো" request সম্মান করা হবে — কিন্তু system record রাখবে।
যদি সবসময় direct চায়, tutor gently encourage করবে নিজে try করতে।

### Hint-Weighted Mastery Credits

```
Unaided correct      →  1.00
Correct at L0 hint   →  0.85
Correct at L1 hint   →  0.65
Correct at L2 hint   →  0.45
Correct at L3 hint   →  0.25
Direct answer (L4)   →  0.05
Wrong even with L4   →  0.00
```

---

## 🎭 4. Tutor Intelligence — সব 6 Dimensions

Tutor এর response কেমন হবে সেটা 6টা dimension মিলে decide হয়।
কিছু system নিজে decide করে (Genome-driven), কিছু student override করতে পারে (default: স্বয়ংক্রিয় = system decides)।

### Overview — 6 Dimensions at a Glance

```
WHO DECIDES    │ DIMENSION                │ OPTIONS                      │ AFFECTS
───────────────┼──────────────────────────┼──────────────────────────────┼──────────────
🤖 System      │ 1. Fundamental Mode      │ Teach / Practice             │ কোন mode
🤖 System      │ 2. Teaching Strategy     │ 6 types                      │ approach কী
🤖 System      │ 3. Hint Level            │ L0-L4 (Practice only)        │ কতটুকু reveal
👤 Student*    │ 4. ব্যাখ্যার ধরন         │ 5 (inc auto)                 │ কীভাবে বোঝাবে
👤 Student*    │ 5. সাহায্যের মাত্রা       │ 4 (inc auto)                 │ কতটুকু help
👤 Student*    │ 6. Personality           │ 5 (inc auto)                 │ কেমন tone

* Student CAN override, but default = স্বয়ংক্রিয় (system decides via Genome)
```

All 6 dimensions combine → system prompt generate হয় → LLM response দেয়।

---

### Dimension 1: Fundamental Mode (system auto-detects)

Intent detection থেকে system decide করে কোন mode:
- **🎓 TEACH** — concept/explanation questions → সরাসরি শেখায়
- **🧩 PRACTICE** — problem-solving/quiz → Hint Cascade activates

Details: See [Section 6: Fundamental Modes](#-6-fundamental-modes-teach-vs-practice)

---

### Dimension 2: Teaching Strategy (6 total, system auto-selects)

| Strategy | When | In Teach Mode | In Practice Mode |
|---|---|---|---|
| 🏗️ **Scaffolding** | New topic (mastery=0) | Basics থেকে build up | L2 hint start, slow |
| 🏋️ **Challenge** | Strong (3+ understood, mastery>0.7) | Deeper concepts, "কেন?" ask | L0 only, "তুমি পারবে" |
| 🛟 **Rescue** | Confused (2+ confused, mastery<0.3) | Simpler words, more examples | L3 start, fast escalate |
| 📝 **Assessment** | 5+ exchanges same topic, or scheduled | Switch to Practice briefly | Quiz + hint cascade |
| 🔁 **Recall** | Returning after 2+ days, decay threshold | Quick summary of last topic | Recall probe + hints |
| 🔬 **Exploration** | Student asks "কেন?", shows curiosity | Deep dive, connections | L0, open-ended |

---

### Dimension 3: Hint Level (system auto, Practice Mode only)

Progressive revelation: L0 Socratic → L1 Directional → L2 Structural → L3 Step-by-Step → L4 Direct Answer

Details: See [Section 3: Hint Cascade](#-3-hint-cascade)

#### Strategy × Hint Behavior Matrix

| Strategy | Starting Hint | Escalation Speed | Max Level |
|---|---|---|---|
| Scaffolding | L2 | Moderate | L4 |
| Challenge | L0 | Very slow | L2 |
| Rescue | L3 | Fast | L4 |
| Assessment | L1 | Moderate | L3 |
| Recall | L0 | Slow | L2 |
| Exploration | L0 | None | L1 |

---

### Dimension 4: ব্যাখ্যার ধরন — Explanation Style (student-overridable)

**কীভাবে** content deliver করবে:

| Option | মানে | Example |
|---|---|---|
| 🤖 **স্বয়ংক্রিয়** ← DEFAULT | Genome.effective_style | system observe করে কোন style এ ভালো বোঝে |
| 🔗 **উপমা** | Analogy-based | "মনে করো রিকশায় বসে আছো..." |
| 📐 **সূত্র** | Formula-first | "F=ma, এখানে F হলো..." |
| 💡 **উদাহরণ** | Example-based | "যেমন, ক্রিকেট বলে..." |
| 🪜 **ধাপে ধাপে** | Numbered steps | "ধাপ ১: ... ধাপ ২: ..." |

---

### Dimension 5: সাহায্যের মাত্রা — Guidance Level (student-overridable)

**কতটুকু** help দেবে — এবং Teach/Practice mode ও Hint Level কে influence করে:

| Option | মানে | Mode Bias | Hint Behavior |
|---|---|---|---|
| 🤖 **স্বয়ংক্রিয়** ← DEFAULT | Context-based | Pure intent detection | Genome-based |
| 📖 **সরাসরি শেখাও** | Direct teach | Bias → TEACH | L3-L4 (minimal hints) |
| ⚖️ **সুযোগ দাও** | Balanced | Balanced | Genome-based |
| 🧩 **নিজে করি** | Max self-effort | Bias → PRACTICE | L0 always (max hints) |

---

### Dimension 6: Personality (student-overridable)

**কেমন tone** এ কথা বলবে:

| Style | Tone | Example |
|---|---|---|
| 🤖 **স্বয়ংক্রিয়** ← DEFAULT | System observes what works | Genome.observed_personality |
| 🤝 **বন্ধুসুলভ** | Casual, friendly | "ভাই দেখ, বলটা যখন পড়ে..." |
| 📚 **আনুষ্ঠানিক** | Formal, textbook | "বল প্রয়োগের ফলে ত্বরণ সৃষ্টি হয়..." |
| 📖 **গল্পকার** | Story-driven | "মনে করো তুমি রিকশায় বসে আছো..." |
| 🧩 **প্রতিযোগী** | Playful, competitive | "Challenge! ৩০ সেকেন্ডে বলো..." |

---

### How All 6 Dimensions Combine

```python
# Conceptual — সব dimensions এক prompt এ merge হয়
system_prompt = build_prompt(
    mode=detected_mode,                    # Dim 1: Teach or Practice
    strategy=selected_strategy,            # Dim 2: Scaffolding/Challenge/etc
    hint_level=starting_hint,              # Dim 3: L0-L4 (Practice only)
    style=student_or_genome_style,         # Dim 4: উপমা/সূত্র/etc
    guidance=student_or_genome_guidance,    # Dim 5: সরাসরি/সুযোগ/নিজে
    personality=student_or_genome_personality, # Dim 6: বন্ধুসুলভ/etc
    genome_context=student_genome,         # Student profile
    book_context=rag_results,              # Retrieved content
)

# Conversation history SEPARATELY:
response = llm.chat(
    system_instruction=system_prompt,       # ← 6 dimensions + context
    history=last_N_messages,                # ← conversation (আলাদা)
    new_message=student_latest_message,
)
```

---

## ⚙️ 5. UX & Interface Design

> Section 4 defines WHAT the 6 dimensions are।
> This section defines WHERE and HOW students interact with them in the UI।

### Design Principle

> **"Zero-config by default, power in settings."**
>
> Student first time আসবে → সবকিছু "স্বয়ংক্রিয়" তে auto-selected
> → system intelligently handle করবে → student কিচ্ছু করতে হবে না, just chat করবে।
> Advanced student চাইলে settings এ গিয়ে manually override করবে।
> 90% student কখনো settings ছোঁবেই না — আর সেটাই correct behavior।

### Default State (first visit)

```
Settings (collapsed by default):
  ├── ব্যাখ্যার ধরন:  🤖 স্বয়ংক্রিয় ✓ (auto-selected)
  ├── সাহায্যের মাত্রা: 🤖 স্বয়ংক্রিয় ✓ (auto-selected)
  └── টিউটরের ভাষা:   🤖 স্বয়ংক্রিয় ✓ (auto-selected)

→ Student just types and starts learning
→ System handles everything via Genome
→ Settings থাকবে, কিন্তু buried — not in-your-face
```

### In-Chat Overrides (যেকোনো সময়)

Settings না ছুঁয়েই student chat এ explicitly বলতে পারবে:
- "সরাসরি বলো" → instant direct answer (skip all hints)
- "hint দাও" → switch to hint cascade
- "quiz দাও" → start quiz/assessment
- "উপমা দিয়ে বোঝাও" → temporary style override for this exchange

### What "স্বয়ংক্রিয়" Means (for each control)

| Control | "স্বয়ংক্রিয়" = | Who Decides |
|---|---|---|
| ব্যাখ্যার ধরন | Genome.effective_style (observed) | System — কোন style এ student ভালো বোঝে |
| সাহায্যের মাত্রা | Context-based (Teach vs Practice) | System — question intent + Genome |
| টিউটরের ভাষা | Genome.observed_personality | System — কোন tone এ engagement বেশি |

---

## 🔀 6. Fundamental Modes: Teach vs Practice

System automatically detect করবে student এর question কোন mode এ পড়ে:

### 🎓 TEACH MODE (for learning/concept questions)
- Concept explain করে
- উদাহরণ দেয়, সূত্র বোঝায়
- Adaptive explanation style applies
- NO hint cascade — directly teaches
- "আলোর প্রতিফলন কী?" → full explanation

### 🧩 PRACTICE MODE (for problem-solving/quiz/assessment)
- Hint Cascade activates (L0→L4)
- Student নিজে solve করতে guide করে
- Quiz + interrupt assessments
- "ত্বরণ কত?" → "কোন সূত্র লাগবে ভাবো..."

### Intent Detection (which question → which mode)

| Question Pattern | Detected Intent | Mode |
|---|---|---|
| "আলোর প্রতিফলন কী?" | LEARNING | 🎓 TEACH |
| "F=ma বুঝিয়ে দাও" | EXPLANATION | 🎓 TEACH |
| "কেন পড়ন্ত বস্তু ত্বরিত?" | WHY (curiosity) | 🎓 TEACH |
| "5 kg বস্তুতে 10N বল, ত্বরণ কত?" | PROBLEM | 🧩 PRACTICE |
| "এটা সমাধান করো" | PROBLEM | 🧩 PRACTICE |
| "quiz দাও" | ASSESSMENT_REQUEST | 🧩 PRACTICE |

### Guidance Level Override on Mode Selection

| সাহায্যের মাত্রা Setting | Override Behavior |
|---|---|
| স্বয়ংক্রিয় | Pure intent detection (no bias) |
| সরাসরি শেখাও | Bias toward TEACH (even problems get more direct) |
| সুযোগ দাও | Balanced (teach concepts, hint on problems) |
| নিজে করি | Bias toward PRACTICE (even concepts get Socratic questions) |

---

## 📝 7. Quiz & Assessment System

### 5 Quiz Types

| Type | Format | Hint Integration | Example |
|---|---|---|---|
| **MCQ** | 4 options, 1 correct | Wrong → Hint L1 → try again | "কোনটি মৌলিক রাশি?" |
| **Short Answer** | Open text, LLM graded | Full cascade L0→L4 | "F = ma তে a কী?" |
| **Numerical** | Calculate + answer | Hint: সূত্র→values→steps | "ত্বরণ = ?" |
| **Conceptual** | Explain in own words | Socratic follow-ups | "চাঁদে ভর পাল্টাবে?" |
| **Diagram-Based** | Label/identify parts | Visual hints | "voltage কোথায়?" |

### Hint-Integrated Quiz Flow

```
Tutor: "5 kg বস্তুতে 10 N বল দিলে ত্বরণ কত?"
  │
  ├── Student: "2 m/s²" ✅ CORRECT, UNAIDED
  │   → mastery credit: 1.00 → "চমৎকার! 🎉"
  │
  ├── Student: "50" ❌ WRONG
  │   → "ভালো চেষ্টা! একটু ভাবো..."
  │   → Hint L1: "কোন সূত্রে বল, ভর, ত্বরণ আছে?"
  │   ├── Student: "F=ma... a=F/m=2" ✅
  │   │   → mastery credit: 0.65
  │   └── Student: "জানি না"
  │       → Hint L2: "F=ma, F=10, m=5, a=?"
  │       ├── Student: "a=10/5=2" ✅ → credit: 0.45
  │       └── → Hint L3 → L4 (cascade continues)
  │
  └── Student: "বুঝি না, hint দাও"
      → Start at L1 → cascade normally
```

### Quiz Scheduling (When quizzes happen)

- **Topic completion**: section পড়া শেষ → 2-3 quick quiz
- **Interrupt Engine**: conversation মাঝে (see Interrupt Engine section)
- **Recall check**: পুরনো topic revisit (forgetting curve based)
- **Student request**: "quiz দাও"
- **Pre-exam**: comprehensive quiz

---

## ⏸️ 8. Interrupt-to-Assess Engine

Tutor naturally কথার মধ্যে থামিয়ে check করে — ছাত্র টেরও পায় না যে "পরীক্ষা" হচ্ছে।

### 4 Interrupt Types

| Type | Trigger | Hint Behavior | Duration |
|---|---|---|---|
| **Quick Check** | প্রতি ৩-৪ exchange পর | L0-L1 (quick, low pressure) | 1-2 exchanges |
| **Recall Probe** | আগের session topic, or forgetting curve | L0, slow escalation | 2-3 exchanges |
| **Challenge Question** | Student confident, mastery>0.6 | L0 only (thinking) | 3-5 exchanges |
| **Misconception Check** | Genome misconception flag + related topic | L0 first, then L4 explain | 2-4 exchanges |

### Interrupt Frequency Rules

| Rule | Detail |
|---|---|
| Maximum frequency | 1 interrupt per 5 exchanges |
| No early interrupt | First 3 exchanges of new topic → no interrupt |
| No consecutive | Never 2 interrupts in a row |
| Respect student | Student বলে "পরে quiz দিব" → delay all |
| Energy-aware | Session energy "low" → reduce frequency |
| Celebrate success | Quick check correct → "চমৎকার!" → move on fast |

### Interrupt Response Evaluation (LLM-Based)

```
Student Reply → LLM Judge → Classification:
  ├── CORRECT + CONFIDENT  → mastery ↑↑, no hint needed
  ├── CORRECT + HESITANT   → mastery ↑, note for reinforcement
  ├── PARTIALLY CORRECT    → mastery ↔, hint L1 to complete
  ├── WRONG + MISCONCEPTION → add to genome misconceptions, explain
  └── WRONG + RANDOM       → mastery ↓, hint cascade starts
```

---

## 💾 9. Memory Architecture

### 3-Tier Model

```
TIER 1: EPHEMERAL (this conversation only)
  ├── Current chat messages
  ├── Current topic context (from RAG)
  ├── Current emotional signals
  ├── Current hint cascade state
  └── Discarded when session ends
         ↓ (session end → auto-summary generated)

TIER 2: SESSION (across conversations)
  ├── Compressed summaries of past sessions
  ├── Key decisions & breakthroughs
  ├── Quiz results with hint levels
  ├── Hint history per topic
  ├── Student notes (text + image descriptions)
  └── Guest: browser localStorage
      Logged-in: cloud database
         ↓ (periodic consolidation)

TIER 3: PERMANENT (Student Genome)
  ├── Knowledge Map (topic mastery scores)
  ├── Learning Profile (observed + selected)
  ├── Misconception registry
  ├── Hint independence trends per topic
  ├── Long-term progress trajectory
  └── Only for LOGGED-IN users
```

### Hint History Storage (within Tier 2)

```
hint_history: {
  "নিউটনের ২য় সূত্র": {
    log: [
      {date: "Aug 15", level: 3, correct: true},
      {date: "Aug 17", level: 1, correct: true},
      {date: "Aug 19", level: 0, correct: true},
    ],
    trend: "improving"
    → Tutor: "গতবার hint লাগছিল, আজকে নিজেই পারলে!"
  },
  "আলোর প্রতিসরণ": {
    log: [
      {date: "Aug 16", level: 3, correct: false},
      {date: "Aug 18", level: 3, correct: true},
    ],
    trend: "stuck"
    → Action: switch explanation style, try different approach
  }
}
```

### Guest vs Logged-In Experience

| Feature | Guest | Logged-In |
|---|---|---|
| Chat with tutor | ✅ | ✅ |
| Hint cascade | ✅ (default levels) | ✅ (genome-personalized) |
| Same-session memory | ✅ | ✅ |
| Cross-session memory | ⚠️ Browser only | ✅ Cloud persistent |
| Student Genome | Basic (level only) | Full (5 layers) |
| Quiz history + hint tracking | Current session | All-time |
| Progress dashboard | ❌ | ✅ Full |
| Note system | ❌ | ✅ |
| Misconception tracking | ❌ | ✅ |
| Interrupt-to-Assess | Basic frequency | Personalized (genome-driven) |

Design philosophy: Guest = "এটা তো ভালো!" → Login করলে "OMG, এটা আমাকে চেনে!"

---

## 📝 10. Note System

### 3 Input Modes

```
Mode 1: TEXT PASTE
  → Chat এ type: "আমার নোট: বল = ভর × ত্বরণ"
  → System detect করে (keyword: "নোট", "note", "মনে রাখতে চাই")
  → Store in Genome Layer 5

Mode 2: IMAGE UPLOAD
  → হাতে লেখা নোটের ছবি upload
  → Gemini Vision → text extraction + accuracy check
  → ভুল note থাকলে misconception flag
  → Genome Layer 5 + Layer 2 update

Mode 3: BUILT-IN EDITOR (future feature)
  → App এ simple note editor
  → Topic-tagged, searchable
  → Tutor reference: "তুমি তো নিজেই লিখেছিলে..."
```

### Note Impact

| Note Content | System Action |
|---|---|
| Correct note | Mastery ↑ for that topic |
| Partially correct | Gentle correction next interaction |
| Wrong note (misconception) | Flag in genome → proactive correction |
| Detailed/organized | Learning profile: "written learner" |

Notes = evidence of understanding, not just storage.

---

## 📊 11. Progress System

### Mastery Formula (Hint-Weighted)

```
mastery(topic) = weighted_average(
    quiz_score(topic, hint_adjusted)  × 0.30   # Hard evidence
    conversation_signals              × 0.20   # Soft evidence
    recall_probe_success              × 0.20   # Retention evidence
    hint_independence_trend           × 0.20   # Real learning signal ← KEY
    note_accuracy(topic)              × 0.10   # Self-study evidence
)
```

**hint_independence_trend** = same topic এ সময়ের সাথে কি কম hint লাগছে?
L3 → L1 → unaided = real learning happening.
This is the most reliable signal of actual knowledge acquisition.

### Mastery Visualization

```
Chapter 2: গতি                    ████████░░░░ 67%
  ├── ২.১ স্থিতি ও গতি              ██████████░░ 83%  (unaided ✓)
  ├── ২.২ সরণ                       █████████░░░ 75%  (L1 hints)
  ├── ২.৩ দ্রুতি ও বেগ               █████░░░░░░░ 42%  ⚠️ (L2-3)
  ├── ২.৪ গতির সমীকরণ               ████████████ 100% ✓ (unaided)
  ├── ২.৫ পড়ন্ত বস্তু               ██████░░░░░░ 50%  (improving↑)
  └── ২.৬ লেখচিত্র                  ███░░░░░░░░░ 25%  🔴 (stuck)

Evidence: Quiz 4/6 | Signals 8✓ 3✗ | Recall 2/3 | Hint trend: ↑
```

### Mastery Decay (Forgetting Curve)

```
mastery_current = mastery_last × decay_factor(days_since_last)
decay_factor(d) = max(0.3, e^(-0.05 × d))

   7 days  → 70% retained
  14 days  → 50% retained
  30 days  → 30% retained (floor — never goes to zero)
```

Decay triggers Recall Probes through the Interrupt Engine.

---

## 🗺️ 12. Complete Routing Map

Step-by-step decision tree — student message আসলে system কী করে:

```
STEP 1: CHECK IN-CHAT OVERRIDE
  ├── "সরাসরি বলো"  → Skip all, direct answer
  ├── "hint দাও"    → Force Practice Mode
  ├── "quiz দাও"    → Force Assessment Strategy
  └── None          → Continue to Step 2

STEP 2: DETECT QUESTION INTENT → SELECT FUNDAMENTAL MODE
  ├── Learning/Concept question  → 🎓 TEACH MODE
  ├── Problem/Calculation        → 🧩 PRACTICE MODE
  ├── Assessment request         → 🧩 PRACTICE MODE
  │
  │ Guidance Level Override:
  ├── "সরাসরি শেখাও" selected → bias toward TEACH
  ├── "নিজে করি" selected     → bias toward PRACTICE
  └── "স্বয়ংক্রিয়"           → pure intent detection

STEP 3: SELECT STRATEGY (Genome-driven)
  ├── New topic (mastery=0)           → 🏗️ Scaffolding
  ├── 3+ understood, mastery>0.7      → 🏋️ Challenge
  ├── 2+ confused, mastery<0.3        → 🛟 Rescue
  ├── 5+ exchanges same topic         → 📝 Assessment
  ├── Returning after 2+ days         → 🔁 Recall
  └── "কেন?" type curiosity          → 🔬 Exploration

STEP 4: DETERMINE HINT LEVEL (Practice Mode only)
  ├── Guidance="নিজে করি"      → Always L0
  ├── Guidance="সরাসরি শেখাও"  → Always L3-L4
  ├── Guidance="সুযোগ দাও"     → Genome-based
  ├── Guidance="স্বয়ংক্রিয়"   → Genome-based
  │   ├── mastery>0.7           → L0
  │   ├── mastery 0.4-0.7      → L1
  │   ├── mastery<0.4          → L2
  │   └── frustrated           → L3
  └── Strategy override:
      ├── Challenge strategy    → L0 (override lower)
      ├── Rescue strategy       → L3 (override higher)
      └── Others               → Use genome default

STEP 5: APPLY STUDENT PREFERENCES
  ├── Explanation Style: উপমা/সূত্র/উদাহরণ/ধাপে ধাপে
  │   (if স্বয়ংক্রিয় → Genome.effective_style)
  │   → HOW to structure the explanation/hint
  ├── Guidance Level: সরাসরি শেখাও/সুযোগ দাও/নিজে করি
  │   (if স্বয়ংক্রিয় → context-based)
  │   → Already applied in STEP 2 (mode bias) & STEP 4 (hint level)
  │   → Here: final check on response directness
  ├── Personality: বন্ধুসুলভ/আনুষ্ঠানিক/গল্পকার/প্রতিযোগী
  │   (if স্বয়ংক্রিয় → Genome.observed_personality)
  │   → TONE of the response
  └── All 3 apply to both TEACH and PRACTICE modes

STEP 6: CHECK INTERRUPT ENGINE
  ├── Should we interrupt with a check?
  ├── Frequency rules pass?
  ├── YES → inject Quick Check / Recall / Challenge / Misconception
  └── NO  → continue with normal response

STEP 7: GENERATE RESPONSE
  → All parameters combined → LLM generates response
  → Response streamed to student

STEP 8: POST-RESPONSE (after student replies)
  ├── Detect comprehension signals
  ├── Evaluate answer (if quiz/hint was given)
  ├── Update Genome (mastery, signals, hint history)
  ├── Update Memory (session log)
  └── Influence next interaction → [back to Step 1]
```

---

## 🖥️ 13. Product Form

### Phase 1: Next.js Progressive Web App (PWA)

| Why Next.js PWA | Benefit |
|---|---|
| Works on all devices | Phone browser = 90% BD students |
| PWA installable | App-like experience, no app store |
| NextAuth | Authentication ready |
| Prisma + PostgreSQL | Database for Genome/Memory |
| Vercel | Good free tier deployment |
| Future: Capacitor | Wrap for app stores when needed |

### Phase 2 (Future): Native Mobile App
### Phase 3 (Future): React Native / Flutter cross-platform

---

## 📋 14. Decisions Log

All confirmed decisions from our brainstorming sessions:

| # | Decision | Date | Context |
|---|---|---|---|
| 1 | Target: Eventually all subjects, start with Physics | 2026-08-19 | Founder decision |
| 2 | Tutor mood = context-aware auto switching + student personality selection | 2026-08-19 | Both dimensions needed |
| 3 | Note sharing: all 3 modes (text, image, built-in editor) | 2026-08-19 | Founder decision |
| 4 | Product form: TBD, recommendation is Next.js PWA | 2026-08-19 | Needs final decision |
| 5 | Authentication: Guest + Login (guest=basic, login=full features) | 2026-08-19 | Founder decision |
| 6 | Business model: Initially free → freemium/subscription | 2026-08-19 | Founder decision |
| 7 | Curriculum: Open — support any curriculum | 2026-08-19 | Founder decision |
| 8 | Hint ≠ toggle, Hint = Practice Mode engine | 2026-08-19 | Design decision |
| 9 | Teach Mode (concepts) vs Practice Mode (problems) — intent-based auto | 2026-08-19 | Design correction |
| 10 | Student CAN control modes — not forced | 2026-08-19 | User feedback |
| 11 | All 3 controls default to "স্বয়ংক্রিয়" — zero-config UX | 2026-08-19 | UX decision |
| 12 | Explanation Style (উপমা/সূত্র/etc) is separate from Guidance Level | 2026-08-19 | Design correction |
| 13 | Controls are auto-selected, student goes to settings only if they want to change | 2026-08-19 | UX decision |
| 14 | Multi-subject: content pipeline আলাদা, Genome + Tutor Intelligence shared | 2026-08-31 | Architecture decision |
| 15 | Cost optimization: MVP plan এর সময় decide করবো | 2026-08-31 | Deferred |
| 16 | **No Parent/Teacher Dashboard** — students secure feel করবে না if parent monitors | 2026-08-31 | Founder decision |
| 17 | Offline support: No — online only for now | 2026-08-31 | Founder decision |
| 18 | Content Pipeline: পরে plan, similar approach (PDF → process → ready) | 2026-08-31 | Deferred |
| 19 | Quiz Bank strategy: পরে plan করবো | 2026-08-31 | Deferred |
| 20 | Gamification: subtle, not distracting — learning > gaming | 2026-08-31 | Agreed |
| 21 | Language: Adaptive — English→English, Bangla→Bangla, Banglish→Bangla | 2026-08-31 | Founder decision |
| 22 | Tutor Personality: 4 styles — বন্ধুসুলভ, আনুষ্ঠানিক, গল্পকার, প্রতিযোগী | 2026-08-31 | Restored from V1 |

---

## ❓ 15. Open Questions

### ✅ Resolved (from Aug 31 discussion)

| Q# | Question | Resolution |
|---|---|---|
| 1 | Multi-subject expansion | ✅ Content pipeline আলাদা, Genome + Tutor Intelligence shared (Decision #14) |
| 2 | Cost optimization | ⏸️ MVP planning এর সময় decide (Decision #15) |
| 3 | Teacher/Parent Dashboard | ❌ NO — student privacy priority, secure feel করবে না (Decision #16) |
| 4 | Offline support | ❌ Online only for now (Decision #17) |
| 5 | Content Pipeline | ⏸️ পরে, similar approach (Decision #18) |
| 6 | Quiz Bank strategy | ⏸️ পরে plan করবো (Decision #19) |
| 7 | Gamification | ✅ Subtle, not distracting — learning > gaming (Decision #20) |
| 8 | Language | ✅ Adaptive — English→English, Bangla→Bangla, Banglish→Bangla (Decision #21) |

### 🔓 New Open Questions (for next discussion)

1. **MVP Scope**: Phase 1 এ কোন কোন feature include হবে? পুরো system একসাথে না — কোন pillar first?

2. **Database Schema**: Student Genome, Memory, Quiz History কীভাবে store হবে? (PostgreSQL schema design needed)

3. **System Prompt Architecture**: Routing Map কীভাবে LLM system prompt এ translate হবে? (One mega prompt vs modular prompts?)

4. **RAG Integration**: Current physics-tutor-agent-v2 এর retrieval engine (LanceDB + embeddings) কীভাবে নতুন system এ integrate হবে?

5. **Real-time Genome Updates**: Conversation চলাকালীন Genome কীভাবে update হবে? (Every message? Batch at session end? Hybrid?)

---

> **Status: BRAINSTORMING PHASE — ALL ORIGINAL OPEN QUESTIONS RESOLVED**
> Next: MVP scope definition → Technical Architecture → Implementation
