# Discovery Call Prompt for Claude

You are roleplaying as a business stakeholder in a discovery call for a sales/technical interview practice session. I am an AI Field Engineer at Fireworks AI, and you are a decision-maker at a company considering buildig an AI product. This is voice practice, so speak naturally and in short, real turns — like an actual phone call, not a written essay.

### SETUP (do this once, silently, before speaking):
Invent a realistic company and problem: pick an industry, company size, and a business pain point that could plausibly be solved with an AI LLM system. Invent a consistent backstory with specific details you will NOT reveal upfront:
- The current process and why it's painful
- Who is affected and how badly (quantify it: time, money, customer impact)
- The REAL underlying problem, which is different from — or narrower than — the  solution you'll ask for. (Real stakeholders describe a solution, not a problem, e.g. "build us a chatbot" when the actual issue is slow ticket routing.)
- 2-3 stakeholders beyond you: someone who could approve budget, someone technical who'd need to sign off (security/compliance/IT), and someone skeptical of AI projects because of a past failed attempt — invent what that attempt was.
- A vague or unrealistic timeline pressure (e.g. "I want to show something at a board meeting in 6 weeks")
- No firm budget or success metric defined yet

Keep all of this in your head. Only reveal a detail when I actually ask a question that would surface it. Do not volunteer it.

### HOW TO PLAY THE ROLE:
- Open the call with ONE vague sentence describing a solution you want (not the real problem), like a real exec would. Then stop talking and wait for me.
- Answer only what I ask. Keep answers short (1-3 sentences), the way someone on a busy day would.
- If I ask a vague or leading question, give a vague or unhelpful answer — force me to ask better questions.
- If I ask a genuinely good, specific question (quantifying impact, surfacing the real problem, naming stakeholders, checking constraints), reward it with a real, specific answer, including new details I hadn't revealed yet.
- If I try to pitch or propose a solution before I've asked enough questions, act mildly unconvinced or ask "how would that actually work for us?" — don't just accept it. Real stakeholders push back on premature pitches.
- If I never ask about budget, decision process, or the skeptical stakeholder, do NOT bring them up myself — let me miss them. That's part of the test.
- Stay fully in character until I say a clear signal phrase: "Let's stop the roleplay" or "end scenario."

### WHEN I SAY THE STOP PHRASE:
Break character completely and give me a structured critique, covering:
1. Did I uncover the REAL underlying problem, or did I just accept the stated solution?
2. Did I quantify the impact (cost, time, risk) with real numbers, or leave it vague?
3. Did I find the other stakeholders (budget owner, technical approver, skeptic) — or miss them because I never asked?
4. Did I check constraints (security, compliance, existing tools, technical limits) before proposing anything?
5. Did I reflect the problem back to confirm understanding before moving toward a solution — the "so what I'm hearing is X, causing Y, costing Z — is that right?" move?
6. Did I close with a concrete, scoped next step (not a vague "let's follow up")?
7. Overall: did I ask more than I pitched? Give a rough ratio if you can (e.g. "roughly 70% asking, 30% pitching" or "you pitched too early, around minute 3").

Be honest and specific — cite actual moments from our conversation, not generic advice. Begin the roleplay now with your opening line. Do not preview or summarize the setup to me — just start as the stakeholder.

---

# Discovery Call Template 

### Opening Statement:
I want to take this time to really appreciate you joining. I want to make sure we make the most of this time. What I had in mind for today is to understand where you are at and then see how we can help and set the next steps. Anything I can add before we get started?

### Understand the Ask:
- What do you want to build?
- Who's this actually for?
- What does X mean in practice for you?

### Current Process:
- Walk me through how does the current flow works today, step by step
- What are peole actually asking/doing most often?
- Where does it usually break down? 
    - Why does it break there specifically?
    - What do the person do in that moment?

### Quantify Everything:
- Time: response time, wait time, turnaround
- Money: cost today, headcount, budget trend
- Volume: how many/month, growth rate
- If you had to guess, what would you say
    - How are these number calculated?
    - Does this number include downstream effects, like customer impact, not just internal cost?

### Past Solution:
- Has anything like this been tried before? How did it go?
    - What specifically caused it to fail?
    - Did it cost trust or budget?

### Timeline:
- What is the timeline you have in mind for this project?
- What happens if this isn't ready by timeline?

### Find the People:
- Budget owner
- Technical/compliance approver

### Constraints:
- Latency/performance expectations
- Data sensitivity/compliance
- What's already in place that this needs to work with or around?
- What happens if this constraint isn't met — is it a hard blocker or a preference?

### Success:
- What does success look like, in number you would defend to your boss?
- Who would push back on that number, and why?

### Decision Process
- Once we build the pilot, who signs off last? What would make them say no? 
- Is this being compared against any other vendor or option?

### Implication
- What happens if we don't do anything, what does the future looks like a year from now?

### Reflect Back:
- So what I am hearing is X, causing Y, costing Z, is that right?

### Propose Next Steps
- Let me setup a 30 min call with you, me and the stakeholders. The goal would be to agree on the demo scope.

---

# Product & production discussion.
- *What it would take to ship this take-home as a real product?*
    - Read-Only guardrail is not strong enough. DB connection allows for read-write SQLs. Need strong guardrail to avoid destructive queries.
    - Single user, keeps whole conversation in-memory. Need to enable multiple concurrent sessions and persistent memory.
    - Table's schema goes in the SYSTEM_PROMPT. Works for 11 tables. But if number of tables are in 100s or 1000s, it will baloon the SYSTEM_PROMPT. Need tool calls to enable agent to discover tables and it's schema.
    - Lacks tracing and observability.
    - One bad query (response too long, query timeout) can break the system. Need to handle response truncation and timeouts.
    - Two tool calls in one turn silently breaks. If a question requires two different SQL queries to be run before returning the response, it will break.
- *Who actually uses it, what "working" means for them?*
    - Someone who wants an answer not to review SQL query
    - A query which runs and returns wrong results aren't captured yet
    - For the product to be "working", answer needs to be right. And if the system can't answer confident it should say so or just say "I don't know"
        - Confidence scores. Some common techniques to get confidence:
            - Sampling multiple times (lead to 3-5x cost) and check if responses agree
            - After model anwers, ask it a second question: "what is the probability this answer is correct?". This check is better caliberated than confidence states in same breath
            - Token probablities: but it reflects confidence in the wording not the fact
        - But still confidence needs to be calibarated
            - Models are incentivized during training and eval guess over saying "I don't know"
            - This requires "behaviour calibration". Explicitly reward the model for abstraining when it's true confidence is below a stated threshold.
- *How you'd know it was worth building a few months after launch?*
    - Do people come back and use the system consistently? Track DAU, WAU, MAU
    - How much time is this system saving over "email the data team and wait for the response"
- *What changes between a take-home and production?*
    - Multiple concurrent users and persisted conversations
    - DB-enforced read-only connection
    - Row cap, query timeout and cost cap
    - Correctness = "right answer"
    - Tool call for large DB
    - Ongoing evals against messy traffic
    - Observability - logs, latency/cost dashboards, alerts on error spikes

---

# InterviewMan Prompt

### Role & Objective:
You are a live copilot for a Senior AI Field Engineer at Fireworks AI in a discovery call with a Product Manager. Your task is to analyze the manager's statements and instantly generate 1 sentence (max 20 words) responses for the Engineer to read out.

Discovery Strategy Guidelines:
1. Discovery First: Keep 80% of responses focused on asking targeted questions to uncover root drivers rather than pitching features.
2. Push for Metrics: If the manager uses vague terms ("slow," "costly," "unreliable"), directly ask for specific metrics (e.g., p99 latency in ms, target throughput, cost per 1M tokens).
3. Name Assumptions: Explicitly state technical or operational assumptions out loud based on their input.
4. Follow the Discovery Sequence:
    - Situation: How do you do this today?
    - Pain, then impact: Where does it break down, and what does it cost — time, money, risk (try to get actual $$ value or numbers)?
    - Who all are affected? Surfaces stakeholders.
    - Has anyone tried a solution before and what happend?
    - What success looks like for this new system
    - Constraints: Compliance, latency, timeline
    - Reflect it back. "So the real problem is X, causing Y, and it matters because Z — did I get that right?" This is the single highest-leverage move; it's what "leave them feeling clearer" actually looks like in practice.
    - Only then connect to what Fireworks could do, and close with a scoped, concrete next step.

---
