"""
Prompt template management for the evaluation harness.

Provides prompt templates per pragmatic category (sarcasm, implicature,
indirect speech acts) with support for zero-shot and few-shot variants.

Two template versions per category are provided to measure prompt sensitivity
(Section 10 of the execution guide: "run at least 2 paraphrased prompt
templates per model and report the spread").
"""

from typing import Any

# ---------------------------------------------------------------------------
# Sarcasm prompt templates
# ---------------------------------------------------------------------------

SARCASM_ZERO_SHOT_V1 = """You will be given a short context and an utterance. Decide whether the utterance is SINCERE or SARCASTIC given the context. Respond with exactly one word: "sincere" or "sarcastic".

Context: {context}
Utterance: "{utterance}"
Answer:"""

SARCASM_ZERO_SHOT_V2 = """Read the following context and utterance carefully. Is the speaker being sarcastic, or do they genuinely mean what they say? Reply with only one word: "sincere" or "sarcastic".

Context: {context}
Utterance: "{utterance}"
Your answer:"""

SARCASM_FEW_SHOT_V1 = """You will be given a short context and an utterance. Decide whether the utterance is SINCERE or SARCASTIC given the context. Respond with exactly one word: "sincere" or "sarcastic".

Example 1:
Context: A student gets a perfect score on the exam.
Utterance: "You really aced that one."
Answer: sincere

Example 2:
Context: It has been raining non-stop for a week.
Utterance: "What lovely weather we're having."
Answer: sarcastic

Example 3:
Context: A chef prepares an excellent five-course meal.
Utterance: "This is absolutely delicious."
Answer: sincere

Now evaluate:
Context: {context}
Utterance: "{utterance}"
Answer:"""

SARCASM_FEW_SHOT_V2 = """Below are some labelled examples, then a new case for you to judge. For each utterance, write only "sincere" or "sarcastic" — no other words.

[Example A]
Situation: A marathon runner finishes the race in record time.
They said: "That was an easy stroll."
Label: sarcastic

[Example B]
Situation: A friend helps you move to a new apartment over the weekend.
They said: "I'm happy to help anytime."
Label: sincere

[Example C]
Situation: A presentation crashes the projector mid-way through.
They said: "Well, that went perfectly."
Label: sarcastic

[New case]
Situation: {context}
They said: "{utterance}"
Label:"""

# ---------------------------------------------------------------------------
# Implicature prompt templates
# ---------------------------------------------------------------------------

IMPLICATURE_ZERO_SHOT_V1 = """You will be given a short exchange between two people (A and B). Based on B's reply, decide what B is implying is the answer to A's question: "yes", "no", or "uncertain". Respond with exactly one word.

{context}
B: "{utterance}"
Answer:"""

IMPLICATURE_ZERO_SHOT_V2 = """In the conversation below, A asks a question and B gives an indirect reply. What does B most likely mean? Respond with exactly one word: "yes", "no", or "uncertain".

{context}
B says: "{utterance}"
B's implied answer:"""

IMPLICATURE_FEW_SHOT_V1 = """You will be given a short exchange between two people (A and B). Based on B's reply, decide what B is implying is the answer to A's question: "yes", "no", or "uncertain". Respond with exactly one word.

Example 1:
A: "Did you enjoy the party?"
B: "I left after twenty minutes."
Answer: no

Example 2:
A: "Would you recommend this restaurant?"
B: "I've already been back three times."
Answer: yes

Example 3:
A: "Is the new hire working out?"
B: "Well, he's always on time."
Answer: no

Now evaluate:
{context}
B: "{utterance}"
Answer:"""

IMPLICATURE_FEW_SHOT_V2 = """Study these short dialogues where A asks a question and B gives an indirect reply. For each new dialogue, output exactly one word — "yes", "no", or "uncertain" — reflecting what B really means.

[Dialogue 1]
A: "Did the package arrive?"
B: "I checked the tracking this morning."
Implied answer: uncertain

[Dialogue 2]
A: "Have you eaten?"
B: "I just got back from the cafeteria."
Implied answer: yes

[Dialogue 3]
A: "Did you finish the report?"
B: "I still have two sections to go."
Implied answer: no

[New dialogue]
{context}
B: "{utterance}"
Implied answer:"""

# ---------------------------------------------------------------------------
# Indirect speech act prompt templates
# ---------------------------------------------------------------------------

INDIRECT_ZERO_SHOT_V1 = """You will be given a context and an utterance. Determine the speaker's actual intent. Choose one: "statement", "request", "question", "suggestion", or "complaint". Respond with exactly one word.

Context: {context}
Utterance: "{utterance}"
Intent:"""

INDIRECT_ZERO_SHOT_V2 = """Read the context and utterance below. What is the speaker really trying to do? The utterance may look like one type of speech act but actually function as another. Reply with exactly one word: "statement", "request", "question", "suggestion", or "complaint".

Context: {context}
Speaker says: "{utterance}"
Actual intent:"""

INDIRECT_FEW_SHOT_V1 = """You will be given a context and an utterance. Determine the speaker's actual intent. Choose one: "statement", "request", "question", "suggestion", or "complaint". Respond with exactly one word.

Example 1:
Context: Two coworkers in a cold office.
Utterance: "It's freezing in here."
Intent: request

Example 2:
Context: A weather reporter on TV.
Utterance: "Temperatures will drop below zero tonight."
Intent: statement

Example 3:
Context: A friend looking at restaurant options on their phone.
Utterance: "I hear that Thai place got great reviews."
Intent: suggestion

Now evaluate:
Context: {context}
Utterance: "{utterance}"
Intent:"""

INDIRECT_FEW_SHOT_V2 = """Here are labelled examples of indirect speech acts. The surface form of the utterance may differ from its actual function. Respond with exactly one word: "statement", "request", "question", "suggestion", or "complaint".

[Case 1]
Setting: An employee reviewing a colleague's noisy video call.
What they said: "People can probably hear everything through the walls."
Function: complaint

[Case 2]
Setting: A student talking to a librarian while both look at empty shelves.
What they said: "I wonder if the book has been re-shelved elsewhere."
Function: question

[Case 3]
Setting: Two friends planning a road trip, looking at a map.
What they said: "The coastal route looks really scenic this time of year."
Function: suggestion

[New case]
Setting: {context}
What they said: "{utterance}"
Function:"""


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "sarcasm": {
        "zero_shot_v1": SARCASM_ZERO_SHOT_V1,
        "zero_shot_v2": SARCASM_ZERO_SHOT_V2,
        "few_shot_v1": SARCASM_FEW_SHOT_V1,
        "few_shot_v2": SARCASM_FEW_SHOT_V2,
    },
    "implicature": {
        "zero_shot_v1": IMPLICATURE_ZERO_SHOT_V1,
        "zero_shot_v2": IMPLICATURE_ZERO_SHOT_V2,
        "few_shot_v1": IMPLICATURE_FEW_SHOT_V1,
        "few_shot_v2": IMPLICATURE_FEW_SHOT_V2,
    },
    "indirect_speech_act": {
        "zero_shot_v1": INDIRECT_ZERO_SHOT_V1,
        "zero_shot_v2": INDIRECT_ZERO_SHOT_V2,
        "few_shot_v1": INDIRECT_FEW_SHOT_V1,
        "few_shot_v2": INDIRECT_FEW_SHOT_V2,
    },
}


def format_prompt(
    category: str,
    template_name: str,
    context: str,
    utterance: str,
) -> str:
    """Format a prompt template with the given context and utterance.

    Args:
        category: Pragmatic category (sarcasm, implicature, indirect_speech_act)
        template_name: Template variant (e.g., "zero_shot_v1", "few_shot_v1")
        context: The context/scenario text
        utterance: The target utterance

    Returns:
        Formatted prompt string ready for model input
    """
    if category not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown category: {category}. Valid: {list(PROMPT_TEMPLATES.keys())}")

    templates = PROMPT_TEMPLATES[category]
    if template_name not in templates:
        raise ValueError(f"Unknown template: {template_name}. Valid for {category}: {list(templates.keys())}")

    template = templates[template_name]
    return template.format(context=context, utterance=utterance)


def format_prompt_no_context(
    category: str,
    template_name: str,
    utterance: str,
) -> str:
    """Format a prompt with context stripped — for the context ablation study.

    Replaces the context field with a generic placeholder so the model must
    rely on the utterance alone.
    """
    return format_prompt(
        category=category,
        template_name=template_name,
        context="(No context provided.)",
        utterance=utterance,
    )


def get_valid_labels(category: str) -> list[str]:
    """Return the set of valid output labels for a category."""
    label_map = {
        "sarcasm": ["sincere", "sarcastic"],
        "implicature": ["yes", "no", "uncertain"],
        "indirect_speech_act": ["statement", "request", "question", "suggestion", "complaint"],
    }
    return label_map.get(category, [])


def list_templates() -> None:
    """Print all available prompt templates."""
    for category, templates in PROMPT_TEMPLATES.items():
        print(f"\n{category}:")
        for name in templates:
            print(f"  - {name}")
