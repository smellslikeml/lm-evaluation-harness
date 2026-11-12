LETTERS = "ABCDEFGHIJ"


def doc_to_text(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    ...
    J. <choice10>
    Answer:
    """
    # The dataset's `question` field already appends the answer choices
    # inline as "\nAnswer Choices: (A) ... (J) ..."; strip that tail so the
    # choices are not duplicated in the prompt.
    stem = doc["question"].split("\nAnswer Choices:")[0].rstrip()

    prompt = "Question: " + stem + "\nChoices:\n"
    for letter in LETTERS:
        prompt += f"{letter}. {doc['options'][letter]}\n"
    prompt += "Answer:"
    return prompt


def doc_to_target(doc) -> int:
    return LETTERS.index(doc["label"].strip().upper())
