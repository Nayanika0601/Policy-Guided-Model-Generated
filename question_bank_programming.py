"""
Beginner programming question bank for the tutoring study.

Each stored question has only:
- concept
- question
- answer
- accepted_answers
"""


QUESTION_BANK = [
    # variables
    {"concept": "variables", "question": "What do we call a named place to store data?", "answer": "variable", "accepted_answers": ["variable", "variables"]},
    {"concept": "variables", "question": "What does a variable store?", "answer": "value", "accepted_answers": ["value", "values"]},
    {"concept": "variables", "question": "What do we call the label of a variable?", "answer": "name", "accepted_answers": ["name", "names"]},
    {"concept": "variables", "question": "What action puts a value into a variable?", "answer": "assign", "accepted_answers": ["assign", "assignment"]},
    {"concept": "variables", "question": "What symbol is used to assign a value?", "answer": "equals", "accepted_answers": ["equals", "equal", "="]},
    {"concept": "variables", "question": "What function asks the user for text?", "answer": "input", "accepted_answers": ["input", "inputs"]},
    {"concept": "variables", "question": "What function shows something on the screen?", "answer": "print", "accepted_answers": ["print", "prints"]},
    {"concept": "variables", "question": "What kind of value is five?", "answer": "number", "accepted_answers": ["number", "numbers"]},
    {"concept": "variables", "question": "What kind of value is hello?", "answer": "text", "accepted_answers": ["text"]},
    {"concept": "variables", "question": "What word means keep a value for later?", "answer": "store", "accepted_answers": ["store", "stored"]},
    {"concept": "variables", "question": "What can change while a program runs?", "answer": "variable", "accepted_answers": ["variable", "variables"]},
    {"concept": "variables", "question": "What should a variable name describe?", "answer": "value", "accepted_answers": ["value", "values"]},
    {"concept": "variables", "question": "What function can display a variable?", "answer": "print", "accepted_answers": ["print", "prints"]},
    {"concept": "variables", "question": "What function can save typed user text?", "answer": "input", "accepted_answers": ["input", "inputs"]},
    {"concept": "variables", "question": "What word means give a variable a value?", "answer": "assign", "accepted_answers": ["assign", "assignment"]},

    # data_types
    {"concept": "data_types", "question": "What type stores text?", "answer": "string", "accepted_answers": ["string", "str"]},
    {"concept": "data_types", "question": "What type stores whole numbers?", "answer": "integer", "accepted_answers": ["integer", "int"]},
    {"concept": "data_types", "question": "What type stores decimal numbers?", "answer": "float", "accepted_answers": ["float"]},
    {"concept": "data_types", "question": "What type stores true or false?", "answer": "boolean", "accepted_answers": ["boolean", "bool"]},
    {"concept": "data_types", "question": "What type stores several items in order?", "answer": "list", "accepted_answers": ["list", "lists"]},
    {"concept": "data_types", "question": "What type stores key value pairs?", "answer": "dictionary", "accepted_answers": ["dictionary", "dict"]},
    {"concept": "data_types", "question": "What type stores items that should not change?", "answer": "tuple", "accepted_answers": ["tuple", "tuples"]},
    {"concept": "data_types", "question": "What type stores unique items?", "answer": "set", "accepted_answers": ["set"]},
    {"concept": "data_types", "question": "What function tells a value's type?", "answer": "type", "accepted_answers": ["type"]},
    {"concept": "data_types", "question": "What value means yes in a boolean?", "answer": "true", "accepted_answers": ["true", "truth"]},
    {"concept": "data_types", "question": "What value means no in a boolean?", "answer": "false", "accepted_answers": ["false", "falls"]},
    {"concept": "data_types", "question": "What type would store a person's name?", "answer": "string", "accepted_answers": ["string", "str"]},
    {"concept": "data_types", "question": "What type would store someone's age?", "answer": "integer", "accepted_answers": ["integer", "int"]},
    {"concept": "data_types", "question": "What type would store a price like two point five?", "answer": "float", "accepted_answers": ["float"]},
    {"concept": "data_types", "question": "What type can hold many names?", "answer": "list", "accepted_answers": ["list", "lists"]},

    # loops
    {"concept": "loops", "question": "What keyword starts a counting loop?", "answer": "for", "accepted_answers": ["for", "four", "fore"]},
    {"concept": "loops", "question": "What keyword repeats while something is true?", "answer": "while", "accepted_answers": ["while", "wild"]},
    {"concept": "loops", "question": "What do we call code that repeats?", "answer": "loop", "accepted_answers": ["loop", "loops"]},
    {"concept": "loops", "question": "What function gives numbers for a loop?", "answer": "range", "accepted_answers": ["range"]},
    {"concept": "loops", "question": "What keyword stops a loop early?", "answer": "break", "accepted_answers": ["break", "brake"]},
    {"concept": "loops", "question": "What keyword skips to the next repeat?", "answer": "continue", "accepted_answers": ["continue", "continued"]},
    {"concept": "loops", "question": "What word means do something again?", "answer": "repeat", "accepted_answers": ["repeat", "repeats"]},
    {"concept": "loops", "question": "What word means keep track of numbers?", "answer": "count", "accepted_answers": ["count", "counts"]},
    {"concept": "loops", "question": "What word means end the repeating?", "answer": "stop", "accepted_answers": ["stop", "stops"]},
    {"concept": "loops", "question": "What word means go through items one by one?", "answer": "iterate", "accepted_answers": ["iterate", "iterates"]},
    {"concept": "loops", "question": "What keyword would repeat over a list?", "answer": "for", "accepted_answers": ["for", "four", "fore"]},
    {"concept": "loops", "question": "What keyword repeats until a condition changes?", "answer": "while", "accepted_answers": ["while", "wild"]},
    {"concept": "loops", "question": "What keyword exits a loop immediately?", "answer": "break", "accepted_answers": ["break", "brake"]},
    {"concept": "loops", "question": "What keyword moves to the next loop step?", "answer": "continue", "accepted_answers": ["continue", "continued"]},
    {"concept": "loops", "question": "What function is often used with for?", "answer": "range", "accepted_answers": ["range"]},

    # conditionals
    {"concept": "conditionals", "question": "What keyword starts a condition?", "answer": "if", "accepted_answers": ["if"]},
    {"concept": "conditionals", "question": "What keyword handles the other case?", "answer": "else", "accepted_answers": ["else"]},
    {"concept": "conditionals", "question": "What keyword means else if?", "answer": "elif", "accepted_answers": ["elif", "else if", "alif"]},
    {"concept": "conditionals", "question": "What word describes a test in an if statement?", "answer": "condition", "accepted_answers": ["condition", "conditions"]},
    {"concept": "conditionals", "question": "What value means a condition passes?", "answer": "true", "accepted_answers": ["true", "truth"]},
    {"concept": "conditionals", "question": "What value means a condition fails?", "answer": "false", "accepted_answers": ["false", "falls"]},
    {"concept": "conditionals", "question": "What word can mean equal in a condition?", "answer": "equals", "accepted_answers": ["equals", "equal", "double equals", "=="]},
    {"concept": "conditionals", "question": "What word means check two values?", "answer": "compare", "accepted_answers": ["compare", "compares"]},
    {"concept": "conditionals", "question": "What operator needs both conditions true?", "answer": "and", "accepted_answers": ["and"]},
    {"concept": "conditionals", "question": "What operator needs one condition true?", "answer": "or", "accepted_answers": ["or"]},
    {"concept": "conditionals", "question": "What operator reverses true and false?", "answer": "not", "accepted_answers": ["not"]},
    {"concept": "conditionals", "question": "What keyword runs code only when a test passes?", "answer": "if", "accepted_answers": ["if"]},
    {"concept": "conditionals", "question": "What keyword gives a fallback choice?", "answer": "else", "accepted_answers": ["else"]},
    {"concept": "conditionals", "question": "What value does a comparison produce?", "answer": "boolean", "accepted_answers": ["boolean", "bool"]},
    {"concept": "conditionals", "question": "What word means choose between paths?", "answer": "condition", "accepted_answers": ["condition", "conditions"]},
]


CONCEPTS = tuple(dict.fromkeys(q["concept"] for q in QUESTION_BANK))


def get_all_questions() -> list[dict]:
    return list(QUESTION_BANK)


def get_by_concept(concept: str) -> list[dict]:
    return [q for q in QUESTION_BANK if q["concept"] == concept]


def get_by_level(level: int) -> list[dict]:
    """Backward-compatible helper for older imports."""
    return get_all_questions()


def validate_bank() -> None:
    required = {"concept", "question", "answer", "accepted_answers"}
    expected_concepts = ("variables", "data_types", "loops", "conditionals")
    banned_answers = {
        "nonlocal",
        "unpacking",
        "enumerate",
        "zip",
        "nested",
        "membership",
        "lambda",
        "comprehension",
        "recursion",
        "scope",
        "iterable",
        "iterator",
    }

    assert len(QUESTION_BANK) == 60, f"Expected 60 questions, found {len(QUESTION_BANK)}"
    assert set(CONCEPTS) == set(expected_concepts), f"Unexpected concepts: {CONCEPTS}"

    for concept in expected_concepts:
        count = len(get_by_concept(concept))
        assert count == 15, f"Expected 15 questions for {concept}, found {count}"

    for i, q in enumerate(QUESTION_BANK, 1):
        assert set(q.keys()) == required, f"Question {i} has wrong keys: {q.keys()}"
        assert q["concept"] in expected_concepts, f"Question {i} has bad concept"
        assert q["answer"] not in banned_answers, f"Question {i} uses banned answer"
        assert isinstance(q["accepted_answers"], list) and q["accepted_answers"], (
            f"Question {i} needs accepted answers"
        )
        assert q["answer"] in q["accepted_answers"], (
            f"Question {i} answer must be accepted"
        )


if __name__ == "__main__":
    validate_bank()
    print(f"Programming question bank loaded: {len(QUESTION_BANK)} total")
    for concept in ("variables", "data_types", "loops", "conditionals"):
        print(f"  {concept}: {len(get_by_concept(concept))}")
    print("All question bank tests passed.")
