import re

from skills import SKILLS


def extract_skills(text):
    """Return dictionary skills found in text, preferring specific matches."""
    if not text:
        return []

    text_lower = str(text).lower()
    matches = []

    # Check longer skill names first so C++ is preferred over C,
    # and Linear Regression is preferred over Regression when spans overlap.
    for skill in sorted(SKILLS, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        for match in re.finditer(pattern, text_lower):
            start, end = match.span()

            # Skip this skill when its text span overlaps a previously
            # accepted, more-specific skill.
            overlaps = any(
                not (end <= existing_start or start >= existing_end)
                for existing_start, existing_end, _ in matches
            )

            if not overlaps:
                matches.append((start, end, skill))

    found_skills = {skill for _, _, skill in matches}
    return sorted(found_skills)
