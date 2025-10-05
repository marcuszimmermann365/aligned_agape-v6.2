
def apply(lines, ctx, stage3=True):
    # Minimal Socratic: ensure at least one bridge-building line
    if stage3 and lines:
        lines.append("Moderation: Könnten wir auf Verifikation & Sequenzierung einigen, um Vertrauen aufzubauen?")
    return lines
