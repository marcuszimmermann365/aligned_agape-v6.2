
class Rulebook:
    def __init__(self, scenario):
        self.scenario = scenario
        self.red = {r["actor"]: set(r["prohibits"]) for r in scenario.get("red_lines",[])}
    def plausibility_mask(self, actor_id, ctx, text):
        # naive block if text contains prohibited token
        for tok in self.red.get(actor_id, set()):
            if tok.split("_")[0].lower() in text.lower():
                return False, f"violates red line: {tok}"
        return True, ""
