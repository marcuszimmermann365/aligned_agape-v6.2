
import numpy as np

class PersonaAgent:
    def __init__(self, actor_spec, passages):
        self.id = actor_spec["id"]
        self.name = actor_spec["name"]
        self.role = actor_spec.get("role")
        self.core_interests = actor_spec.get("core_interests", [])
        self.style = (passages[0]["text"] if passages else "")[:80]
        self.hidden = np.random.randn(8).astype(float)
        self._tick = 0
    def policy(self, obs):
        self._tick += 1
        self.last_action_vec = (np.tanh(self.hidden) + 0.05*np.random.randn(*self.hidden.shape)).astype(float)
        return f"Wir schlagen Sequenzierung und Verifikation vor (tick {self._tick})."
    def decoder(self, H):
        return H  # identity decoder for proxy
    def observe_next(self):
        return (self.hidden + 0.1*np.random.randn(*self.hidden.shape)).astype(float)
    def observe_cond(self):
        return (np.roll(self.hidden, 1)).astype(float)
