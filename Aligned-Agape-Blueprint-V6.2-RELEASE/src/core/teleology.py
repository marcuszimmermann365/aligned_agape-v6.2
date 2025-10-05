
class Teleology:
    def __init__(self, scenario):
        self.series_J4 = []; self.series_SCM = []
        self.milestones = scenario.get("milestones", [])
        self.milestone_index = 0
        self.milestone_total = len(self.milestones)
    def update_series(self, J4, SCM):
        if isinstance(J4, list): self.series_J4 = J4
        if isinstance(SCM, list): self.series_SCM = SCM
    def is_ripe(self, J1,J2,J3,J4,J5,SCM, window=2):
        if len(self.series_J4) < window+1 or len(self.series_SCM) < window+1:
            return False
        dJ4 = self.series_J4[-1] - self.series_J4[-2]
        dSCM = self.series_SCM[-1] - self.series_SCM[-2]
        return (dJ4 > 0 and dSCM > 0)
    def readiness(self, J1,J2,J3,J4,J5,SCM):
        return max(0.0, min(1.0, 0.5*SCM + 0.1*J4))
    def current_milestone(self):
        if self.milestone_index < self.milestone_total:
            return self.milestones[self.milestone_index]
        return "—"
