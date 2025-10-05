
from pathlib import Path
import json, datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

def textblock(c, x, y, text, max_width):
    lines = []
    for para in text.split("\n"):
        current = ""
        for word in para.split(" "):
            try_text = (current + " " + word).strip()
            if c.stringWidth(try_text, "Helvetica", 10) <= max_width:
                current = try_text
            else:
                lines.append(current); current = word
        lines.append(current)
    for ln in lines:
        c.drawString(x, y, ln); y -= 12
    return y

def export_summary():
    last = sorted(OUT.glob("turn_*.json"))[-1]
    data = json.loads(last.read_text(encoding="utf-8"))
    pdf = OUT / "run_summary.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    W, H = A4
    x, y = 2*cm, H-2*cm
    c.setFont("Helvetica-Bold", 14); c.drawString(x, y, "Geopolitical Training Simulator – Run Summary"); y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(x, y, "Generated: " + datetime.datetime.now().isoformat(timespec='seconds')); y -= 14
    c.drawString(x, y, "Last Turn File: " + last.name); y -= 18
    J = data.get("J", {})
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Metrics"); y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"J1={J.get('J1',0):0.3f}  J2={J.get('J2',0):0.3f}  J3={J.get('J3',0):0.3f}  J4={J.get('J4',0):0.3f}  J5={J.get('J5',0):0.3f}  SCM={data.get('SCM',0):0.3f}"); y -= 14
    c.drawString(x, y, f"Ripe: {data.get('ripe')}  Readiness: {data.get('readiness',0):0.2f}"); y -= 18
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Compact Proposal"); y -= 14
    c.setFont("Helvetica", 10)
    y = textblock(c, x, y, data.get("proposal") or "(kein Kompaktvorschlag im letzten Zug)", max_width=16*cm); y -= 8
    c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Vote"); y -= 14
    c.setFont("Helvetica", 10)
    v = data.get("vote") or {}
    c.drawString(x, y, f"Status: {v.get('status','—')}  Reason: {v.get('reason','—')}"); y -= 12
    c.showPage(); c.save()
    return pdf

if __name__ == "__main__":
    p = export_summary()
    print("Wrote:", p)
