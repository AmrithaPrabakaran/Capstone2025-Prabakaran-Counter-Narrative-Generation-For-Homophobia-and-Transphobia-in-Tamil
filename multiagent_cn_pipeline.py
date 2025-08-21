import os, json, csv, time, requests

API_KEY = os.environ.get("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o-mini"

def chat(messages, temperature=0.7, max_tokens=1200):
    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def make_personas(t):
    sys = {"role":"system","content":"நீங்கள் தமிழில் தெளிவாகவும் சுருக்கமாகவும் பதிலளிக்கும் மொழி நிபுணர்."}
    usr = {"role":"user","content":(
        "கொடுக்கப்பட்ட வெறுப்புச் சொல் உரை (T) அடிப்படையில் 6–10 மனிதப்பாத்திரங்களை உருவாக்கவும். "
        "ஒவ்வொரு வரியும் JSON ஆக இருக்க வேண்டும். தலா: agent_id, persona (ஒரு வரி), counter_narrative (தமிழில், மரியாதையுடன்). "
        "எல்லா counter_narrative-களும் ஒருவருக்கொன்று வேறுபட்ட கோணங்களில் T-ஐ எதிர்க்க வேண்டும். "
        f"T: {t}\n"
        "வடிவம்:\n"
        '{"agent_id": 0, "persona": "...", "counter_narrative": "..."}\n'
        '{"agent_id": 1, "persona": "...", "counter_narrative": "..."}'
    )}
    return chat([sys, usr], temperature=0.6, max_tokens=1400)

def select_team(t, candidates_jsonl):
    sys = {"role":"system","content":"நீங்கள் தமிழில் தீர்மானங்களை தெளிவாக விளக்கும் உதவி அமைப்பு."}
    usr = {"role":"user","content":(
        "கீழே உள்ள மனிதப்பாத்திரக் குளத்திலிருந்து மூன்று பேரை முதன்மை குழுவாகத் தேர்வு செய்யவும். "
        "பல்வேறு பார்வைகள் பிரதிநிதித்துவப்படுத்தப்பட வேண்டும். "
        "ஒவ்வொரு தேர்விற்கும் reason புலத்தில் ஒரு வரி காரணத்தை கொடுக்கவும். "
        f"T: {t}\n\n"
        "Candidates (JSONL):\n"
        f"{candidates_jsonl}\n\n"
        "வெளியீடு JSONL வடிவம்:\n"
        '{"agent_id": X, "persona": "...", "counter_narrative": "...", "reason": "..."}'
    )}
    return chat([sys, usr], temperature=0.4, max_tokens=1000)

def run_debate(t, team_jsonl, rounds=3):
    sys = {"role":"system","content":"நீங்கள் விவாத ஒழுங்குபடுத்துநர். தமிழில் கட்டுப்பாடான விவாதத்தை நடத்தவும்."}
    usr = {"role":"user","content":(
        "முதன்மை குழு: மூன்று மனிதப்பாத்திரங்கள். எதிர் பக்கம்: பகுப்பாய்வாளர் முகவர் (AN). "
        "முதன்மை குழு T-ஐ எதிர்க்கும்; AN சவாலை முன்வைக்கும். "
        f"T: {t}\n\n"
        "குழு (JSONL):\n"
        f"{team_jsonl}\n\n"
        f"சுற்றுகள் (R): {rounds}\n"
        "வெளியீடு:\nstart_of_discussion\n...விவாத உரை...\nend_of_discussion\n\n"
        "_start_of_plan\n"
        "1) ...\n2) ...\n3) ...\n_optional_ack: ... (தேவைப்பட்டால்)\n"
        "end_of_plan"
    )}
    return chat([sys, usr], temperature=0.6, max_tokens=1800)

def distill_plan(t, discussion_block):
    sys = {"role":"system","content":"நீங்கள் தமிழில் திட்டச் சுருக்கம் எழுதும் உதவி முறைமை."}
    usr = {"role":"user","content":(
        "கீழே உள்ள விவாதப் பதிவை மட்டும் அடிப்படையாகக் கொண்டு அதிகபட்சம் மூன்று முக்கிய அம்சங்களாக சுருக்கவும். "
        "முக்கிய அம்சங்கள் தர்க்க ரீதியாக ஒழுங்குபடுத்தப்பட்டிருக்க வேண்டும். "
        "தேவைப்பட்டால் ஒரு short acknowledgment சேர்க்கலாம். "
        f"T: {t}\n\n"
        f"{discussion_block}\n\n"
        "_start_of_plan\n"
        "1) ...\n2) ...\n3) ...\n_optional_ack: ...\n"
        "end_of_plan"
    )}
    return chat([sys, usr], temperature=0.3, max_tokens=800)

def generate_cn(t, plan_block):
    sys = {"role":"system","content":"நீங்கள் தமிழில் மரியாதையான எதிர்-கதை எழுதும் உதவி அமைப்பு."}
    usr = {"role":"user","content":(
        "கீழே உள்ள திட்டத்தை மட்டும் பின்பற்றி தமிழ் எதிர்-கதை ஒன்றை ஒரு அல்லது இரண்டு வரிகளில் எழுதவும். "
        "மரியாதை, உட்சேர்ப்பு, இரக்கம் ஆகியவை வெளிப்பட வேண்டும்; ஆக்கிரமிப்பு இருக்கக் கூடாது. "
        f"T: {t}\n\n"
        f"{plan_block}\n\n"
        "Counter narrative (1–2 வரிகள்):"
    )}
    return chat([sys, usr], temperature=0.7, max_tokens=200)

def load_texts(path):
    rows = []
    with open(path, newline='', encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("text"):
                rows.append(row["text"])
    return rows

def save_rows(path, rows, header):
    write_header = not os.path.exists(path)
    with open(path, "a", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def process_batch(texts, out_path, rounds=3, pause=0.6):
    out = []
    for idx, t in enumerate(texts, 1):
        personas = make_personas(t)
        time.sleep(pause)
        team = select_team(t, personas)
        time.sleep(pause)
        debate = run_debate(t, team, rounds=rounds)
        time.sleep(pause)
        plan = distill_plan(t, debate)
        time.sleep(pause)
        cn = generate_cn(t, plan)
        out.append({
            "id": idx,
            "text": t,
            "personas_jsonl": personas,
            "team_jsonl": team,
            "discussion": debate,
            "plan": plan,
            "counter_narrative_ta": cn
        })
        time.sleep(pause)
    save_rows(out_path, out, ["id","text","personas_jsonl","team_jsonl","discussion","plan","counter_narrative_ta"])

if __name__ == "__main__":
    inp = os.environ.get("TEST_SET_CSV", "test_set.csv")
    outp = os.environ.get("OUTPUT_CSV", "cn_outputs.csv")
    rs = int(os.environ.get("ROUNDS", "3"))
    texts = load_texts(inp)
    process_batch(texts, outp, rounds=rs, pause=0.6)
    print(outp)
