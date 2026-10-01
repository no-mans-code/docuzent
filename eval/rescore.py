"""Grades saved answers again against the question set as it is now - after a pattern was found too strict (a right
answer in other words marked wrong) - without asking the model anything. Rewrites the report's JSON and Markdown.

    python eval/rescore.py eval/results/<book>.json eval/sets/<set>.json
"""
import json
import re
import sys


def check(q, answer):
    problems = []
    for p in q.get("must", []):
        if not re.search(p, answer, re.I):
            problems.append(f"missing /{p}/")
    if q.get("any_of") and not any(re.search(p, answer, re.I) for p in q["any_of"]):
        problems.append(f"none of {q['any_of']}")
    for p in q.get("never", []):
        if re.search(p, answer, re.I):
            problems.append(f"must not say /{p}/")
    return problems


def main():
    report_path, set_path = sys.argv[1], sys.argv[2]
    report = json.load(open(report_path, encoding="utf-8"))
    questions = json.load(open(set_path, encoding="utf-8"))["questions"]
    changed = 0
    for r in report["questions"]:
        q = questions[r["n"] - 1]
        if "(off-leash)" in r["run"] and q.get("offleash"):
            q = q["offleash"]
        problems = check(q, r["answer"])
        if (not problems) != r["pass"]:
            changed += 1
            print(f"{r['run']} #{r['n']}: {'PASS' if r['pass'] else 'FAIL'} -> {'PASS' if not problems else 'FAIL'}")
        r["pass"], r["problems"] = not problems, problems
    for run in report["runs"]:
        rows = [r for r in report["questions"] if r["run"] == run["run"]]
        run["passed"] = sum(r["pass"] for r in rows)
        run["held_out_passed"] = sum(r["pass"] for r in rows if r["set"] == "held-out")
    json.dump(report, open(report_path, "w", encoding="utf-8", newline=""), indent=2, ensure_ascii=False)
    md = report_path[:-5] + ".md"
    lines = open(md, encoding="utf-8").read().split("\n")
    out = []
    for line in lines:
        m = re.match(r"^\| (.+?) \| \d+/(\d+) \| \d+/(\d+) \|(.*)$", line)
        run = next((x for x in report["runs"] if m and x["run"] == m.group(1)), None)
        out.append(f"| {run['run']} | {run['passed']}/{run['total']} | {run['held_out_passed']}/{run['held_out_total']} |{m.group(4)}" if run else line)
    text = "\n".join(out)
    text = text.split("\nFailures:")[0].rstrip() + "\n"
    fails = [r for r in report["questions"] if not r["pass"]]
    if fails:
        text += "\nFailures:\n\n" + "".join(f"- **{f['run']}** #{f['n']} {f['question']} - {'; '.join(f['problems'])}\n  > {f['answer'][:300].replace(chr(10), ' ')}\n" for f in fails)
    open(md, "w", encoding="utf-8", newline="").write(text)
    print(f"{changed} grade(s) changed")


main()
