import json

infile = "src/generation/generation_logs/gemini-3-flash-preview_orig-session_testlog_top999context_jsonformat_useronlytrue.jsonl"
outfile = "src/generation/generation_logs/gemini-3-flash-preview_orig-session_testlog_top999context_jsonformat_useronlytrue.fixed.jsonl"

decoder = json.JSONDecoder()
fixed_objects = []

with open(infile, "r") as f:
    for line_num, line in enumerate(f, 1):
        s = line.strip()
        if not s:
            continue

        pos = 0
        line_had_object = False

        while pos < len(s):
            while pos < len(s) and s[pos].isspace():
                pos += 1
            if pos >= len(s):
                break

            try:
                obj, end = decoder.raw_decode(s, pos)
                fixed_objects.append(obj)
                line_had_object = True
                pos = end
            except json.JSONDecodeError as e:
                print(f"Could not parse line {line_num} at char {pos}: {e}")
                print("Context:")
                print(s[max(0, pos-120):pos+300])
                break

        if not line_had_object:
            print(f"No JSON object parsed from line {line_num}")

with open(outfile, "w") as f:
    for obj in fixed_objects:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {len(fixed_objects)} clean JSON objects to:")
print(outfile)