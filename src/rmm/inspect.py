import json
from pprint import pprint

out_file = "src/rmm/logs/lucid_c.json_rmm_retrievallog_flat-contriever_1"

with open(out_file, "r") as f:
    first = json.loads(next(f))

extraction_log = first["prospective_reflection"]["extraction_log"]
memory_bank = first["prospective_reflection"]["memory_bank"]

print("\n=== EXTRACTION LOG ===")
for sess in extraction_log:
    print(f"\nSession: {sess['session_id']}")
    print("Dialogue turns:")
    for turn in sess["dialogue_turns"]:
        print(f"  Turn {turn['turn_id']}")
        print(f"    SPEAKER_1: {turn['speaker_1']}")
        print(f"    SPEAKER_2: {turn['speaker_2']}")

    print("Extracted memories:")
    for mem in sess["extracted_memories"]:
        print(f"  summary   : {mem['summary']}")
        print(f"  reference : {mem['reference']}")

print("\n=== MEMORY BANK REFERENCES ===")
for i, mem in enumerate(memory_bank):
    print(f"\nMemory {i}: {mem['summary']}")
    for ref in mem["references"]:
        print(f"  session_id: {ref['session_id']}")
        print(f"  turn_ids  : {ref['turn_ids']}")
        print("  raw_turns :")
        for t in ref["raw_turns"]:
            print(f"    Turn {t['turn_id']}")
            print(f"      SPEAKER_1: {t['speaker_1']}")
            print(f"      SPEAKER_2: {t['speaker_2']}")





# import json

# out_file = "src/rmm/logs/YOUR_OUTPUT_FILE"

with open(out_file, "r") as f:
    first = json.loads(next(f))

sess_map = {
    s["session_id"]: {t["turn_id"]: t for t in s["dialogue_turns"]}
    for s in first["prospective_reflection"]["extraction_log"]
}

for mem in first["prospective_reflection"]["memory_bank"]:
    print(f"\nMEMORY: {mem['summary']}")
    for ref in mem["references"]:
        sid = ref["session_id"]
        print(f"  session: {sid}")
        for tid in ref["turn_ids"]:
            turn = sess_map[sid][tid]
            print(f"    referenced Turn {tid}")
            print(f"      SPEAKER_1: {turn['speaker_1']}")
            print(f"      SPEAKER_2: {turn['speaker_2']}")
