# import json

# # 🔧 Hardcoded input file
# path = "src/generation/generation_logs/gemini-3-flash-preview_no-retrieval_testlog_top5context_jsonformat_useronlyfalse.jsonl"

# def check_jsonl(path):
#     print(f"Checking file: {path}\n")

#     with open(path, "r") as f:
#         for i, line in enumerate(f, 1):
#             line = line.strip()

#             if not line:
#                 continue

#             try:
#                 json.loads(line)
#             except Exception as e:
#                 print(f"❌ Bad line {i}: {e}")
#                 print("---- Problematic content (truncated) ----")
#                 print(line[:500])
#                 print("----------------------------------------")
#                 return

#     print("✅ No issues found — file is valid JSONL.")

# if __name__ == "__main__":
#     check_jsonl(path)


path = "src/generation/generation_logs/gemini-3-flash-preview_no-retrieval_testlog_top5context_jsonformat_useronlyfalse.jsonl"

target = 1780
start = target - 3
end = target + 8

with open(path, "r") as f:
    for i, line in enumerate(f, 1):
        if start <= i <= end:
            print(f"{i}: {repr(line)}")