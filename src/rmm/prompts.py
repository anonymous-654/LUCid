EXTRACTION_SYSTEM_PROMPT = """Function: Memory extraction for SPEAKER_1

Task Description:
Given a session of dialogue between SPEAKER_1 and SPEAKER_2, extract the personal summaries of SPEAKER_1, with references to the corresponding turn IDs. Ensure the output adheres to the following rules:

• Output results in JSON format. The top-level key is "extracted_memories". The value should be a list of dictionaries, where each dictionary has the keys "summary" and "reference":
  – summary: A concise personal summary, which captures relevant information about SPEAKER_1’s experiences, preferences, and background, across multiple turns.
  – reference: A list of references, each in the format of [turn_id] indicating where the information appears.

• If no personal summary can be extracted, return NO_TRAIT.

Example:
INPUT:
• Turn 0:
  – SPEAKER_1: Did you check out that new gym in town?
  – SPEAKER_2: Yeah, I did. I’m not sure I like the vibe there, though.
• Turn 1:
  – SPEAKER_1: What was wrong with it?
  – SPEAKER_2: The folks there seemed to care more about how they looked than working out. It was a little too trendy for me. I’m pretty plain.
• Turn 2:
  – SPEAKER_1: Ah, got it. Well, maybe one of the older gyms will work out better for you—or I guess you could get that treadmill you were talking about before. Are you leaning one way or the other yet?
  – SPEAKER_2: I’m leaning towards the treadmill. I think it will work better for my lifestyle. I just don’t know which type to get. There are so many choices out there. Do you use a treadmill at your gym? Do you have a suggestion for a home one?
• Turn 3:
  – SPEAKER_1: I usually just lift weights there, to be honest. But I think I’ve heard good things about the NordicTrack?
  – SPEAKER_2: Yeah, I’ve heard good things about that, too. I like the idea of a multi-exercise piece of equipment. As long as the weather isn’t too bad, then I prefer to go for a run. But since it rains quite a bit here, I like the idea of an inside option. How is the weather in New England?
• Turn 4:
  – SPEAKER_1: Oh, it can get pretty foggy and rainy here too, I’m afraid. But as I’m sure you’ve heard, it’s really beautiful in the fall! Are there four distinct seasons where you are, too?
  – SPEAKER_2: Yes, I’ve heard about the fall colors. I may get there one day. Yes, we have seasons—rain, lighter rain, summer, and more rain! Ha!
• Turn 5:
  – SPEAKER_1: Haha! I lived overseas in the tropics once. Sounds just like it!
  – SPEAKER_2: The tropics sound great. It’s not as warm as the tropics, but I like it. I’m from Alaska, so I’m pretty weather-tough.

OUTPUT:
{{
  "extracted_memories": [
    {{
      "summary": "SPEAKER_1 asked about a new gym in town and suggested older gyms or a treadmill as alternatives.",
      "reference": [0, 2]
    }},
    {{
      "summary": "SPEAKER_1 usually lifts weights at the gym rather than using a treadmill.",
      "reference": [3]
    }},
    {{
      "summary": "SPEAKER_1 has heard good things about the NordicTrack treadmill.",
      "reference": [3]
    }},
    {{
      "summary": "SPEAKER_1 lives in New England and experiences foggy and rainy weather but enjoys the fall season.",
      "reference": [4]
    }},
    {{
      "summary": "SPEAKER_1 has lived overseas in the tropics before.",
      "reference": [5]
    }}
  ]
}}

Task:
Follow the JSON format demonstrated in the example above and extract the personal summaries for SPEAKER_1 from the following dialogue session.

Input: {}
Output:"""


UPDATE_SYSTEM_PROMPT = """Task Description:
Given a list of history personal summaries for a specific user and a new and similar personal summary from the same user, update the personal history summaries following the instructions below:

• Input format: Both the history personal summaries and the new personal summary are provided in JSON format, with the top-level keys of "history_summaries" and "new_summary".

• Possible update actions:
  – Add: If the new personal summary is not relevant to any history personal summary, add it.
    Format: Add()
  – Merge: If the new personal summary is relevant to a history personal summary, merge them as an updated summary.
    Format: Merge(index, merged_summary)

Note: index is the position of the relevant history summary in the list.
merged_summary is the merged summary of the new summary and the relevant history summary.
Two summaries are considered relevant if they discuss the same aspect of the user’s personal information or experiences.

• If multiple actions need to be executed, output each action in a single line, and separate them with a newline character ("\\n").

• Do not include additional explanations or examples in the output—only return the required action functions.

Example:
INPUT:
• History Personal Summaries:
  {{"history_summaries": ["SPEAKER_1 works out although he doesn’t particularly enjoy it."]}}
• New Personal Summary:
  {{"new_summary": "SPEAKER_1 exercises every Monday and Thursday."}}

OUTPUT ACTION:
Merge(0, SPEAKER_1 exercises every Monday and Thursday, although he doesn’t particularly enjoy it.)

Task:
Follow the example format above to update the personal history for the given case.

INPUT:
• History Personal Summaries: {}
• New Personal Summary: {}

OUTPUT ACTION:"""