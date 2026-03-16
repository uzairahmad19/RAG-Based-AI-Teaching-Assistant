import os
import math
import json

n = 5 # Number of chunks to merge

for filename in os.listdir('transcripts'):
    filepath = os.path.join('transcripts', filename)
    if filename.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            new_chunks = []
            num_chunks = len(data['chunk'])
            num_groups = math.ceil(num_chunks / n)
            for i in range(num_groups):
                start_index = i * n
                end_index = min((i + 1) * n, num_chunks)
                merged_chunk = {
                    'name' : data['chunk'][start_index]['name'],
                    'number': data['chunk'][start_index]['number'],
                    'start': data['chunk'][start_index]['start'],
                    'end': data['chunk'][end_index - 1]['end'],
                    'text': ' '.join(chunk['text'] for chunk in data['chunk'][start_index:end_index])
                }
                new_chunks.append(merged_chunk)
        os.makedirs('merged_transcripts', exist_ok=True)
        new_filepath = os.path.join('merged_transcripts', filename)
        with open(new_filepath, 'w', encoding='utf-8') as f:
            json.dump({'chunk': new_chunks, 'text': data['text']}, f, ensure_ascii=False, indent=4)