def clean_text_for_tts(text):
    lines = text.strip().splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("- response:"):
            line= line[len("- response:"):]
        elif line.strip().startswith("==="):
            continue        
        cleaned_lines.append(line.strip())
    cleaned_text = ' '.join(cleaned_lines)
    cleaned_text = cleaned_text.replace(":", "")
    return cleaned_text


