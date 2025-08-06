import logging
from llama_cpp import Llama

try:
    llm = Llama(model_path="<add your path to phi-3 mini model here>",n_ctx=4096,n_threads=4,verbose=False)
except Exception as e:
    logging.error(f"Error loading model: {e}")
    llm = None

scene_cache = {}

def generate_ai_description(labels, relationships, counts):
    prompt = (
        "You are an AI assistant helping a visually impaired person understand their surroundings.\n\n"
        f"The following are detected in the scene: \n"
    )
   
    for label in labels:
        if label[0]=='person':
            prompt += f"- A person named {label[1]}\n"
        else:
            prompt += f"- An object named {label[1]}\n"        

    if relationships:
        prompt += "Their spatial relationships are as follows:\n"
        for rel in relationships:
            prompt += f"- {rel}\n"

    if counts:
        prompt += "The number of each object and/or person is:\n"
        for label, count in counts.items():
            obj_text = f"{count} {label}" if count == 1 else f"{count} {label}s"
            prompt += f"- {obj_text}\n"

    prompt += (
        "\nGenerate a short and literal description of the scene using ONLY the above facts."
        "Don't include words like Answer, Question, Input, Output, Assistant."
        "Don't repeat describing an object or a person more than twice."
        "Scene Description:"
    )
   
    logging.info(prompt)
    if llm == None:
        return 0
   
    response = llm(
        prompt,
        max_tokens=150,
        temperature=0.2,
        top_p=0.9,
        stop=["<|endoftext|>", "\nScene Description:"],
        repeat_penalty=1.15
    )

    output = response["choices"][0]["text"].strip()
    if "Scene Description:" in output:
        output = output.split("Scene Description:")[-1].strip()
    return output


def describe_the_scene(detected_objects, relationships, counts_dict):
    labels = [(obj['name'],obj['label']) for obj in detected_objects]
    label_set = frozenset((
    tuple(sorted(labels)),
    tuple(sorted(relationships)),
    tuple(sorted(counts_dict.items()))
))
    if label_set in scene_cache:
        return scene_cache[label_set]
   
    description = generate_ai_description(labels, relationships, counts_dict)
    if description !=0:
        scene_cache[label_set] = description
    return description


