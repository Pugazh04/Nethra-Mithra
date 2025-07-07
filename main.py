from Object_Detection import detect_objects
from Scene_GNN import construct_relationships
from LLM import generate_ai_description
from gTTS  import text_to_speech

det_obj= []
j= 0
while True:  
    objects = detect_objects()
    j+=1
    if j==1:
        for k in objects:
            det_obj.append(k)
    else:
        for i in objects:
            if i['label'] not in [d.get('label') for d in det_obj]:
                det_obj.append(i)
    det_labels = {obj.get("label") for obj in det_obj}
    if len(det_labels) >= 2:
        break
print(det_obj)
rel= construct_relationships(det_obj)
print(rel)
desc= generate_ai_description(obj, rel)
audio= text_to_speech(desc)