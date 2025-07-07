from ultralytics import YOLO
import cv2  

def construct_relationships(objects):
    
    relationships = []
    seen_relations = set() 
    num_objects = len(objects)

    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            obj1, obj2 = objects[i], objects[j]
            x1, y1, x2, y2 = obj1["bbox"]
            x1_o, y1_o, x2_o, y2_o = obj2["bbox"]

            
            center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            center2 = ((x1_o + x2_o) / 2, (y1_o + y2_o) / 2)

            dx = center1[0] - center2[0]  # Horizontal distance
            dy = center1[1] - center2[1]  # Vertical distance

            # Determine spatial relationship
            relation = None

            if abs(dx) < 50 and abs(dy) < 50:
                relation = "near"
            elif abs(dx) > abs(dy):  
                relation = "to the right of" if dx > 0 else "to the left of"
            else:  
                relation = "above" if dy < 0 else "below"

            if relation:
                formatted_relation = f"{obj1['label']} {relation} {obj2['label']}"

                
                if (obj1["label"] == obj2["label"]) and ("near" in formatted_relation or "below" in formatted_relation):
                    continue  # Ignore "person below person", "car near car" cases

                if formatted_relation not in seen_relations:
                    seen_relations.add(formatted_relation)
                    relationships.append(formatted_relation)

    return relationships
