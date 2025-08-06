import math

def reverse_relation(relation):
    if " and " in relation:
        parts = relation.split(" and ")
        return " and ".join(reverse_relation(p) for p in parts)

    mapping = {
        "to the left of": "to the right of",
        "to the right of": "to the left of",
        "above": "below",
        "below": "above",
        "near": "near"
    }
    return mapping.get(relation, relation)


def compute_relation(bbox1, bbox2, label1, label2):
    x1, y1, x2, y2 = bbox1
    a1, b1, a2, b2 = bbox2

    size1 = (x2 - x1) * (y2 - y1)
    size2 = (a2 - a1) * (b2 - b1)
    size_ratio = size1 / size2 if size2 > 0 else 1

    if size_ratio > 5 or size_ratio < 0.2:
        return None

    center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    center2 = ((a1 + a2) / 2, (b1 + b2) / 2)

    width1 = x2 - x1
    height1 = y2 - y1
    width2 = a2 - a1
    height2 = b2 - b1

    dx = center2[0] - center1[0]
    dy = center2[1] - center1[1]

    norm_dx = dx / max(width1, width2)
    norm_dy = dy / max(height1, height2)

    horizontal = ""
    vertical = ""

    if abs(norm_dx) > 0.6:
        horizontal = "to the right of" if norm_dx > 0 else "to the left of"

    if abs(norm_dy) > 0.6:
        vertical = "below" if norm_dy > 0 else "above"

    if horizontal and vertical:
        relation = f"{vertical} and {horizontal}"
    elif horizontal:
        relation = horizontal
    elif vertical:
        relation = vertical
    else:
        relation = "near"

    if label1 == label2 and relation in {"near", "below"}:
        return None

    return f"{label1} is {relation} {label2}"


def construct_relationships(objects):
    relationships = []
    seen_relations = set()

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):        
            obj1, obj2 = objects[i], objects[j]
            bbox1 = obj1["bbox"]
            bbox2 = obj2["bbox"]
            label1 = obj1["label"]
            label2 = obj2["label"]

            relation = compute_relation(bbox1, bbox2, label1, label2)
            reverse = compute_relation(bbox2, bbox1, label2, label1)

            if relation and relation not in seen_relations and (reverse is None or reverse_relation(reverse.split(" is ", 1)[1]) != relation.split(" is ", 1)[1]):
                seen_relations.add(relation)
                relationships.append(relation)
   
    all_labels = {obj["label"] for obj in objects}
    related_labels = {rel.split(" is ")[0] for rel in relationships}
   
    for obj in objects:
        label = obj["label"]
        if label not in related_labels:
           
            x1, y1, x2, y2 = obj["bbox"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            position = []

            if center_x < 213:  
                position.append("left")
            elif center_x > 426:
                position.append("right")
            else:
                position.append("center")

            if center_y < 160:  
                position.append("top")
            elif center_y > 320:
                position.append("bottom")
            else:
                position.append("middle")

            pos_desc = " and ".join(position)
            relationships.append(f"{label} is at the {pos_desc} of the image")

    return relationships



