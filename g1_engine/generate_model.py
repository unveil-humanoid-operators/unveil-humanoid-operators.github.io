"""
generate_model.py
-----------------
One-time script: parse g1_29dof.xml → model/g1_model.json

Run from g1_engine/ directory:
    python generate_model.py

The output JSON is loaded by viewer.html at runtime (no backend needed).
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def _vec(s, default):
    return [float(x) for x in s.split()] if s else list(default)


def _quat(s):
    return [float(x) for x in s.split()] if s else [1.0, 0.0, 0.0, 0.0]


def parse_model(xml_path: str) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find(".//worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> in XML")

    bodies = []

    def walk(elem, parent_name):
        if elem.tag != "body":
            return
        name   = elem.get("name", "")
        pos    = _vec(elem.get("pos"),  [0.0, 0.0, 0.0])
        quat   = _quat(elem.get("quat"))

        joint = None
        for child in elem:
            if child.tag == "joint":
                joint = {
                    "name": child.get("name", ""),
                    "type": child.get("type", "hinge"),
                    "axis": _vec(child.get("axis"), [0.0, 0.0, 1.0]),
                }
                break

        seen, meshes = set(), []
        for child in elem:
            if child.tag != "geom" or child.get("type") != "mesh":
                continue
            mesh_name = child.get("mesh")
            if not mesh_name or mesh_name in seen:
                continue
            seen.add(mesh_name)
            rgba = _vec(child.get("rgba", "0.72 0.72 0.72 1"), [0.72, 0.72, 0.72, 1.0])
            meshes.append({
                "name": mesh_name,
                "rgba": rgba[:3],
                "pos":  _vec(child.get("pos"),  [0.0, 0.0, 0.0]),
                "quat": _quat(child.get("quat")),
            })

        bodies.append({
            "name":   name,
            "parent": parent_name,
            "pos":    pos,
            "quat":   quat,
            "joint":  joint,
            "meshes": meshes,
        })

        for child in elem:
            if child.tag == "body":
                walk(child, name)

    for child in worldbody:
        if child.tag == "body":
            walk(child, None)

    return bodies


if __name__ == "__main__":
    here      = Path(__file__).parent
    xml_path  = here / "model" / "g1_29dof.xml"
    out_path  = here / "model" / "g1_model.json"

    bodies = parse_model(str(xml_path))
    with open(out_path, "w") as f:
        json.dump({"bodies": bodies}, f, separators=(",", ":"))

    print(f"OK  {len(bodies)} bodies  ->  {out_path}")
