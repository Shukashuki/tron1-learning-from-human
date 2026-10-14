"""Make a one-field MuJoCo contact-margin diagnostic from the paired model.

The paired USD-matched MJCF has zero contact margin on all geoms. The official
upstream MJCF authors 1 mm. This changes only the inherited default geom
margin, not wheel size, COM, controller, actor, reference or contact solref.
Neither setting is claimed equivalent to PhysX's runtime contact offset.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


PAIRED_MODEL_SHA256 = "bd92264b72b4d09084732888dbd81fd94a64a1678978e58a2ab0dd191887f513"
ALLOWED_MARGINS = (0.0005, 0.001)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def margin_variant_xml(source, margin_m):
    if margin_m not in ALLOWED_MARGINS:
        raise ValueError("Only the two predeclared 0.5/1.0 mm diagnostic margins are allowed")
    source = Path(source)
    if digest(source) != PAIRED_MODEL_SHA256:
        raise ValueError("Expected exact USD-matched paired MuJoCo model")
    root = ET.parse(source).getroot()
    original = copy.deepcopy(root)
    geom_default = root.find("./default/geom")
    if geom_default is None or geom_default.get("margin") != "0":
        raise ValueError("Expected zero-margin default collider")
    geom_default.set("margin", format(margin_m, ".4f"))
    # Structural equality after undoing the sole edit proves that neither the
    # USD-matched wheel cylinders nor the base inertial changed in the XML.
    geom_default.set("margin", "0")
    if ET.tostring(root) != ET.tostring(original):
        raise ValueError("Unexpected second model edit")
    geom_default.set("margin", format(margin_m, ".4f"))
    return ET.tostring(root, encoding="unicode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--margin-m", type=float, choices=ALLOWED_MARGINS, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_model.exists() or args.output_report.exists():
        parser.error("Preserving prior diagnostic files")
    xml = margin_variant_xml(args.source_model, args.margin_m)
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_model.write_text(xml)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(args.output_model.resolve()))
    relevant = [model.geom(name).id for name in ("floor", "wheel_L_collision", "wheel_R_collision")]
    actual = model.geom_margin[relevant].tolist()
    if any(abs(value - args.margin_m) > 1e-12 for value in actual):
        raise ValueError("Compiled contact margin differs from requested setting")
    widths = [float(model.geom_size[model.geom(name).id, 1])
              for name in ("wheel_L_collision", "wheel_R_collision")]
    if widths != [0.025, 0.025]:
        raise ValueError("Diagnostic must retain USD-matched 50 mm wheel widths")
    report = {"kind": "evaluation_only_default_contact_margin_variant",
              "source_model_sha256": PAIRED_MODEL_SHA256,
              "output_model_sha256": digest(args.output_model),
              "only_xml_edit": f"default/geom@margin: 0 -> {format(args.margin_m, '.4f')} m",
              "compiled_floor_and_wheel_margins_m": actual,
              "compiled_wheel_halfwidths_m": widths,
              "contact_offset_equivalence_to_physx_claimed": False}
    args.output_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
