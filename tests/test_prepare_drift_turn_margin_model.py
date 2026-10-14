"""The margin diagnostic must preserve the rest of the paired MJCF."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_drift_turn_margin_model import margin_variant_xml


class MarginVariantTests(unittest.TestCase):
    def test_only_default_collider_margin_changes(self):
        source = ROOT / "outputs/sim2sim-cmu-16_03/mujoco_model.xml"
        original = ET.parse(source).getroot()
        for value in (.0005, .001):
            variant = ET.fromstring(margin_variant_xml(source, value))
            geom = variant.find("./default/geom")
            self.assertEqual(float(geom.get("margin")), value)
            geom.set("margin", "0")
            self.assertEqual(ET.tostring(original), ET.tostring(variant))
        with self.assertRaisesRegex(ValueError, "predeclared"):
            margin_variant_xml(source, .002)


if __name__ == "__main__":
    unittest.main()
