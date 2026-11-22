import json
from pathlib import Path
import unittest


class RecordsTest(unittest.TestCase):
    def test_known_input_contract(self):
        source = Path(__file__).resolve().parents[1] / "data" / "records.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        activities = {item['activity'] for item in payload['activities']}
        for item in payload['choices']:
            available = set(item['offered']) - set(item['wait_not_invited'])
            assert set(item['wait_not_invited']) <= set(item['offered']) <= activities
            assert {item['selected'], item['switch_to']} - {None} <= available
        assert set(payload['evaluation_cycles']) <= {item['cycle'] for item in payload['choices']}
        assert all(item['activity'] in activities for item in payload['confirmed'] + payload['rotation'])
