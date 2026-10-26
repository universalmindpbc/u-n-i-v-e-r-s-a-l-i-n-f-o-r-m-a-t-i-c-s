"""Archive the exact website review; leave DeSci queued pending service repair."""
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[3]
here = Path(__file__).parent
sys.path.insert(0, str(root / 'research/informatics/scientific-swarm'))
from publication import Outbox, ExistingConnector
outbox = Outbox(here / 'publication')
record = json.loads((root / 'Informatics/quantum/reviews/oxtr-study-01/r1/review.json').read_text())
status = outbox.enqueue(record)
if '--upload' in sys.argv:
    adapter = ExistingConnector('synapse', '/private/tmp/informatics-synapse-qa/bin/python', 'syn78366147',
        '/Users/benkorbel/Documents/m o d u l 8 i n f i n i t y/i n f o r m a t i c s/private/connector-credentials/synapse-auth-token',
        root / 'Informatics/connectors')
    receipt = outbox.deliver(status['revision'], 'synapse', adapter)
    status['synapse_folder'] = receipt['remote_id']
status['destinations'] = outbox.status(status['revision'])
status['desci_note'] = 'Queued until the existing DeSci document service is repaired; no replacement draft created.'
(here / 'publication-status.json').write_text(json.dumps(status, indent=2) + '\n')
print(json.dumps(status))
