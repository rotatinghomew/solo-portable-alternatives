"""Parent-aid activity rotation analysis.

The package deliberately keeps three things separate:

* what was actually open to a family (the choice set),
* what the family did (including explicit decline and later switching),
* what the scheduler may assume when a whole set of offerings changes.

Nothing in the prediction path uses an activity that was never offered, and a
family always contributes at most one participant.
"""

__version__ = "0.1.0"
