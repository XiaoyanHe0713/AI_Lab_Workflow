"""The harnesses (Work package 3): programs around the trained roles. Every number that enters a reward, a
scoreboard or a figure is computed by a program; where a harness needs language it calls a frontier
provider during the build (GPT first, Claude second, behind one interface), every call goes through the
budget guard and into the ledger, and nothing a harness writes enters a reward."""
from .base import Harness, HarnessReply, harness_provider, parse_json  # noqa: F401
