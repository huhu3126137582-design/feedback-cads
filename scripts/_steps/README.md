# Internal workflow steps

These modules implement individual generation, evaluation, selection, audit,
reporting, and plotting steps. They are intentionally kept modular for unit
tests and independent audit boundaries.

Use `python scripts/feedback_cads_cli.py ...` as the public interface. Direct execution of a
file in this directory is supported for debugging, but is not the documented
reproduction workflow.
