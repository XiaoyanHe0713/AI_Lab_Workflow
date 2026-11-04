"""The certifier library. Every certifier: deterministic under a seed, a runtime cap, a version,
at least 20 known-answer cases under pytest."""
from .base import Certificate, Certifier, CertifierTimeout, run_with_cap  # noqa: F401
from .quadrature import QuadratureExactnessCertifier  # noqa: F401
from .witness import WitnessCertifier  # noqa: F401
from .pep import PEPRateCertifier, METHODS as PEP_METHODS  # noqa: F401

REGISTRY = {
    QuadratureExactnessCertifier.name: QuadratureExactnessCertifier,
    WitnessCertifier.name: WitnessCertifier,
    PEPRateCertifier.name: PEPRateCertifier,
}


def versions() -> dict[str, str]:
    return {name: cls.version for name, cls in REGISTRY.items()}
