from io import BytesIO

import joblib

from .types import ModelArtifact

ARTIFACT_VERSION = 1


def serialize_artifact(artifact: ModelArtifact) -> bytes:
    buffer = BytesIO()
    joblib.dump(artifact, buffer)
    return buffer.getvalue()


def deserialize_artifact(data: bytes) -> ModelArtifact:
    artifact = joblib.load(BytesIO(data))
    if not isinstance(artifact, ModelArtifact):
        raise ValueError("The uploaded file is not an MLStudio model artifact.")
    if artifact.version != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported artifact version {artifact.version}; "
            f"expected {ARTIFACT_VERSION}."
        )
    return artifact
