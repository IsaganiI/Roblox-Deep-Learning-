"""Read and write files either on local disk or in S3.

Any path starting with "s3://" goes to S3 (boto3 is imported only then);
everything else is a normal local path. This is what makes moving to AWS a
configuration change instead of a rewrite.
"""

import io
import json
import os

import pandas as pd


def is_s3(uri: str) -> bool:
    return uri.startswith("s3://")


def join(base: str, name: str) -> str:
    if is_s3(base):
        return base.rstrip("/") + "/" + name
    return os.path.join(base, name)


def _bucket_key(uri: str):
    bucket, _, key = uri[len("s3://"):].partition("/")
    return bucket, key


def _s3():
    import boto3  # only needed when an s3:// path is used
    return boto3.client("s3")


def exists(uri: str) -> bool:
    if is_s3(uri):
        bucket, key = _bucket_key(uri)
        resp = _s3().list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
        return any(obj["Key"] == key for obj in resp.get("Contents", []))
    return os.path.exists(uri)


def read_bytes(uri: str) -> bytes:
    if is_s3(uri):
        bucket, key = _bucket_key(uri)
        return _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(uri, "rb") as f:
        return f.read()


def write_bytes(uri: str, data: bytes) -> None:
    if is_s3(uri):
        bucket, key = _bucket_key(uri)
        _s3().put_object(Bucket=bucket, Key=key, Body=data)
        return
    parent = os.path.dirname(uri)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(uri, "wb") as f:
        f.write(data)


def read_csv(uri: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(read_bytes(uri)), **kwargs)


def write_csv(df: pd.DataFrame, uri: str) -> None:
    write_bytes(uri, df.to_csv(index=False).encode("utf-8"))


def read_json(uri: str):
    return json.loads(read_bytes(uri).decode("utf-8"))


def write_json(obj, uri: str) -> None:
    write_bytes(uri, json.dumps(obj, indent=2, default=str).encode("utf-8"))
