"""HTTPS helpers for the catalog/survey downloads (VizieR, CDS hips2fits).

Some servers send an incomplete certificate chain: Windows completes it on its own, Python's OpenSSL does not
and fails with CERTIFICATE_VERIFY_FAILED. `truststore` lets the operating system verify the certificate, so the
connection stays fully verified; without it the default context is used.
"""
import ssl
import urllib.request

_CTX = None


def ssl_context():
    global _CTX
    if _CTX is None:
        try:
            import truststore
            _CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            _CTX = ssl.create_default_context()
    return _CTX


def urlopen(req, timeout=60):
    return urllib.request.urlopen(req, timeout=timeout, context=ssl_context())
