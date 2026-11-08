"""The two PDF writers that tests/mm/day.py (copied from tests/day.py of MultimediaSearch at fe0ece5) imports from MultimediaSearch's tests/corpus.py, which is a
file of 1,100 lines of fixtures for its documents: `pdf` and `text_pdf`, moved from it unchanged."""

import re
import zlib


def pdf(stream: bytes, trailer: bytes = b"") -> bytes:
    """A one-page PDF whose content stream is `stream` (FlateDecode), with valid xref and startxref offsets. `trailer`
    adds entries to the trailer dictionary."""
    objects = [b"<</Type/Catalog/Pages 2 0 R>>", b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
               b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 120]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
               b"<</Length %d/Filter/FlateDecode>>stream\n" % len(stream) + stream + b"\nendstream",
               b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % n + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer<</Size %d/Root 1 0 R%s>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, trailer, xref)
    return bytes(out)


def text_pdf(lines: list[str]) -> bytes:
    """A PDF with a text layer: each of `lines` (Latin-1) is shown by a `Tj` operator in a deflated content stream."""
    show = b"".join(b"(%s) Tj 0 -18 Td\n" % re.sub(rb"([\\()])", rb"\\\1", line.encode("latin-1")) for line in lines)
    return pdf(zlib.compress(b"BT /F1 14 Tf 20 90 Td\n" + show + b"ET", 9))
