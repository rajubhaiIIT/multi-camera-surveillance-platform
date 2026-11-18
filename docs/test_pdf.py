# test_pdf.py
import sys
from pathlib import Path

class PDFWriter:
    def __init__(self):
        self.objects = []
        self.pages = []

    def _add_object(self, content: bytes) -> int:
        self.objects.append(content)
        return len(self.objects)

    def generate(self) -> bytes:
        # Build PDF stream
        out = bytearray()
        out.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        
        # Object 1: Catalog
        catalog_idx = 1
        pages_idx = 2
        font_idx = 3
        font_bold_idx = 4
        
        # Prepare page objects
        page_objs = []
        content_objs = []
        
        # We will allocate IDs
        # 1: Catalog
        # 2: Pages
        # 3: Font Regular (Helvetica)
        # 4: Font Bold (Helvetica-Bold)
        # Next objects: Contents and Page for each page
        current_id = 5
        
        page_records = []
        for p in self.pages:
            c_id = current_id
            current_id += 1
            p_id = current_id
            current_id += 1
            page_records.append((p_id, c_id, p))
            
        # Write Catalog
        offsets = {}
        def write_obj(oid: int, body: bytes):
            offsets[oid] = len(out)
            out.extend(f"{oid} 0 obj\n".encode("ascii"))
            out.extend(body)
            out.extend(b"\nendobj\n")
            
        write_obj(catalog_idx, f"<< /Type /Catalog /Pages {pages_idx} 0 R >>".encode("ascii"))
        
        # Write Pages root
        kids_str = " ".join([f"{p_id} 0 R" for p_id, _, _ in page_records])
        write_obj(pages_idx, f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_records)} >>".encode("ascii"))
        
        # Write Fonts
        write_obj(font_idx, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        write_obj(font_bold_idx, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        
        # Write Pages & Contents
        for p_id, c_id, content in page_records:
            stream_bytes = content
            c_body = f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii") + stream_bytes + b"\nendstream"
            write_obj(c_id, c_body)
            
            p_body = (
                f"<< /Type /Page /Parent {pages_idx} 0 R "
                f"/MediaBox [0 0 612 792] "
                f"/Contents {c_id} 0 R "
                f"/Resources << /Font << /F1 {font_idx} 0 R /F2 {font_bold_idx} 0 R >> >> >>"
            ).encode("ascii")
            write_obj(p_id, p_body)
            
        # xref
        xref_offset = len(out)
        out.extend(b"xref\n")
        out.extend(f"0 {len(offsets) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for i in range(1, len(offsets) + 1):
            out.extend(f"{offsets[i]:010d} 00000 n \n".encode("ascii"))
            
        out.extend(b"trailer\n")
        out.extend(f"<< /Size {len(offsets) + 1} /Root {catalog_idx} 0 R >>\n".encode("ascii"))
        out.extend(b"startxref\n")
        out.extend(f"{xref_offset}\n%%EOF\n".encode("ascii"))
        return bytes(out)

writer = PDFWriter()
stream = b"BT /F2 20 Tf 50 720 Td (Intelligent Multi-Camera Surveillance Platform) Tj ET\n"
stream += b"BT /F1 12 Tf 50 690 Td (System Architecture & Technical Overview) Tj ET\n"
writer.pages.append(stream)
pdf_data = writer.generate()

out_path = Path("docs/Surveillance_Platform_Overview.pdf")
out_path.write_bytes(pdf_data)
print(f"Generated {out_path} ({len(pdf_data)} bytes)")
