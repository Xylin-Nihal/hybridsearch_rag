import uuid
import pymupdf

from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Text,
    Title,
    NarrativeText,
    ListItem,
    Table,
)
from langchain_core.documents import Document

from .config import IMAGE_DIR, PDF_PATH


def extract_images(pdf_path):
    print("\n========== IMAGE EXTRACTION ==========")

    pdf = pymupdf.open(str(pdf_path))
    image_records = []

    for page_index, page in enumerate(pdf):
        page_number = page_index + 1
        images = page.get_images(full=True)

        print(f"Page {page_number}: {len(images)} images")

        for image_index, image in enumerate(images):
            xref = image[0]
            image_data = pdf.extract_image(xref)

            image_bytes = image_data["image"]
            extension = image_data["ext"]
            image_id = str(uuid.uuid4())

            filename = (
                f"page_{page_number}_"
                f"image_{image_index + 1}_"
                f"{image_id}.{extension}"
            )

            image_path = IMAGE_DIR / filename

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            record = {
                "image_id": image_id,
                "page_number": page_number,
                "image_index": image_index,
                "image_path": str(image_path),
                "type": "image",
            }

            image_records.append(record)
            print("Saved:", image_path)

    pdf.close()
    return image_records


def extract_elements(pdf_path):
    print("\n========== DOCUMENT EXTRACTION ==========")

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        infer_table_structure=True,
    )

    print(f"Total elements: {len(elements)}")

    documents = []

    for element in elements:
        page_number = getattr(
            element.metadata,
            "page_number",
            None
        )

        element_id = str(uuid.uuid4())

        if isinstance(element, Title):
            content = str(element).strip()

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "element_id": element_id,
                        "type": "title",
                        "page_number": page_number,
                        "source": str(PDF_PATH),
                    },
                )
            )

        elif isinstance(
            element,
            (Text, NarrativeText, ListItem)
        ):
            content = str(element).strip()

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "element_id": element_id,
                        "type": "text",
                        "page_number": page_number,
                        "source": str(PDF_PATH),
                    },
                )
            )

        elif isinstance(element, Table):
            table_html = getattr(
                element.metadata,
                "text_as_html",
                None
            )

            if table_html:
                content = "TABLE:\n" + table_html
            else:
                content = "TABLE:\n" + str(element)

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "element_id": element_id,
                        "type": "table",
                        "page_number": page_number,
                        "source": str(PDF_PATH),
                        "atomic": True,
                    },
                )
            )

    print(f"Text/Table/Title elements: {len(documents)}")

    return documents