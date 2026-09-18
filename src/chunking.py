import re
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import (
    SemanticChunker,
    RecursiveCharacterTextSplitter,
)

from .config import PDF_PATH, FIXED_OVERLAP


def is_numbered_section(title):
    match = re.match(
        r"^\s*(\d+(?:\.\d+)*)",
        title
    )

    if not match:
        return None

    number = match.group(1)
    return len(number.split("."))


def build_sections(documents):
    print("\n========== BUILDING SECTIONS ==========")

    sections = []

    current_section_title = "Document Introduction"
    current_section_path = []
    current_section_elements = []

    section_stack = []

    def flush_section():
        nonlocal current_section_elements

        if current_section_elements:
            sections.append({
                "section_id": str(uuid.uuid4()),
                "section_title": current_section_title,
                "section_path": list(current_section_path),
                "elements": current_section_elements,
            })

        current_section_elements = []

    for document in documents:
        element_type = document.metadata["type"]

        if element_type == "title":
            title = document.page_content.strip()

            if not title:
                continue

            flush_section()

            level = is_numbered_section(title)

            if level is None:
                section_stack = [title]
            else:
                if len(section_stack) < level:
                    section_stack.extend(
                        [""] * (level - len(section_stack))
                    )

                section_stack = (
                    section_stack[:level - 1]
                    + [title]
                )

                section_stack = [
                    x for x in section_stack
                    if x
                ]

            current_section_title = title
            current_section_path = list(section_stack)

        else:
            document.metadata["section_title"] = (
                current_section_title
            )

            document.metadata["section_path"] = (
                list(current_section_path)
            )

            current_section_elements.append(document)

    flush_section()

    print(f"Sections created: {len(sections)}")

    for section in sections:
        print(
            f"  - {section['section_title']} "
            f"({len(section['elements'])} elements)"
        )

    return sections


def apply_fixed_overlap(chunks, overlap_size=50):
    if len(chunks) <= 1:
        return chunks

    overlapped_chunks = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped_chunks.append(chunk)
            continue

        previous_chunk = chunks[i - 1]
        previous_text = previous_chunk.page_content

        overlap = previous_text[-overlap_size:]

        new_text = (
            overlap
            + "\n"
            + chunk.page_content
        )

        new_metadata = dict(chunk.metadata)
        new_metadata["fixed_overlap"] = overlap_size

        overlapped_chunks.append(
            Document(
                page_content=new_text,
                metadata=new_metadata,
            )
        )

    return overlapped_chunks


def semantic_chunking(sections, embeddings):
    print("\n========== SEMANTIC CHUNKING ==========")

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
        add_start_index=True,
    )

    final_chunks = []

    for section in sections:
        section_elements = section["elements"]
        section_title = section["section_title"]
        section_path = section["section_path"]
        section_id = section["section_id"]

        text_documents = []
        atomic_documents = []

        for document in section_elements:
            element_type = document.metadata["type"]

            if element_type == "text":
                text_documents.append(document)

            elif element_type == "table":
                atomic_documents.append(document)

        if text_documents:
            combined_text = "\n\n".join(
                doc.page_content
                for doc in text_documents
            )

            section_document = Document(
                page_content=combined_text,
                metadata={
                    "section_title": section_title,
                    "section_path": section_path,
                    "section_id": section_id,
                    "type": "text",
                    "page_number": text_documents[0].metadata.get(
                        "page_number"
                    ),
                    "source": str(PDF_PATH),
                },
            )

            semantic_chunks = semantic_splitter.split_documents(
                [section_document]
            )

            semantic_chunks = apply_fixed_overlap(
                semantic_chunks,
                overlap_size=FIXED_OVERLAP,
            )

            for chunk in semantic_chunks:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
                chunk.metadata["chunk_type"] = "text"
                chunk.metadata["section_title"] = section_title
                chunk.metadata["section_path"] = section_path
                chunk.metadata["section_id"] = section_id
                chunk.metadata["atomic"] = False

                final_chunks.append(chunk)

        for table in atomic_documents:
            table.metadata["chunk_id"] = str(uuid.uuid4())
            table.metadata["chunk_type"] = "table"
            table.metadata["section_title"] = section_title
            table.metadata["section_path"] = section_path
            table.metadata["section_id"] = section_id
            table.metadata["atomic"] = True

            final_chunks.append(table)

    print(f"Semantic chunks: {len(final_chunks)}")

    return final_chunks


def recursive_character_chunking(sections):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    recursive_chunks = []

    for section in sections:
        text = "\n\n".join(
            element.page_content
            for element in section["elements"]
            if element.metadata["type"] == "text"
        )

        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk_text in chunks:
            recursive_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_type": "text_recursive",
                        "section_title": section["section_title"],
                        "section_path": section["section_path"],
                        "section_id": section["section_id"],
                        "atomic": False,
                    },
                )
            )

    return recursive_chunks


def create_image_chunks(image_records, sections):
    print("\n========== IMAGE CHUNKS ==========")

    image_chunks = []

    for image in image_records:
        page_number = image["page_number"]

        matching_section = None

        for section in sections:
            for element in section["elements"]:
                element_page = element.metadata.get("page_number")

                if element_page == page_number:
                    matching_section = section
                    break

            if matching_section:
                break

        if matching_section:
            section_title = matching_section["section_title"]
            section_path = matching_section["section_path"]
            section_id = matching_section["section_id"]
        else:
            section_title = "Unknown Section"
            section_path = []
            section_id = None

        image_text = (
            f"IMAGE\n"
            f"Section: {section_title}\n"
            f"Page: {page_number}\n"
            f"Figure/Image {image['image_index'] + 1}\n"
            f"Image extracted from the document."
        )

        image_chunk = Document(
            page_content=image_text,
            metadata={
                "chunk_id": str(uuid.uuid4()),
                "chunk_type": "image",
                "image_id": image["image_id"],
                "image_path": image["image_path"],
                "page_number": page_number,
                "section_title": section_title,
                "section_path": section_path,
                "section_id": section_id,
                "atomic": True,
                "source": str(PDF_PATH),
            },
        )

        image_chunks.append(image_chunk)

    print(f"Image chunks: {len(image_chunks)}")

    return image_chunks


def add_neighbor_relationships(chunks):
    print("\n========== BUILDING NEIGHBORS ==========")

    chunk_lookup = {
        chunk.metadata["chunk_id"]: chunk
        for chunk in chunks
    }

    previous_by_section = {}

    for chunk in chunks:
        section_id = chunk.metadata.get("section_id")
        chunk_id = chunk.metadata["chunk_id"]

        if section_id not in previous_by_section:
            previous_by_section[section_id] = None

        previous_chunk_id = previous_by_section[section_id]

        chunk.metadata["prev_chunk_id"] = previous_chunk_id
        chunk.metadata["next_chunk_id"] = None

        if previous_chunk_id:
            chunk_lookup[
                previous_chunk_id
            ].metadata["next_chunk_id"] = chunk_id

        previous_by_section[section_id] = chunk_id

    return chunks