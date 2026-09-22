import re
import uuid

from unstructured.partition.pdf import partition_pdf


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive newlines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# EXTRACT PDF
# ============================================================

def extract_pdf(pdf_path, vision_model=None):

    print("\n========== PDF EXTRACTION ==========")

    elements = partition_pdf(
        filename=pdf_path,

        strategy="fast",

        infer_table_structure=True,

        extract_image_block_types=[
            "Image",
            "Table"
        ],

        extract_image_block_to_payload=True,

        chunking_strategy=None,
    )

    print(
        f"Extracted {len(elements)} PDF elements."
    )

    processed_elements = []

    for index, element in enumerate(elements):

        element_type = type(element).__name__

        # ====================================================
        # TITLE
        # ====================================================

        if element_type == "Title":

            text = clean_text(
                str(element)
            )

            if text:

                processed_elements.append({
                    "element_id": str(uuid.uuid4()),
                    "type": "title",
                    "content": text,
                    "original_index": index,
                    "metadata": {},
                })

        # ====================================================
        # TEXT
        # ====================================================

        elif element_type in [
            "NarrativeText",
            "Text",
            "ListItem"
        ]:

            text = clean_text(
                str(element)
            )

            if text:

                processed_elements.append({
                    "element_id": str(uuid.uuid4()),
                    "type": "text",
                    "content": text,
                    "original_index": index,
                    "metadata": {},
                })

        # ====================================================
        # IMAGE / TABLE
        # ====================================================

        elif element_type in [
            "Image",
            "Table"
        ]:

            image_base64 = None

            metadata = getattr(
                element,
                "metadata",
                None
            )

            if metadata:

                image_base64 = getattr(
                    metadata,
                    "image_base64",
                    None
                )

            # ------------------------------------------------
            # If there is no image payload
            # ------------------------------------------------

            if not image_base64:

                print(
                    f"[WARNING] No image payload "
                    f"for {element_type}"
                )

                continue

            # ------------------------------------------------
            # Vision API
            # ------------------------------------------------

            description = ""

            if vision_model:

                visual_type = (
                    "table"
                    if element_type == "Table"
                    else "image"
                )

                try:

                    print(
                        f"Vision API processing "
                        f"{element_type}..."
                    )

                    description = vision_model.describe(
                        image_base64,
                        visual_type
                    )

                except Exception as e:

                    print(
                        f"[ERROR] Vision API failed: {e}"
                    )

                    description = (
                        f"[Vision processing failed for "
                        f"{visual_type}]"
                    )

            # ------------------------------------------------
            # Store in original position
            # ------------------------------------------------

            processed_elements.append({
                "element_id": str(uuid.uuid4()),
                "type": (
                    "table"
                    if element_type == "Table"
                    else "image"
                ),
                "content": description,
                "original_index": index,
                "metadata": {
                    "image_base64": image_base64,
                },
            })

    # ========================================================
    # Sort by original PDF order
    # ========================================================

    processed_elements.sort(
        key=lambda x: x["original_index"]
    )

    print(
        f"Processed {len(processed_elements)} elements."
    )

    return processed_elements


# ============================================================
# BUILD SECTION STRUCTURE
# ============================================================

def build_sections(elements):

    print("\n========== BUILDING SECTIONS ==========")

    sections = []

    current_section = None

    section_counter = 0

    for element in elements:

        # ====================================================
        # NEW SECTION
        # ====================================================

        if element["type"] == "title":

            # Save previous section
            if current_section:

                sections.append(
                    current_section
                )

            section_counter += 1

            current_section = {
                "section_id": str(uuid.uuid4()),

                "section_index": section_counter,

                "section_title":
                    element["content"],

                "section_path": [
                    element["content"]
                ],

                "elements": [],
            }

        # ====================================================
        # CONTENT BEFORE FIRST TITLE
        # ====================================================

        else:

            if current_section is None:

                current_section = {
                    "section_id": str(uuid.uuid4()),

                    "section_index": 0,

                    "section_title":
                        "Document Introduction",

                    "section_path": [
                        "Document Introduction"
                    ],

                    "elements": [],
                }

            current_section["elements"].append(
                element
            )

    # ========================================================
    # Save final section
    # ========================================================

    if current_section:

        sections.append(
            current_section
        )

    print(
        f"Created {len(sections)} sections."
    )

    return sections


# ============================================================
# DISPLAY SECTION STRUCTURE
# ============================================================

def print_sections(sections):

    print("\n========== SECTION STRUCTURE ==========")

    for section in sections:

        print(
            f"\nSECTION "
            f"{section['section_index']}: "
            f"{section['section_title']}"
        )

        for element in section["elements"]:

            print(
                f"  └── {element['type'].upper()}"
            )