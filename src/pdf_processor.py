import re
from unstructured.partition.pdf import partition_pdf


# ============================================================
# HEADING VALIDATION
# ============================================================

def is_valid_heading(text):

    if not text:
        return False

    text = text.strip()

    if len(text) < 3:
        return False

    if len(text) > 150:
        return False

    lower = text.lower()

    # Table / experiment labels
    # Example: (A) base, (E) big
    if re.match(r"^\([A-Z]\)", text):
        return False

    # Bibliography-like fragments
    suspicious_phrases = [
        "arxiv preprint",
        "proceedings of",
        "in advances in",
        "conference on",
    ]

    if any(
        phrase in lower
        for phrase in suspicious_phrases
    ):
        return False

    # Mathematical / parameter fragments
    suspicious_terms = [
        "dmodel",
        "d_k",
        "dk",
        "d_v",
        "dv",
        "sin",
        "cos",
        "log",
    ]

    compact = lower.replace(" ", "")

    if any(
        term in compact
        for term in suspicious_terms
    ):
        return False

    # Must contain actual alphabetic content
    if not re.search(r"[A-Za-z]{2,}", text):
        return False

    return True


# ============================================================
# RAW UNSTRUCTURED ELEMENT TYPE
# ============================================================

def get_raw_element_type(element):

    category = getattr(
        element,
        "category",
        ""
    ).lower()

    if category == "title":
        return "title"

    if category == "image":
        return "image"

    if category == "table":
        return "table"

    return "text"


# ============================================================
# GET IMAGE/TABLE PAYLOAD
# ============================================================

def get_visual_data(element):

    metadata = getattr(
        element,
        "metadata",
        None
    )

    if metadata is None:
        return None

    if hasattr(metadata, "to_dict"):
        metadata = metadata.to_dict()

    if not isinstance(metadata, dict):
        return None

    image_base64 = metadata.get(
        "image_base64"
    )

    if not image_base64:
        return None

    return {
        "image_base64": image_base64,

        "mime_type": metadata.get(
            "image_mime_type",
            "image/jpeg"
        ),
    }


# ============================================================
# EXTRACT PDF
# ============================================================

import time


def extract_pdf(pdf_path, vision_model):

    elements = partition_pdf(
        filename=pdf_path,

        strategy="hi_res",

        infer_table_structure=True,

        extract_image_block_types=[
            "Image",
            "Table"
        ],

        extract_image_block_to_payload=True,

        chunking_strategy=None,
    )

    # ========================================================
    # Convert Unstructured objects → our dictionaries
    # ========================================================

    processed_elements = []

    for element in elements:

        element_type = get_raw_element_type(
            element
        )

        text = str(element).strip()

        # ====================================================
        # IMAGE / TABLE
        # ====================================================

        if element_type in [
            "image",
            "table"
        ]:

            visual_data = get_visual_data(
                element
            )

            if visual_data:

                print(
                    f"[VISION] Processing "
                    f"{element_type}..."
                )

                # =================================================
                # GEMINI RETRY
                # =================================================

                description = None

                for attempt in range(3):

                    try:

                        description = vision_model.describe(
                            visual_data["image_base64"],
                            visual_type=element_type
                        )

                        # Success
                        break

                    except Exception as e:

                        print(
                            f"[VISION] Attempt "
                            f"{attempt + 1}/3 failed: {e}"
                        )

                        if attempt < 2:

                            wait_time = 2 ** attempt

                            print(
                                f"[VISION] Retrying "
                                f"in {wait_time} seconds..."
                            )

                            time.sleep(wait_time)

                # =================================================
                # FALLBACK
                # =================================================

                if description is None:

                    print(
                        f"[VISION] Failed to describe "
                        f"{element_type} after 3 attempts."
                    )

                    description = (
                        f"{element_type.capitalize()} "
                        "from the PDF. "
                        "Vision description unavailable."
                    )

                # =================================================
                # KEEP BOTH:
                #
                # 1. Vision description → retrieval
                # 2. Actual image/table → output
                # =================================================

                processed_elements.append({

                    "type": element_type,

                    # Used for embedding/search
                    "text": description,

                    # ACTUAL IMAGE/TABLE
                    "image_base64":
                        visual_data["image_base64"],

                    "mime_type":
                        visual_data["mime_type"],
                })

            else:

                print(
                    f"[WARNING] {element_type} "
                    f"has no image payload."
                )

                processed_elements.append({

                    "type": element_type,

                    "text": "",

                    "image_base64": None,

                    "mime_type": None,
                })

            continue

        # ====================================================
        # TITLE / TEXT
        # ====================================================

        processed_elements.append({

            "type": element_type,

            "text": text,

            "image_base64": None,

            "mime_type": None,
        })

    # ========================================================
    # RETURN
    # ========================================================

    return processed_elements


# ============================================================
# BUILD SECTIONS
# ============================================================

def build_sections(elements):

    print(
        "\n========== BUILDING SECTIONS =========="
    )

    sections = []

    current_section_title = (
        "Document Introduction"
    )

    current_section_elements = []

    section_stack = []

    def flush_section():

        nonlocal current_section_elements

        if not current_section_elements:
            return

        sections.append({

            "section_id":
                f"section_{len(sections)}",

            "section_title":
                current_section_title,

            "section_path":
                section_stack.copy(),

            "elements":
                current_section_elements.copy(),
        })

        current_section_elements = []

    # ========================================================
    # PROCESS OUR DICTIONARIES
    # ========================================================

    for element in elements:

        element_type = element["type"]

        text = element.get(
            "text",
            ""
        ).strip()

        # ====================================================
        # TITLE
        # ====================================================

        if element_type == "title":

            # -----------------------------------------------
            # Genuine heading
            # -----------------------------------------------

            if is_valid_heading(text):

                print(
                    f"[SECTION] {text}"
                )

                flush_section()

                current_section_title = text

                # -------------------------------------------
                # Hierarchy
                # -------------------------------------------

                level_match = re.match(
                    r"^(\d+(?:\.\d+)*)\s+",
                    text
                )

                if level_match:

                    section_number = (
                        level_match.group(1)
                    )

                    level = (
                        section_number.count(".")
                        + 1
                    )

                    section_stack = (
                        section_stack[:level - 1]
                    )

                    section_stack.append(
                        text
                    )

                else:

                    section_stack = [
                        text
                    ]

                continue

            # -----------------------------------------------
            # False title
            # -----------------------------------------------

            print(
                f"[FALSE TITLE] {text} "
                f"→ treating as text"
            )

            element_type = "text"

        # ====================================================
        # IMAGE / TABLE
        # ====================================================

        if element_type in [
            "image",
            "table"
        ]:

            current_section_elements.append({

                "type": element_type,

                # Vision description
                "text": text,

                # Actual image/table
                "image_base64":
                    element.get(
                        "image_base64"
                    ),

                "mime_type":
                    element.get(
                        "mime_type"
                    ),
            })

            continue

        # ====================================================
        # TEXT
        # ====================================================

        current_section_elements.append({

            "type": "text",

            "text": text,

            "image_base64": None,

            "mime_type": None,
        })

    # ========================================================
    # FINAL SECTION
    # ========================================================

    flush_section()

    return sections


# ============================================================
# PRINT SECTIONS
# ============================================================

"""def print_sections(sections):

    print(
        "\n========== SECTION STRUCTURE =========="
    )

    for section in sections:

        print("\n--------------------------------------")

        print(
            f"Section: "
            f"{section['section_title']}"
        )

        print(
            "Path: "
            + " > ".join(
                section["section_path"]
            )
        )

        print(
            f"Elements: "
            f"{len(section['elements'])}"
        )

        for element in section["elements"]:

            print(
                f"  [{element['type']}] "
                f"{element.get('text', '')[:100]}"
            )"""