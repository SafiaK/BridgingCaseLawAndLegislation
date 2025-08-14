import requests
import xml.etree.ElementTree as ET
import os

class LegislationParser:
    """
    A parser for UK legislation that extracts titles, terms, and sections from legislation.gov.uk XML data.
    """

    def __init__(self, url, is_section=False):
        """
        Initializes the parser with the legislation URL and determines the appropriate XML structure.

        Args:
            url (str): URL to fetch legislation XML.
            is_section (bool): Whether a specific section ID is being fetched.
        """
        self.debug = False  # Toggle debugging output

        # Extract section ID and base URL
        self.element_id, base_url = self.getTheSectionIdAndBaseUrl(url)
        self.url = base_url + "/data.akn"

        # Ensure secure URL
        if not self.url.startswith("https:"):
            self.url = self.url.replace("http", "https")

        # Define namespaces
        self.namespace = {'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0'}

        # Load XML tree
        self.tree = self._load_legislation()

    def _load_legislation(self):
        """Fetches and parses the XML from the given URL."""
        try:
            response = requests.get(self.url)
            if response.status_code == 200:
                return ET.ElementTree(ET.fromstring(response.content))
            else:
                raise Exception(f"Failed to load legislation data: {response.status_code}")
        except ET.ParseError:
            raise Exception("Error parsing the XML document")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")

    def get_legislation_title(self):
        """
        Fetches the title of the legislation from the XML document.

        Returns:
            str: The title of the legislation or an error message.
        """
        try:
            root = self.tree.getroot()
            
            # Check multiple possible title locations
            title_elements = [
                root.find(".//akn:longTitle", self.namespace),  # For Acts (ukpga)
                root.find(".//akn:shortTitle", self.namespace),  # Alternative title location
                root.find(".//akn:docTitle", self.namespace),  # Common title tag
                root.find(".//akn:title", self.namespace),  # General title
                root.find(".//akn:name", self.namespace),  # Used in some cases
                root.find(".//akn:heading", self.namespace)  # Used in UKSI documents
            ]
            
            for title_element in title_elements:
                if title_element is not None and title_element.text:
                    return title_element.text.strip()

            return "Title not found in the XML document"
        except Exception as e:
            return f"An error occurred while extracting title: {e}"

    def get_sections(self):
        """
        Retrieves all sections or regulations from the XML document.

        Returns:
            list: A list of dictionaries containing section ID and text.
        """
        root = self.tree.getroot()

        # Try multiple possible element types for sections
        section_elements = (
            root.findall(".//akn:section", self.namespace) +  # UK Acts (ukpga)
            root.findall(".//akn:regulation", self.namespace) +  # UK Statutory Instruments (uksi)
            root.findall(".//akn:part", self.namespace) +  # Some UKSI use <akn:part>
            root.findall(".//akn:chapter", self.namespace) +  # Some UKSI use <akn:chapter>
            root.findall(".//akn:article", self.namespace)  # Some laws use <akn:article>
        )

        sections = []
        for section in section_elements:
            section_id = section.attrib.get("eId", "unknown")
            section_text = self._extract_text(section)
            sections.append({"id": section_id, "text": section_text})

        return sections

    def save_all_sections_to_files(self, output_dir="legislation_sections"):
        """
        Saves all sections or regulations in the legislation to individual text files.

        Args:
            output_dir (str): Directory where section text files will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)

        sections = self.get_sections()
        if not sections:
            print("No sections or regulations found.")
            return

        for section in sections:
            file_path = os.path.join(output_dir, f"{section['id']}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(section['text'])
            print(f"Saved: {file_path}")

    def _extract_text(self, element):
        """Extracts all text content from an XML element and its children."""
        texts = []

        def process_node(node):
            if node.text and node.text.strip():
                texts.append(node.text.strip())
            for child in node:
                process_node(child)
                if child.tail and child.tail.strip():
                    texts.append(child.tail.strip())

        process_node(element)
        return " ".join(texts)

    def getTheSectionIdAndBaseUrl(self, url):
        """
        Extracts the base URL and section ID from a given UK legislation URL.

        Args:
            url (str): Legislation URL.

        Returns:
            tuple: (section ID, base URL)
        """
        url_parts = url.split('/')
        if 'id' in url_parts:
            url_parts.remove('id')

        section_idx = -1
        for i, part in enumerate(url_parts):
            if part.lower() in ['section', 'regulation', 'part', 'chapter', 'article']:
                section_idx = i
                break

        if section_idx == -1:
            return "", '/'.join(url_parts)

        section_id = '-'.join(url_parts[section_idx:]).lower()
        base_url = '/'.join(url_parts[:section_idx])
        return section_id, base_url

    def set_debug(self, debug_mode):
        """Enables or disables debug output."""
        self.debug = debug_mode


# Example usage:
if __name__ == "__main__":
    url_act = "https://www.legislation.gov.uk/ukpga/2018/16"
    url_act = "https://www.legislation.gov.uk/ukpga/2010/15"
    url_regulation = "https://www.legislation.gov.uk/uksi/2013/435"

    # Parse UK Act
    parser_act = LegislationParser(url_act, False)
    print("Act Title:", parser_act.get_legislation_title())
    parser_act.save_all_sections_to_files("data/legislation/ukpga/2010/15")

    '''
    # Parse UK Statutory Instrument (Regulation)
    parser_regulation = LegislationParser(url_regulation, False)
    print("Regulation Title:", parser_regulation.get_legislation_title())
    parser_regulation.save_all_sections_to_files("data/legislation/uksi/2013/435")
    '''