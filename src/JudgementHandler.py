import xml.etree.ElementTree as ET
import html


from lxml import etree
import os
import pandas as pd


from xml.etree import ElementTree

class JudgmentParser:
    def __init__(self, xml_file):
        tree = ET.parse(xml_file)
        self.root = tree.getroot()
        self.namespace = {
            'akn': 'http://docs.oasis-open.org/legaldocml/ns/akn/3.0',
            'uk': 'https://caselaw.nationalarchives.gov.uk/akn'
        }
    def has_legislation_reference(self):
        """Checks if there is any legislation reference in the judgment."""
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            for ref in body.findall('.//akn:ref', self.namespace):
                # Check if it's a legislation reference
                if ref.attrib.get('{https://caselaw.nationalarchives.gov.uk/akn}type') == 'legislation':
                    return True
        return False
    def extract_text_with_internal_tags(self, element):
        # Extract all text including nested tags, preserving the order
        return ''.join(element.itertext()).strip()
    def extract_legislation_refs_withID(self, element):
        """Helper function to extract legislation references from an XML element
        
        Handles both direct section references and section references that need to be 
        combined with the legislation ID to form complete references.
        """
        legislation_refs = []
        refs = element.findall('.//akn:ref', self.namespace)
        for ref in refs:
            if ref.attrib.get('{https://caselaw.nationalarchives.gov.uk/akn}type') == 'legislation':
                ref_text = self.extract_text_with_internal_tags(ref)
                href = ref.get('href', '')
                
                # Extract legislation ID and section from href
                legislation_id = None
                section_num = None
                section_text = None
                
                if href:
                    parts = href.split('/')
                    
                    # Handle section references
                    section_keywords = ['section', 'schedule', 'chapter', 'regulation', 'article']
                    
                    # Check if the href already contains a section reference
                    contains_section = any(keyword in href for keyword in section_keywords)
                    
                    # Extract the base legislation ID (everything up to the keyword or the entire path if no keyword)
                    if contains_section:
                        # Find the position of the section keyword
                        keyword_idx = next((i for i, part in enumerate(parts) if part in section_keywords), None)
                        if keyword_idx is not None:
                            # Get legislation ID from parts before the keyword
                            legislation_id = '/'.join(parts[:keyword_idx])
                            
                            # Extract the section keyword and its value
                            section_keyword = parts[keyword_idx]
                            section_path = '/'.join(parts[keyword_idx+1:])
                            section_num = section_path
                            section_text = f"{section_keyword}/{section_path}"
                    else:
                        # No section reference in the href, it's just the legislation ID
                        legislation_id = href
                
                # Now check the ref_text to see if it contains a section reference that's not in the href
                # Example: "section 55" in "section 55A(5) FLA 1986"
                additional_refs = []
                
                # If ref_text contains a section reference not in the href, create an additional reference
                if legislation_id and ref_text:
                    # Look for section patterns in the ref text (e.g., "section 55A")
                    section_patterns = []
                    for keyword in section_keywords:
                        # Match patterns like "section 55A" or "section 55A(5)"
                        match = re.search(rf"{keyword}\s+([0-9A-Za-z]+(?:\([0-9A-Za-z]+\))?)", ref_text)
                        if match:
                            section_patterns.append((keyword, match.group(1)))
                    
                    # Create additional references for sections mentioned in text but not in href
                    for keyword, pattern in section_patterns:
                        # If the href doesn't contain this exact section or it's a different section
                        pattern_clean = pattern.split('(')[0]  # Remove parenthetical parts like (5)
                        
                        # Skip if this section is already covered by the href
                        if section_num and pattern_clean == section_num:
                            continue
                        
                        # Otherwise create a new reference with the corrected section
                        additional_href = f"{legislation_id}/{keyword}/{pattern_clean}"
                        additional_refs.append({
                            'text': ref_text,
                            'href': additional_href,
                            'legislation_section': (legislation_id, f"{keyword}/{pattern_clean}"),
                            'note': 'Extracted from text reference'
                        })
                
                # Add the original reference
                legislation_refs.append({
                    'text': ref_text,
                    'href': href,
                    'legislation_section': (legislation_id, section_text) if legislation_id else None
                })
                
                # Add any additional references extracted from the text
                legislation_refs.extend(additional_refs)
                
        return legislation_refs
    def extract_legislation_refs(self, element):
        """Helper function to extract legislation references from an XML element"""
        legislation_refs = []
        refs = element.findall('.//akn:ref', self.namespace)
        for ref in refs:
            if ref.attrib.get('{https://caselaw.nationalarchives.gov.uk/akn}type') == 'legislation':
                ref_text = self.extract_text_with_internal_tags(ref)
                href = ref.get('href', '')
                
                # Extract legislation ID and section from href
                legislation_id = None
                section_num = None
                if href:
                    parts = href.split('/')
                    base_uri = '/'.join(parts[:-1])
                    if any(keyword in href for keyword in ['section', 'schedule', 'chapter', 'regulation', 'article']):
                        try:
                            # Get the full section path after the keyword
                            for keyword in ['section', 'schedule', 'chapter', 'regulation', 'article']:
                                if keyword in href:
                                    section_parts = href.split(f'{keyword}/')[1].split('/')
                                    section_path = '/'.join(section_parts)
                                    section_num = section_path  # Keep full section path like "1/3"
                                    break
                            
                            # Get legislation ID from parts before the keyword
                            keyword_idx = next(i for i, part in enumerate(parts) if part in ['section', 'schedule', 'chapter', 'regulation', 'article'])
                            legislation_id = '/'.join(parts[:keyword_idx])
                        except:
                            section_num = None
                    elif len(parts) >= 2:
                        # No specific keyword - use last parts for year/number
                        legislation_id = base_uri
                    
                legislation_refs.append({
                    'text': ref_text,
                    'href': href,
                    'legislation_section': (legislation_id, section_num) if legislation_id else None
                })
        return legislation_refs
    def get_judgment_body_paragraphs_subpara_text(self):
        

        paragraphs = []
        para_id_counter = {}
        paras = self.get_judgment_body_paragraphs_xml()
        for para in paras:
            para_id = para.get('eId')
            
            # Process subparagraphs within the paragraph
            subparagraphs = para.findall('.//akn:subparagraph', self.namespace)

            if subparagraphs:
                # Has subparagraphs - only process those
                if para_id not in para_id_counter:
                    para_id_counter[para_id] = 1
                
                for sub in subparagraphs:
                    sub_text = self.extract_text_with_internal_tags(sub)
                    sub_id = f"{para_id}_{para_id_counter[para_id]}"
                    para_id_counter[para_id] += 1

                    if sub_text:
                        refs = self.extract_legislation_refs(sub)
                        legislation_sections = [ref['legislation_section'] for ref in refs if ref['legislation_section']]
                        paragraphs.append({
                            'id': sub_id,
                            'text': sub_text,
                            'references': refs,
                            'legislation_sections': legislation_sections
                        })
            else:
                # No subparagraphs - process the paragraph text directly
                para_text = self.extract_text_with_internal_tags(para)
                if para_text:
                    refs = self.extract_legislation_refs(para)
                    legislation_sections = [ref['legislation_section'] for ref in refs if ref['legislation_section']]
                    paragraphs.append({
                        'id': para_id,
                        'text': para_text,
                        'references': refs,
                        'legislation_sections': legislation_sections
                    })

        return paragraphs
    def check_legislation_reference(self,paragraph):
        """
        Check if the paragraph contains a reference to legislation.
        """
        refs = paragraph.findall('.//akn:ref', self.namespace)  # Look for akn:ref tags
    
        for ref in refs:
            # Check if the 'uk:type' attribute is 'legislation'
            if ref.attrib.get('{https://caselaw.nationalarchives.gov.uk/akn}type') == 'legislation':
                return True
        return False
    def get_caselaw_meta(self):
        """Returns the case metadata such as URI and ID."""
        meta = {}
        identification = self.root.find('akn:judgment/akn:meta/akn:identification', self.namespace)
        
        if identification is not None:
            work = identification.find('akn:FRBRExpression', self.namespace)
            if work is not None:
                meta['uri'] = work.find('akn:FRBRuri', self.namespace).attrib['value']
                meta['date'] = work.find('akn:FRBRdate', self.namespace).attrib['date']
        
        return meta

    def get_judgment_body_paragraphs_xml(self):
        """Returns all paragraphs of the judgment body as XML elements."""
        paragraphs = []
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            paragraphs = body.findall('akn:paragraph', self.namespace)
        return paragraphs
    
    '''
    def get_judgment_body_paragraphs_text(self):
        """Returns all paragraphs of the judgment body as plain text, including references."""
        result = []
        case_meta = self.get_caselaw_meta()
        case_uri = case_meta.get('uri', '')
        
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        
        if body is not None:
            # First check if body is not None and has paragraphs
            paragraphs = body.findall('akn:paragraph', self.namespace)
            print("Found paragraphs:", len(paragraphs) if paragraphs else 0)
            if len(paragraphs) == 0:
                # Try finding paragraphs within levels if none found directly
                paragraphs = body.findall('.//akn:paragraph', self.namespace)
                if len(paragraphs) == 0:
                    # As a last resort, try finding paragraphs within levels
                    levels = body.findall('akn:level', self.namespace)
                    for level in levels:
                        level_paragraphs = level.findall('akn:paragraph', self.namespace)
                        paragraphs.extend(level_paragraphs)


            for para in paragraphs:
               global para_counter
               if not hasattr(self, 'para_counter'):
                   self.para_counter = 1
               
               para_id = para.attrib.get('eId', '')

               if not para_id:
                   para_id = f'para_{self.para_counter}'
                   self.para_counter += 1
            
               text = self.get_paragraph_text(para)
               refs = self.extract_legislation_refs(para)
               heading = self.extract_paragraph_heading(para)
               
               result.append((case_uri, para_id, text, refs,heading))
               
                   
                #text_content = para.find('akn:content', self.namespace).text
        return result
    '''
    def get_judgment_body_paragraphs_text(self):
        """Returns all paragraphs of the judgment body as plain text, including references."""
        result = []
        case_meta = self.get_caselaw_meta()
        case_uri = case_meta.get('uri', '')
        current_heading = None  # Track the latest heading
        
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        
        if body is not None:
            paragraphs = body.findall('akn:paragraph', self.namespace)
            print("Found paragraphs:", len(paragraphs) if paragraphs else 0)
            if len(paragraphs) == 0:
                paragraphs = body.findall('.//akn:paragraph', self.namespace)
                if len(paragraphs) == 0:
                    levels = body.findall('akn:level', self.namespace)
                    for level in levels:
                        level_paragraphs = level.findall('akn:paragraph', self.namespace)
                        paragraphs.extend(level_paragraphs)
                        # Extract headings from levels
                        new_heading = self.extract_paragraph_heading(level)
                        if new_heading:
                            current_heading = new_heading  # Update heading

            for para in paragraphs:
                global para_counter
                if not hasattr(self, 'para_counter_previous'):
                    self.para_counter = 0
                
                para_id = para.attrib.get('eId', '')
                if not para_id:
                    para_id = f'para_{self.para_counter}'
                    self.para_counter += 1
              
            
                text = self.get_paragraph_text(para)
                refs = self.extract_legislation_refs(para)
                
                # Check if this paragraph itself has a new heading
                new_heading = self.extract_paragraph_heading(para)
                if new_heading:
                    current_heading = new_heading  # Update heading
                
                result.append((case_uri, para_id, text, refs))
                
        return result
    def extract_paragraph_heading(self, paragraph):
        """Extract heading from bold-styled text."""
        heading = None
        for span in paragraph.findall('.//akn:span', self.namespace):
            style = span.attrib.get('style', '').lower()
            if 'font-weight:bold' in style:
                heading = span.text.strip() if span.text else heading
                break
        return heading
    def extract_paragraph_heading(self, element):
        """Extracts a heading from bold text in <p> elements inside <level> structures."""
        for p in element.findall('.//akn:p', self.namespace):
            for span in p.findall('.//akn:span', self.namespace):
                style = span.attrib.get('style', '').lower()
                if 'font-weight:bold' in style:
                    return span.text.strip() if span.text else None
        return None

    def get_paragraph_text(self,paragraph):
        """
        Extracts the text from the paragraph, including references, and handles special characters.
        """
        text_pieces = []
    
        # Iterate over elements and extract text
        for elem in paragraph.iter():
            if elem.tag.endswith('ref') or elem.tag.endswith('a'):  # Handle references
                text_pieces.append(self.extract_text_with_internal_tags(elem))
                '''
                if 'uk:canonical' in elem.attrib:
                    text_pieces.append(f"({elem.attrib['uk:canonical']})")  # Add canonical reference
                elif 'href' in elem.attrib:
                    text_pieces.append(f" {elem.attrib['href']} ")  # Add href if no canonical reference
                '''
            else:
                if elem.text:
                    text_pieces.append(elem.text)

            if elem.tail:
                #print(elem.tail)
                text_pieces.append(elem.tail)

        # Join all text pieces
        paragraph_text = ' '.join(text_pieces)

        # Decode HTML entities like &#8217;, &#8220;, etc.
        paragraph_text = html.unescape(paragraph_text)

        return paragraph_text

    def get_references(self):
        """Returns a list of references used in the judgment body paragraphs."""
        references = []
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            for para in body.findall('akn:paragraph', self.namespace):
                for ref in para.findall('.//akn:ref', self.namespace):
                    references.append(ref.attrib['href'])
        return references
    def get_legislation_references(self):
        """Returns a list of all legislation references in the judgment."""
        legislation_refs = []
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            for ref in body.findall('.//akn:ref', self.namespace):
                # Check if it's a legislation reference
                if ref.attrib.get('{https://caselaw.nationalarchives.gov.uk/akn}type') == 'legislation':
                    href = ref.attrib.get('href')
                    if href:
                        legislation_refs.append(href)
        return list(set(legislation_refs))  # Remove duplicates
    
    def get_paragraph_by_eId(self, eId):
        """Returns the paragraph text corresponding to the given eId."""
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            paragraph = body.find(f".//akn:paragraph[@eId='{eId}']", self.namespace)
            if paragraph is not None:
                return paragraph.find('akn:content', self.namespace).text
        return None
    
    def get_all_paragraphs_with_legislation_ref(self):
        """
        Returns a list of tuples containing (caseUri, paragraphId, text, references)
        for each paragraph in the judgment body.
        """
        result = []
        case_meta = self.get_caselaw_meta()
        case_uri = case_meta.get('uri', '')
        
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            for para in body.findall('akn:paragraph', self.namespace):
                if self.check_legislation_reference(para):
                    # Get paragraph ID
                    para_id = para.attrib.get('eId', '')
                    content = self.get_paragraph_text(para)
                    refs = self.extract_legislation_refs(para) #[ref.attrib['href'] for ref in paragraph.findall('.//akn:ref', self.namespace)]
                
                    result_text = self.get_paragraph_with_references_by_eId(para_id)
                    text = content
                    refs = refs
                

                    # Create tuple and append to result
                    result.append((case_uri, para_id, text, refs))
        
        return result

    def get_references_by_paragraph_eId(self, eId):
        """Returns the paragraph text along with any references by the given eId."""
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            paragraph = body.find(f".//akn:paragraph[@eId='{eId}']", self.namespace)
            if paragraph is not None:
                refs = [ref.attrib['href'] for ref in paragraph.findall('.//akn:ref', self.namespace)]
                return refs
        return []
    
    def get_paragraph_with_references_by_eId(self, eId):
        """Returns the paragraph text along with any references by the given eId."""
        body = self.root.find('akn:judgment/akn:judgmentBody/akn:decision', self.namespace)
        if body is not None:
            paragraph = body.find(f".//akn:paragraph[@eId='{eId}']", self.namespace)
            #print("paragraph",eId)
            if paragraph is not None:
                content = self.get_paragraph_text(paragraph)
                refs = self.extract_legislation_refs(paragraph) #[ref.attrib['href'] for ref in paragraph.findall('.//akn:ref', self.namespace)]
                return {
                    'content': content,
                    'references': refs
                }
        return None

if __name__ == '__main__':
    import sys
    import json
    
    
        
    #xml_file = "data/test/ukftt_grc_2025_251.xml"
    xml_file = "data/test/ewhc_fam_2022_1890.xml"
    handler = JudgmentParser(xml_file)
    
    # Get all paragraphs with legislation references
    results = handler.get_judgment_body_paragraphs_text()
    
    # Print results in JSON format
    output = []
    for case_uri, para_id, text, refs in results:
        output.append({
            'caseUri': case_uri,
            'paragraphId': para_id, 
            'text': text,
            'references': refs
        })
    
    with open('output.json', 'w') as f:
        json.dump(output, f, indent=2)