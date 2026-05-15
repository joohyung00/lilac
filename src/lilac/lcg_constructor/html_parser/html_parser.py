
import os
import json
import yaml
import argparse
from tqdm import tqdm

from bs4 import BeautifulSoup, Tag
from urllib.parse import unquote

from src.lilac.lcg_constructor.html_parser.constants import (
    WikipediaAddress, TempConstant
)
from src.utils.utils import read_yaml, REPO_ROOT, input_subpath, artifact_subpath





def parse_arguments():
    
    CONFIG_PATH = f"{REPO_ROOT}/config/lcg_constructor/a.yaml"
    config = read_yaml(CONFIG_PATH)
    
    parser = argparse.ArgumentParser(description="Parse HTML documents and generate embeddings.")
    parser.add_argument("--target_data", type=str, choices = [config["type_to_dataset"]["multimodalqa"]])
    args = parser.parse_args()
    
    return args, config



def main():
    args, config = parse_arguments()
    
    # Initialize the HTML parser
    html_parser = HtmlParser(args, config)
    
    # Run the parser
    html_parser.run()


class HtmlParser:


    def __init__(self, args, config):

        # NB: this auxiliary parser is not on the main pipeline path; it operates on
        # the "val" split for legacy HTML ingestion.
        ds_name = args.target_data
        self._htmls_directory       = artifact_subpath(config, ds_name, "htmls_dirname", "val")
        self._parsed_documents_path = artifact_subpath(config, ds_name, "parsed_documents_dirname", "val")

        return


    def run(self):
        
        self.parseHtmlDocuments()

        return



    def parseHtmlDocuments(self):
        """
        Parse the given HTML documents to find specific "div" elements and their child tags.
        
        Args:
            html_docs (list of str): List of HTML document strings to parse.

        Returns:
            list of dict: A list containing parsed data for each document.
            
            - div       -> headings
            - p         -> passage
            - ul        -> lists
            - table     -> table
            - figure    -> image

            - dl        -> text, but ignorable
            - style     -> ignorable
            - link      -> ignorable
        """
        
            
        def extractWikipediaChildren(webpage_obj):
            
            html_doc = webpage_obj["html"]
            
            soup = BeautifulSoup(html_doc, 'html.parser')
            div_elements = soup.find_all("div", class_ = "mw-content-ltr mw-parser-output")

            # Parse title
            title = ""
            h1_element = soup.find("h1", class_ = "firstHeading mw-first-heading")
            for descendent in h1_element.descendants:
                if isinstance(descendent, str):
                    if title != "":
                        title += " "
                    title += descendent.strip()
            id_sequence.append("h1_1")
            id_to_component["h1_1"] = title
            id_to_html["h1_1"] = h1_element.prettify()
                                
            # Locate main body
            if len(div_elements) != 1:  
                print("Unexpected: Expected exactly one matching div element, but found " + str(len(div_elements)))
            elif len(div_elements) == 0:
                print("Unexpected: No matching div element found.")
                return []
            div_element = div_elements[0]
            
            # Make a list of main body's children
            children_list = recursivelyParseChildren(div_element, [])
            
            return title, children_list
        
        
        def recursivelyParseChildren(element, children_list):
            for child in element.children:
                if child.name == "meta":
                    recursivelyParseChildren(child, children_list)
                else:
                    children_list.append(child)
                
            return children_list
        
        
        def extractWikipediaCommonsChildren(webpage_obj):
            
            html_doc = webpage_obj["html"]
            
            soup = BeautifulSoup(html_doc, 'html.parser')
            div_elements = soup.find_all("div", class_ = "mw-content-ltr mw-parser-output")

            # Parse title
            title = ""
            h1_element = soup.find("h1", class_ = "firstHeading mw-first-heading")
            for descendent in h1_element.descendants:
                if isinstance(descendent, str):
                    if title != "":
                        title += " "
                    title += descendent.strip()
            id_sequence.append("h1_1")
            id_to_component["h1_1"] = title
            id_to_html["h1_1"] = h1_element.prettify()
                                
            full_image_div = soup.find("div", class_="fullImageLink")
            # Find img tag within it
            img_tag = None
            if full_image_div:
                img_tag = full_image_div.find("img")
    
                
            summary_header = soup.find("h2", id="Summary")
            # Find <h2 id="Summary">

            summary_table = soup.find("table", class_="vevent")
            # Find <table class="fileinfotpl-type-information vevent">
            
            # If any of it is none, then don't put it in the children list
            children_list = []
            if img_tag != None:  children_list.append(img_tag)
            if summary_header != None:  children_list.append(summary_header)
            if summary_table != None:  children_list.append(summary_table)
            
            return title, children_list
        
        
        def extractMWikipediaChildren(webpage_obj):
            
            html_doc = webpage_obj["html"]
            soup = BeautifulSoup(html_doc, 'html.parser')
            
            # Parse title
            title = ""
            h1_element = soup.find("h1", class_ = "firstHeading mw-first-heading")
            for descendent in h1_element.descendants:
                if isinstance(descendent, str):
                    if title != "":
                        title += " "
                    title += descendent.strip()
            id_sequence.append("h1_1")
            id_to_component["h1_1"] = title
            id_to_html["h1_1"] = h1_element.prettify()
            
            
            div_elements = soup.find_all("div", class_ = "mw-content-ltr mw-parser-output")             
            # Locate main body
            if len(div_elements) != 1:  
                print("Unexpected: Expected exactly one matching div element, but found " + str(len(div_elements)))
            elif len(div_elements) == 0:
                print("Unexpected: No matching div element found.")
                return []
            div_element = div_elements[0]
            
            # Make a list of main body's children
            children_list = get_div_children(div_element)
            
            return title, children_list

        def get_div_children(div_element):
            children = []
            for child in div_element.children:
                if isinstance(child, Tag):
                    if child.name == 'section':
                        # Add the children of <section> instead of the section itself
                        for section_child in child.children:
                            if isinstance(section_child, Tag):
                                children.append(section_child)
                    else:
                        children.append(child)
            return children
        
            
            
        html_filenames = os.listdir(self._htmls_directory)
        html_filenames.sort()


        for html_filename in tqdm(html_filenames):
            
            hierarchy = None
            id_sequence = []
            id_to_component = {}
            id_to_html = {}
            
            with open(os.path.join(self._htmls_directory, html_filename), "r") as f:
                webpage_obj = json.load(f)
                
            if WikipediaAddress.EN_WIKIPEDIA_BASE_URL.value in webpage_obj["url"]:
                title, children_list = extractWikipediaChildren(webpage_obj)
            elif WikipediaAddress.COMMONS_WIKIMEDIA_BASE_URL.value in webpage_obj["url"]:
                title, children_list = extractWikipediaCommonsChildren(webpage_obj)
            else:
                title, children_list = extractMWikipediaChildren(webpage_obj)
            
            # Parse the children sequence
                # Make hierarchy
                # Parse each of the components
            hierarchy = self.parseChildrenList(title, children_list, id_sequence, id_to_component, id_to_html)
            
            parsed_html_doc = {
                "title": title,
                "hierarchy": hierarchy, 
                "id_sequence": id_sequence,
                "id_to_component": id_to_component,
                "id_to_html": id_to_html
            }
            
            parsed_html_doc = {
                "title": title,
                "hierarchy": hierarchy,
                "id_sequence": id_sequence,
                "text": {k : v for k, v in id_to_component.items() if "p" in k},
                "table": {k : v for k, v in id_to_component.items() if "t" in k},
                "image": {k : v for k, v in id_to_component.items() if "i" in k},
                "sentence": {},
                "proposition": {},
                "table_segment": {},
                "subimage": {},
                "id_to_html": id_to_html
            }

            with open(os.path.join(self._parsed_documents_path, html_filename), "w") as f:
                json.dump(parsed_html_doc, f, indent = 4)
        
        return
    
    
    

    def parseChildrenList(self, page_title, children_list, id_sequence, id_to_component, id_to_html):
        
        def headingNumToIdx(num):
            return num - 2
        
        def locateHierarchyDict(hierarchy_dict_root, hierarchy_cursor):
            dictt = hierarchy_dict_root
            for key in hierarchy_cursor:
                if key == None:
                    continue
                dictt = dictt[key]
            return dictt        
        
        hierarchy = {}
        current_hierarchy_cursor = []
                
        heading_id = 1
        passage_id = 1
        table_id = 1
        image_id = 1
    
        
        for child in children_list:

            component_found = False
            if child.name == "div" and child.get("class") and any(cls.startswith("mw-heading") for cls in child["class"]):
                
                heading_level = int(child["class"][1].split("mw-heading")[-1])
                idx_level = headingNumToIdx(heading_level)
                heading_text = child.find(["h2", "h3", "h4", "h5", "h6"]).text.strip() if child.find(["h2", "h3", "h4", "h5", "h6"]) else "Unknown"
                
                # Fill in id_to_component, id_to_html, id_sequence
                id = "h" + str(idx_level) + "_" + str(heading_id)
                id_sequence.append(id)
                id_to_component[id] = {"text": heading_text}
                id_to_html[id] = child.prettify()
                heading_id += 1
                
                # Fill in hierarchy
                if heading_text in ["References", "External links", "Notes", "See also"]:
                    break
                
                if len(current_hierarchy_cursor) <= idx_level:
                    current_hierarchy_dict = locateHierarchyDict(hierarchy, current_hierarchy_cursor)
                    current_hierarchy_dict[heading_text] = {}
                    
                    if idx_level == len(current_hierarchy_cursor):
                        current_hierarchy_cursor.append(heading_text)
                    else:
                        # Append None to fill the gap
                        for _ in range(len(current_hierarchy_cursor), idx_level):
                            current_hierarchy_cursor.append(None)
                        current_hierarchy_cursor.append(heading_text)
                    
                    
                elif len(current_hierarchy_cursor) > idx_level:
                    current_hierarchy_cursor = current_hierarchy_cursor[:idx_level]
                    
                    current_hierarchy_dict = locateHierarchyDict(hierarchy, current_hierarchy_cursor)
                    current_hierarchy_dict[heading_text] = {}
                    
                    current_hierarchy_cursor.append(heading_text)
                
            
            elif child.name == "p":
                passage = self.parseHtmlPassage(child)
                id = "po_" + str(passage_id)
                
                id_sequence.append(id)
                id_to_component[id] = passage
                id_to_html[id] = child.prettify()
                passage_id += 1
                component_found = True
                
            elif child.name == "ul":
                llist = self.parseHtmlList(child)
                id = "pl_" + str(passage_id)
                
                id_sequence.append(id)
                id_to_component[id] = llist
                id_to_html[id] = child.prettify()
                passage_id += 1
                component_found = True
            
            elif child.name == "table":
                
                # If its class is "box-More_citations_needed_section", then continue
                if child.get("class") and "box-More_citations_needed_section" in child["class"]:
                    continue
                
                # If its class is infobox or wikitable
                ordinary_table = True
                if not (child.get("class") and ("infobox" in child["class"] or "wikitable" in child["class"])):
                
                    # Find all tables within child's descendants
                    if not hasattr(child, "find_all"):
                        continue
                    
                    tables_found = child.find_all("table")
                
                    if len(tables_found) > 0:
                        
                        ordinary_table = False
                        # Remove tables that are ancestors of other tables
                        tables_to_remove = set()
                        for table in tables_found:
                            ancestor_tables = table.find_parents("table")
                            tables_to_remove.update(ancestor_tables)
                            
                        # Filter tables to exclude ancestor tables
                        filtered_tables = [table for table in tables_found if table not in tables_to_remove]
                        
                        for table in filtered_tables:
                        
                            parsed_table = self.parseHtmlTable(table, page_title)
                            table_id_str = "t_" + str(table_id)
                            
                            id_sequence.append(table_id_str)
                            id_to_component[table_id_str] = parsed_table
                            id_to_html[table_id_str] = table.prettify()
                            table_id += 1

                            current_hierarchy_dict = locateHierarchyDict(hierarchy, current_hierarchy_cursor)
                            if TempConstant.COMPONENTS.value not in current_hierarchy_dict:
                                current_hierarchy_dict[TempConstant.COMPONENTS.value] = []
                            current_hierarchy_dict[TempConstant.COMPONENTS.value].append(table_id_str)
                
                if ordinary_table:
                    table = self.parseHtmlTable(child, page_title)
                    id = "t_" + str(table_id)
                    
                    id_sequence.append(id)
                    id_to_component[id] = table
                    id_to_html[id] = child.prettify()
                    table_id += 1
                    component_found = True
                
            elif child.name == "figure" or child.name == "img":
                image = self.parseHtmlImage(child)
                id = "i_" + str(image_id)
                
                id_sequence.append(id)
                id_to_component[id] = image
                id_to_html[id] = child.prettify()
                image_id += 1
                component_found = True
                
            else:
                # Find all tables within child's descendants
                if not hasattr(child, "find_all"):  # Check if child is a Tag
                    continue
                
                tables_found = child.find_all("table")
                
                # Remove tables that are ancestors of other tables
                tables_to_remove = set()
                for table in tables_found:
                    ancestor_tables = table.find_parents("table")
                    tables_to_remove.update(ancestor_tables)

                # Filter tables to exclude ancestor tables
                filtered_tables = [table for table in tables_found if table not in tables_to_remove]

                for table in filtered_tables:
                    
                    parsed_table = self.parseHtmlTable(table, page_title)
                    table_id_str = "t_" + str(table_id)
                    
                    id_sequence.append(table_id_str)
                    id_to_component[table_id_str] = parsed_table
                    id_to_html[table_id_str] = table.prettify()
                    table_id += 1

                    current_hierarchy_dict = locateHierarchyDict(hierarchy, current_hierarchy_cursor)
                    if TempConstant.COMPONENTS.value not in current_hierarchy_dict:
                        current_hierarchy_dict[TempConstant.COMPONENTS.value] = []
                    current_hierarchy_dict[TempConstant.COMPONENTS.value].append(table_id_str)
                
            if component_found:
                current_hierarchy_dict = locateHierarchyDict(hierarchy, current_hierarchy_cursor)
                if TempConstant.COMPONENTS.value not in current_hierarchy_dict:
                    current_hierarchy_dict[TempConstant.COMPONENTS.value] = []
                current_hierarchy_dict[TempConstant.COMPONENTS.value].append(id)
                
        return hierarchy
            
        
    def parseHtmlPassage(self, element):
        """
        Parse a <p> element to extract text and hyperlinks.
        
        Args:
            element (bs4.element.Tag): The <p> element to parse.
        
        Returns:
            dict: Contains the extracted text and hyperlink mappings.
        """
        
        text, hyperlinks = self.extractTextInDFS(element, "", [])
        
        return {"text": text, TempConstant.HYPERLINKS.value: hyperlinks}
        
        
    def parseHtmlList(self, element):
        """
        Parse a <ul> element to extract the list items and their content.

        Args:
            element (bs4.element.Tag): The <ul> element to parse.

        Returns:
            dict: A single dictionary containing concatenated text and adjusted hyperlinks.
        """
        
        # Initialize the final result dictionary
        final_result = {
            "text": "",
            TempConstant.HYPERLINKS.value: []
        }

        # Find all <li> elements within the <ul>
        list_items = element.find_all("li", recursive=False)
        accumulated_text_length = 0  # Tracks the accumulated length of the text

        for li in list_items:
            # Extract text and hyperlinks for each <li>
            item_text, item_hyperlinks = self.extractTextInDFS(li, "", [])

            # Update the final result
            if final_result["text"]:
                final_result["text"] += "\n"  # Add newline before concatenating
            final_result["text"] += "- " + item_text

            # Adjust hyperlink offsets and append to the final result
            for hyperlink_obj in item_hyperlinks:
                adjusted_link = {
                    "text": hyperlink_obj["text"],
                    TempConstant.HYPERLINK.value: hyperlink_obj[TempConstant.HYPERLINK.value],
                    "start": hyperlink_obj["start"] + accumulated_text_length,
                    "end": hyperlink_obj["end"] + accumulated_text_length
                }
                final_result[TempConstant.HYPERLINKS.value].append(adjusted_link)

            # Update the accumulated text length
            accumulated_text_length += len(item_text) + 1  # +1 for the newline

        return final_result
    
    
    def parseHtmlTable(self, html, wikipedia_page_title):
        
        table_soup = html
        table = {
            "webpage_title": wikipedia_page_title,
            "table_caption": None,
            "columns": 0,
            "rows": 0,
            "table": [],
            TempConstant.REFS.value: {}
        }

        # Process table caption (if exists)
        caption = table_soup.find("caption")
        if caption:
            table["table_caption"] = caption.text.strip()

        # Process rows
        row_elements = table_soup.find_all("tr")
        
        first_row = True
        num_rows = 0
        
        ref_counter = 1
        ref_memory = {}
        
        for row_idx, row_element in enumerate(row_elements):
            
            col_idx = 0
            row_data = []
            cells = row_element.find_all(["th", "td"])
                        
            for cell in cells:

                # Handle prior multicolumn, multirows
                # if (col_idx, row_idx) in ref_memory:
                while (col_idx, row_idx) in ref_memory:
                    ref_id = ref_memory[(col_idx, row_idx)]
                    row_data.append({"ref": ref_id})
                    del ref_memory[(col_idx, row_idx)]
                    col_idx += 1

                text, hyperlinks = self.extractTextInDFS(cell, "", [])
                cell_data = {
                    "text": text,
                    TempConstant.HYPERLINKS.value: hyperlinks,
                    "header": cell.name == "th"
                }
                
                # Check if there is "img" tag in the cell
                img_tag = cell.find("img")
                if img_tag:
                    img_src = img_tag.get("src")
                    if img_src:
                        if "https://" not in img_src:
                            cell_data["image"] = f"https://en.wikipedia.org{img_src}"
                        else:
                            cell_data["image"] = img_src

                # Handle colspan and rowspan
                colspan = cell.get("colspan", "1")
                try:                colspan = int(colspan) if colspan.isdigit() else 1
                except ValueError:  colspan = 1
                rowspan = cell.get("rowspan", "1")
                try:                rowspan = int(rowspan) if rowspan.isdigit() else 1
                except ValueError:  rowspan = 1

                if colspan > 1 or rowspan > 1:
                    # Update ref_memory
                    for i in range(col_idx, col_idx + colspan):
                        for j in range(row_idx + 1, row_idx + rowspan):
                            ref_memory[(i, j)] = ref_counter
                    
                    # Append ref to row_data
                    ref_id = str(ref_counter)
                    ref_counter += 1
                    table[TempConstant.REFS.value][ref_id] = cell_data
                    
                    for _ in range(colspan):
                        row_data.append({"ref": ref_id})
                else:
                    row_data.append(cell_data)
                
                col_idx += colspan
            
            while (col_idx, row_idx) in ref_memory:
                ref_id = ref_memory[(col_idx, row_idx)]
                row_data.append({"ref": ref_id})
                del ref_memory[(col_idx, row_idx)]
                col_idx += 1
        
            table["table"].append(row_data)
            
            if first_row:
                table["columns"] = col_idx
                first_row = False
            num_rows += 1
            
        table["rows"] = num_rows

        return table
        
        
    def parseHtmlImage(self, element):
        """
        Parse a <figure> element to extract the image source and caption.

        Args:
            element (bs4.element.Tag): The <figure> element to parse.

        Returns:
            dict: Contains the extracted image source, caption text, and caption hyperlinks.
        """
        
        # Initialize the result dictionary
        result = {
            "image": None,
            "caption": {
                "text": "",
                TempConstant.HYPERLINKS.value: []
            }
        }

        # Extract the image source
        img_tag = element.find("img")
        if img_tag and img_tag.get("src"):
            result["image"] = f"https://en.wikipedia.org{img_tag['src']}"
        else:
            if element.name == "img" and element.get("src"):
                result["image"] = f"https://en.wikipedia.org{element['src']}"


        # Extract the caption
        figcaption = element.find("figcaption")
        if figcaption:
            caption_text, caption_hyperlinks = self.extractTextInDFS(figcaption, "", [])
            result["caption"]["text"] = caption_text
            result["caption"][TempConstant.HYPERLINKS.value] = caption_hyperlinks
            
        return result




    def loadHtmlDocuments(self):
        """
        Read and save all HTML files under self._html_documents_path as strings in self._html_documents.
        """
        
        self._html_documents = []
        
        if not os.path.exists(self._htmls_directory):
            raise FileNotFoundError(f"Path {self._htmls_directory} does not exist.")
        target_files = os.listdir(self._htmls_directory)
        
        target_files.sort()
                
        for file_name in tqdm(target_files):
            file_path = os.path.join(self._htmls_directory, file_name)

            if os.path.isfile(file_path) and file_name.endswith('.json'):
                with open(file_path, 'r') as json_file:
                    json_object = json.load(json_file)
                    html_string = json_object["html"]
                    self._html_documents.append([file_name, html_string])
        
        print("\tLoaded " + str(len(self._html_documents)) + " HTML documents.")

        
        

    def loadParsedHtmlDocuments(self):
        """
        Load parsed HTML documents from the serialized file.
        """
        
        self._parsed_htmls = {}
        
        if not os.path.exists(self._parsed_documents_path):
            raise FileNotFoundError(f"Path {self._parsed_documents_path} does not exist.")
        
        target_files = os.listdir(self._parsed_documents_path)
        target_files = [file for file in target_files if file.endswith(".json")]
        target_files.sort()
        
        for file_name in tqdm(target_files):
            file_path = os.path.join(self._parsed_documents_path, file_name)
            with open(file_path, "r") as f:
                self._parsed_htmls[file_name] = json.load(f)
                
        print("\tLoaded " + str(len(self._parsed_htmls)) + " parsed HTML documents.")
                
        return
    
    

    #######################################################
    #   ____  _                    _   _  _    _  _       #
    #  / ___|| |  __ _  ___  ___  | | | || |_ (_)| | ___  #
    # | |    | | / _` |/ __|/ __| | | | || __|| || |/ __| #
    # | |___ | || (_| |\__ \\__ \ | |_| || |_ | || |\__ \ #
    #  \____||_| \__,_||___/|___/  \___/  \__||_||_||___/ #
    #######################################################                    


    def extractTextInDFS(self, node, accumulated_text, accumulated_hyperlinks):
        """
        A helper function for recursively parsing HTML elements to extract text and hyperlinks.

        Args:
            node (bs4.element.Tag or str): The current HTML node to parse.

        Returns:
            dict: Contains the extracted text and hyperlinks.
        """

        if node.name == "a":
            
            # Handle hyperlinks
            start_index = len(accumulated_text)
            link_text = node.get_text().strip()
            link_text = " ".join(link_text.split())
            end_index = start_index + len(link_text)

            accumulated_hyperlinks.append({
                "text": link_text,
                TempConstant.HYPERLINK.value: node.get("href"),
                "start": start_index,
                "end": end_index
            })
            accumulated_text += " " + link_text
            
        elif isinstance(node, str):
            accumulated_text += " " + " ".join(node.strip().split())
            
        elif node.name in ["sup", "style"]: 
            pass
        
        else:
            for child in node.children:
                child_text, child_hyperlinks = self.extractTextInDFS(child, accumulated_text, accumulated_hyperlinks)
                accumulated_text = child_text
                # accumulated_hyperlinks.extend(child_hyperlinks)

        return accumulated_text, accumulated_hyperlinks













    
    
if __name__ == "__main__":
    main()