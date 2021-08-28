import logging
from haystack import Finder
from haystack.database.sql import SQLDocumentStore
from haystack.retriever.tfidf import TfidfRetriever
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers
import json
import os
from pathlib import Path
import re
import tqdm
from xml.etree import ElementTree as ET
from zipfile import ZipFile

logger = logging.getLogger("haystack")
logger.setLevel(logging.ERROR)

def get_full_texts(data_directory):
    """
    Returns a generator that yields dictionaries for each full text.
   
    :param data_directory: The directory where the data is.
    :type data_directory: str
    ...
    :returns: A generator of dict objects.
    :rtype: generator
    """
    if not os.path.isdir(data_directory):
        raise IOError("""The data directory was not found. Download the data from:
            "https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/download
            and place it under a directory called '%s'.""" % data_directory)
    print("high")
    paths = [os.path.join(l[0], p) for l in os.walk(data_directory) for p in
        l[-1] if p[-5:] == ".json"]
    print("low")
    for path in tqdm.tqdm(paths):
#         print(path)
        data = json.load(open(path, "r"))
#         print('###########################')
#         print(data.keys())
#         print(data)
        yield data
        
def get_authors(full_text):
    """
    Returns a list of formatted authors extracted from the full text dict object.
    
    :param full_text: A full text dict as returned by 'get_full_texts'.
    :type full_text: dict
    ...
    :returns: A list of author strings.
    :rtype: list
    """
    authors = []
    for author in full_text["authors"]:
        author = "".join([c for c in author["first"][:1]] + author["middle"]) + " " + author["last"]
        authors.append(author)
    return authors

def get_sections(full_text, abstracts_only=False):
    """
    Returns a generator that yields (section name, text) tuples for each
    section in a full text dictionary.
   
    :param full_text: A full text dict as returned by 'get_full_texts'.
    :type full_text: dict
    :param abstracts_only: If True, only abstracts are indexed. Defaults to False.
    :type abstracts_only: bool
    ...
    :returns: A generator of (section, text) tuples.
    :rtype: generator
    """
    
    sections = []
#     print(full_text['abstract'])
#     print(full_text.keys())
#     print(full_text['pdf_parse']['body_text'][0]['section'])
    if "abstract" in full_text.keys():
#         sections += [full_text["abstract"]]
        sections.append({"section":full_text['pdf_parse']['abstract'][0]['section'], 'text':full_text['abstract']})
    if not abstracts_only:
        if "body_text" in full_text['pdf_parse'].keys():
#             sections += [full_text["body_text"]]
            sections.append({"section":full_text['pdf_parse']['body_text'][0]['section'], 'text':full_text['pdf_parse']['body_text'][0]['text']})
#     print(sections)
    
    for section in sections:
#         print(i, section)
        yield section['section'], section['text']
#     print(sections)
#     yield 0, full_text['abstract']

def write_documents_to_db(document_store, document_dir, clean_func=None, only_empty_db=False, 
    split_paragraphs=False, abstracts_only=False):
    """
    Writes documents to a sqlite database.
    
    :param document_store: A SQLDocumentStore to store section data in.
    :type document_store: SQLDocumentStore.
    :param document_dir: The directory where the documents are.
    :type document_dir: str
    :param abstracts_only: Only index abstracts.
    :type abstracts_only: bool
    """
    # check if db has already docs
    if only_empty_db:
        n_docs = document_store.get_document_count()
        print(n_docs)
        if n_docs > 0:
            print(f"Skip writing documents since DB already contains {n_docs} docs ...  "
                        "(Disable `only_empty_db`, if you want to add docs anyway.)")
            return None

    # read and add docs
    docs_to_index = []
    count = 1
    for full_text in get_full_texts(document_dir):
#         print(full_text.keys())
        title = full_text["title"].strip()
        if title == "":
            title = "[No title available]"
        authors = ", ".join(get_authors(full_text))
        for section, text in get_sections(full_text, abstracts_only=abstracts_only):
            
            # Skip if not talking about SARS-CoV-2 or COVID-19
#             if len(re.findall("(SARS.CoV.2|COVID.19|2019.nCoV)", text, flags=re.IGNORECASE)) == 0:
#                 continue
#             print(text)
            docs_to_index.append({
                "text": text, 
                "id": count, 
                "name" : "%s|||%s|||%s|||%s" % (title, section, authors, full_text["paper_id"]), 
            })
            count += 1
#             print(len(docs_to_index))
            if len(docs_to_index) == 500:
                document_store.write_documents(docs_to_index)                
                docs_to_index = []
    print('outaside for loop')
    if len(docs_to_index) > 0:
        document_store.write_documents(docs_to_index)
    print(f"Wrote {count} docs to DB")

def get_finder(data_directory, abstracts_only=False):
    """
    Returns a haystack Finder object with a TfIdfRetriever and a FARMReader based on 
    Transformers.
    
    :param data_directory: The directory where the data is located.
    :type data_directory: str
    :param abstracts_only: If true, only the abstracts section is indexed.
    :type abstracts_only: bool
    ...
    :return: A haystack Finder.
    :rtype: Finder
    """
    document_store = SQLDocumentStore(url="sqlite:///backend/qa.db")
    write_documents_to_db(document_store=document_store, 
        document_dir=data_directory, only_empty_db=True, 
        abstracts_only=abstracts_only)
    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(
        model_name_or_path="backend/my_model", 
        use_gpu=True)
        #model_name_or_path="gdario/biobert_bioasq", 
        #use_gpu=False)
    finder = Finder(reader, retriever)
    return finder

class Result(object):
    """
    A data structure to store question answering results (at document-section level).
    """
    def __init__(self, title, authors, section_title, text, spans):
        """
        See Result class.
        
        :param title: The title of the document.
        :type title: str
        :param url: URL of the document.
        :type url: str
        :param authors: A string with all the authors in the document.
        :type authors: str
        :param section_title: The title of the section.
        :type section_title: str
        :param text: The text of the section.
        :type text: str
        :param spans: A list of span dicts.
        """
        self.title = title
#         self.url = url
        self.authors = authors
        self.section_title = section_title
        self.text = text
        self.spans = spans

def get_results(finder, top_k_retriever, top_k_reader, candidate_doc_ids, question):
    """
    Builds a list of Result objects for a given question.
    
    :param finder: A haystack Finder instance.
    :type finder: Finder
    :param top_k_retriever: The number of top document-sections obtained by the retriever.
    :type top_k_retriever: int
    :param top_k_reader: The number of top answers obtained by the reader.
    :type top_k_reader: int
    :param candidate_doc_ids: A list of doc ids to filter on.
    :type candidate_doc_ids: list
    :param question: A question to be answered.
    :type question: str
    ...
    :return: A list of Result instances.
    :rtype: list
    """
    paragraphs, meta_data = finder.retriever.retrieve(question, top_k=top_k_retriever, candidate_doc_ids=candidate_doc_ids)
    results = []

    if len(paragraphs) > 0:

        # 3) Apply reader to get granular answer(s)
        len_chars = sum([len (p) for p in paragraphs])
        predictions = finder.reader.predict(question=question,
            paragraphs=paragraphs,
            meta_data_paragraphs=meta_data,
            top_k=top_k_reader)

        # Add corresponding document_name if an answer contains the document_id (only supported in FARMReader)
        for prediction in predictions["answers"]:
            document = finder.retriever.document_store.get_document_by_id(prediction["document_id"])
            title, section_title, authors, paper_id = document["name"].split("|||")
#             url = "https://cord-19.apps.allenai.org/paper/%s" % paper_id
            spans = [{
                "start" : prediction["offset_start"], 
                "end" : prediction["offset_end"], 
            }]
            result = Result(
                title, 
                 
                authors, 
                section_title, 
                prediction["context"], 
                spans
                 
            )
            results.append(result)
    
    return results

def add_search_result_element(container, result):
    """
    Adds a search result node to an html container.
    
    :param container: An ElementTree node.
    :type container: ElementTree
    :param result: A result to be added.
    :type result: Result
    """
    # Title
    div = ET.SubElement(container, "div")
    a = ET.SubElement(div, "a", href='#', target="_blank")
    a.text = result.title

    # Authors
    div = ET.SubElement(container, "div")
    b = ET.SubElement(div, "b")
    b.text = result.authors

    # Section Title
    div = ET.SubElement(container, "div")
    b = ET.SubElement(div, "b", style="color: grey;")
    b.text = result.section_title
    
    # Snippet
    cursor = 0
    for span in result.spans:
        div = ET.SubElement(container, "div")
        p = ET.SubElement(div, "p")
        span_element = ET.SubElement(p, "span")
        span_element.text = result.text[:span["start"]]
        span_element = ET.SubElement(p, "span", style="background-color: #DCDCDC; border-radius: 5px; padding: 5px;")
        span_element.text = result.text[span["start"]:span["end"]]
        cursor = span["end"]
    if cursor < len(result.text):
        span_element = ET.SubElement(p, "span")
        span_element.text = result.text[cursor:]

def generate_html(question, results):
    """
    Generate HTML to display for a given question and a list of results.
    
    :param question: The question that was asked.
    :type question: str
    :param results: A list of Result instances.
    :type results: list
    ...
    :return: HTML content showing the question and the results (with links).
    :rtype: str
    """
    container = ET.Element("div")
    
    # Add question
    div = ET.SubElement(container, "div")
    h = ET.SubElement(div, "h1")
    h.text = question
    
    # Add answers
    for result in results:
        add_search_result_element(container, result)
        ET.SubElement(container, "hr")
    html = str(ET.tostring(container))[2:-1]
    return html
    