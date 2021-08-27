import logging
import json
import os
import toml
import tqdm
import re
from xml.etree import ElementTree as ET
from pathlib import Path

from haystack import Finder
from haystack.database.sql import SQLDocumentStore
from haystack.retriever.tfidf import TfidfRetriever
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers

cfg_file = ("config.toml")
cfg = toml.load(open(cfg_file), _dict=dict)

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
        raise IOError("The data directory was not found")
    paths = [os.path.join(l[0], p) for l in os.walk(data_directory) for p in
        l[-1] if p[-5:] == ".json"]
    for path in tqdm.tqdm(paths):
        data = json.load(open(path, "r"))
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
    for author in full_text["metadata"]["authors"]:
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
    if "abstract" in full_text.keys():
        sections += full_text["abstract"]
    if not abstracts_only:
        if "body_text" in full_text.keys():
            sections += full_text["body_text"]
    for section in sections:
        yield section["section"], section["text"]


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
    document_store = SQLDocumentStore(url=cfg["db_path"])
    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(
        model_name_or_path=cfg["model_path"], 
        use_gpu=True)
    finder = Finder(reader, retriever)
    return finder

class Result(object):
    """
    A data structure to store question answering results (at document-section level).
    """
    def __init__(self, title, url, authors, section_title, text, spans):
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
        self.url = url
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
            url = "https://cord-19.apps.allenai.org/paper/%s" % paper_id
            spans = [{
                "start" : prediction["offset_start"], 
                "end" : prediction["offset_end"], 
            }]
            result = Result(
                title, 
                url, 
                authors, 
                section_title, 
                prediction["context"], 
                spans, 
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
    a = ET.SubElement(div, "a", href=result.url, target="_blank")
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