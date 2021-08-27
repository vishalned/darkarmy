import os
import toml

from backend.post_processing import *
#from post_processing import *

cfg_file = ("config.toml")
cfg = toml.load(open(cfg_file), _dict=dict)

finder = get_finder(os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset"), abstracts_only=False)

def get_results(question, finder=finder, top_k_retriever=cfg["top_k_retriever"], top_k_reader=cfg["top_k_reader"], candidate_doc_ids=None):
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

# DEMO
question = "What is AI all about?"
results = get_results(question, finder)
html_string = generate_html(question, results)