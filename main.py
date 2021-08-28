import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

from backend.inference import * 

#@st.cache(allow_output_mutation = True, suppress_st_warning=True)
def finder_func():
    return get_finder("backend/dataset", abstracts_only=False)

finder = finder_func()

#user input
st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Dark Army - Your AI tutor ...')
user_input = st.text_area("Enter the Question", '')

#speech to text
stt_button = Button(label="Speak" ,width = 100)

stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
 
    recognition.onresult = function (e) {
        var value = "";
        for (var i = e.resultIndex; i < e.results.length; ++i) {
            if (e.results[i].isFinal) {
                value += e.results[i][0].transcript;
            }
        }
        if ( value != "") {
            document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
        }
    }
    recognition.start();
    """))

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=50,
    debounce_time=0)

if result:
    if "GET_TEXT" in result:
        st.write(result.get("GET_TEXT"))

#text to speech
# text = st.text_input("Say what ?")
# tts_button = Button(label="Speak", width=100)


# tts_button.js_on_event("button_click", CustomJS(code=f"""
#     var u = new SpeechSynthesisUtterance();
#     u.text = "{text}";
#     u.lang = 'en-US';

#     speechSynthesis.speak(u);
#     """))

# st.bokeh_chart(tts_button)

n_results = st.slider('Number results?', 1, 5, 3)
run_button = st.button('run')

if run_button:
    results = get_results(finder=finder,candidate_doc_ids=None, top_k_retriever=25, top_k_reader=n_results, question=user_input)
    html_string = generate_html(user_input, results)
    html_out = st.components.v1.html(html=html_string,height=500,scrolling=True)