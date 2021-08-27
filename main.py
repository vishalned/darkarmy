import streamlit as st
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

import backend.inference

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Dark Army - Your AI tutor ...')
user_input = st.text_area("Enter the Question", '')
print(user_input)


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
text = st.text_input("Say what ?")
tts_button = Button(label="Speak", width=100)

tts_button.js_on_event("button_click", CustomJS(code=f"""
    var u = new SpeechSynthesisUtterance();
    u.text = "{text}";
    u.lang = 'en-US';

    speechSynthesis.speak(u);
    """))

st.bokeh_chart(tts_button)

