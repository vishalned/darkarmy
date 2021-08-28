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

n_results = st.slider('Number results?', 1, 5, 3)

question = "Whats AI?"

results = backend.inference.get_results(question,top_k_reader=n_results)
html_string = backend.inference.generate_html(question, results)

#html_str = """<div><div><h1>What is the incubation period of SARS-CoV-2?</h1></div><div><a href="https://cord-19.apps.allenai.org/paper/81191a43e52981527457fa8003a0e2cc6b7b6442" target="_blank">First known person-to-person transmission of severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) in the USA</a></div><div><b>I Ghinai, TD Mcpherson, JC Hunter, HL Kirking, D Christiansen, K Joshi, R Rubin, S Morales-Estrada, SR Black, M Pacilli, MJ Fricchione, RK Chugh, KA Walblay, S Ahmed, WC Stoecker, NF Hasan, DP Burdsall, HE Reese, M Wallace, C Wang, D Moeller, J Korpics, SA Novosad, I Benowitz, MW Jacobs, VS Dasari, MT Patel, J Kauerauf, M Charles, NO Ezike, V Chu, CM Midgley, MA Rolfes, SI Gerber, X Lu, S Lindstrom, JR Verani, JE </b></div><div><b style="color: grey;">Discussion</b></div><div><p><span>rtPCR might not be sufficient to definitively rule out infection over a </span><span style="background-color: #DCDCDC; border-radius: 5px; padding: 5px;">14-day </span><span>incubation period, and only a convenience sample of a minority of healt</span></p></div><hr /><div><a href="https://cord-19.apps.allenai.org/paper/14083b334f48819d6a42a2ed917adfa200c59df4" target="_blank">Current State and Predicting Future Scenario of Highly Infected Nations for COVID-19 Pandemic</a></div><div><b>NL Patil</b></div><div><b style="color: grey;">Introduction</b></div><div><p><span>e contacts (within about 6 feet) [1] and has mean incubation period of </span><span style="background-color: #DCDCDC; border-radius: 5px; padding: 5px;">6.4 days </span><span>with range of 2.1 to 11.1 days [2] . Asymptotic spread has made detect</span></p></div><hr /><div><a href="https://cord-19.apps.allenai.org/paper/6314943df23e3c79ded21bd2d5ced6bfd35be893" target="_blank">[No title available]</a></div><div><b /></div><div><b style="color: grey;">Clinical disease course of COVID-19</b></div><div><p><span>The official incubation period for SARS-CoV-2 is </span><span style="background-color: #DCDCDC; border-radius: 5px; padding: 5px;">2-14 days </span><span>and therefore 14 days is the chosen cut-off for self-quarantine. Guan = 138) showed that th</span></p></div><hr /><div><a href="https://cord-19.apps.allenai.org/paper/PMC7138423" target="_blank">A Comprehensive Literature Review on the Clinical Presentation, and Management of the Pandemic Coronavirus Disease 2019 (COVID-19)</a></div><div><b>A Muacevic, JR Adler, P Kakodkar, N Kaka, M Baig</b></div><div><b style="color: grey;">Review</b></div><div><p><span>The official incubation period for SARS-CoV-2 is </span><span style="background-color: #DCDCDC; border-radius: 5px; padding: 5px;">2-14 days </span><span>and therefore 14 days is the chosen cut-off for self-quarantine. Guan et al. (n = 1324) est</span></p></div><hr /><div><a href="https://cord-19.apps.allenai.org/paper/PMC7096724" target="_blank">Prevalence of Underlying Diseases in Hospitalized Patients with COVID-19: a Systematic Review and Meta-Analysis</a></div><div><b>A Emami, F Javanmardi, N Pirbonyeh, A Akbari</b></div><div><b style="color: grey;">Discussion</b></div><div><p><span>as follows: diagnosis, mode of transmission, long incubation period (</span><span style="background-color: #DCDCDC; border-radius: 5px; padding: 5px;">3 to 14 days</span><span>), predicting the number of infected cases in the community, and insu</span></p></div><hr /></div><div><div>"""
html_out = st.components.v1.html(html=html_string,height=300,scrolling=True)